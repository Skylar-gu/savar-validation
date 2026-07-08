"""
R5 rung — rollout (multi-step autoregressive) fine-tuning of the MeshGNN forecaster.

Sibling to train/gnn/gnn_forecaster.py; imports the model + dataset from it so the
architecture is never duplicated (K=3, HIDDEN=256, N_MP=4, gcn message passing —
IDENTICAL to the base forecaster; only the *training objective* changes).

Why the GNN and not cnn_forecaster.py
-------------------------------------
checkpoints/overlap02/best.pt is a MeshGNN state dict (encoder / layers[i].{mlp,norm}
/ decoder). The CNN in train/cnn/cnn_forecaster.py (conv3d / res / head + BatchNorm)
cannot load it. R5 must fine-tune the *same* frozen-comparable model that R1 produced,
so this script builds on gnn_forecaster.MeshGNN. (Design rationale to record:
R5-on-overlap vs R1-on-overlap isolates the rollout-training effect on one world.)

What changes vs the base single-step trainer
---------------------------------------------
  * warm-start from an existing checkpoint (default checkpoints/overlap02/best.pt),
  * loss = mean over R autoregressive steps of MSE(pred_r, true_future_r): the model
    is unrolled R steps, each prediction fed back as the newest frame of the K-window,
    gradients flowing through the whole chain (BPTT — GraphCast's curriculum),
  * a CURRICULUM that ramps the horizon R over epochs (e.g. 1 -> 4 -> 8), configurable.

Everything is env-parameterised in the same style as gnn_forecaster.py.

  GNN_ROLLOUT_INIT_FROM   base checkpoint to fine-tune          (checkpoints/overlap02/best.pt)
  GNN_ROLLOUT_CKPT_DIR    output dir for best.pt                (checkpoints/overlap02_rollout)
  GNN_ROLLOUT_SPLIT_DIR   split dir (train/ val/)               (data/splits_overlap02)
  GNN_ROLLOUT_SCHEDULE    horizon curriculum "RxE,RxE,..."      ("1x2,4x4,8x6" -> 12 epochs)
  GNN_ROLLOUT_EVAL_R      val rollout horizon (comparable ep-to-ep, default = max R)
  GNN_ROLLOUT_LR          fine-tune LR (lower than base 3e-4)   (1e-4)
  GNN_ROLLOUT_BATCH       batch size (rollout is memory-heavier)(16)
  GNN_ROLLOUT_SEED        determinism seed                      (0)
  GNN_ROLLOUT_WORKERS     dataloader workers                    (4)

  ckpt saved: {epoch, model_state, val_rmse (1-step, comparable to R1), val_rollout_rmse,
               eval_r, schedule}; best selected on val_rollout_rmse.

Smoke test (CPU, synthetic tensors — no data, no training):
    CUDA_VISIBLE_DEVICES="" python train/gnn/rollout_finetune.py --smoke
"""

import os, sys, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(line_buffering=True)

# import the model + dataset from the base forecaster (no arch duplication).
# Same-dir import is the normal path (this script lives beside gnn_forecaster.py).
# SAVAR_REPO_ROOT is an optional fallback for running from a worktree that does not
# yet contain the base module (e.g. a stale scaffolding worktree).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.environ.get("SAVAR_REPO_ROOT")
if _repo_root:
    sys.path.append(os.path.join(_repo_root, "train", "gnn"))
from gnn_forecaster import MeshGNN, MultiRealisationDataset, forecast_corr, K, HIDDEN, N_MP

# ── config (mirrors gnn_forecaster.py's env style) ────────────────────────────
INIT_FROM  = os.environ.get("GNN_ROLLOUT_INIT_FROM", "checkpoints/overlap02/best.pt")
CKPT_DIR   = os.environ.get("GNN_ROLLOUT_CKPT_DIR",  "checkpoints/overlap02_rollout")
SPLIT_DIR  = os.environ.get("GNN_ROLLOUT_SPLIT_DIR", os.path.join("data", "splits_overlap02"))
SCHEDULE   = os.environ.get("GNN_ROLLOUT_SCHEDULE",  "1x2,4x4,8x6")
LR         = float(os.environ.get("GNN_ROLLOUT_LR", 1e-4))
BATCH_SIZE = int(os.environ.get("GNN_ROLLOUT_BATCH", 16))
SEED       = int(os.environ.get("GNN_ROLLOUT_SEED", 0))
N_WORKERS  = int(os.environ.get("GNN_ROLLOUT_WORKERS", 4))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_schedule(spec):
    """"1x2,4x4,8x6" -> [1,1,4,4,4,4,8,8,8,8,8,8]  (horizon R per epoch)."""
    horizons = []
    for seg in spec.split(","):
        seg = seg.strip()
        if not seg:
            continue
        r_str, _, e_str = seg.partition("x")
        r, e = int(r_str), int(e_str) if e_str else 1
        horizons.extend([r] * e)
    if not horizons:
        raise ValueError(f"empty rollout schedule from spec {spec!r}")
    return horizons


# ── dataset: window + R_max true future frames ────────────────────────────────
class RolloutDataset(MultiRealisationDataset):
    """Same segment loading as the base dataset; each item is a k-frame input window
    plus the next `r_max` ground-truth frames (targets for the unrolled steps)."""
    def __init__(self, split_dir, k=K, r_max=1):
        super().__init__(split_dir, k)          # loads self.segs (T, ny, nx)
        self.r_max = r_max
        self.index = [(s, i) for s, seg in enumerate(self.segs)
                      for i in range(len(seg) - k - r_max + 1)]

    def __getitem__(self, idx):
        s, i = self.index[idx]
        seg = self.segs[s]
        x  = seg[i : i + self.k]                       # (k, ny, nx)
        ys = seg[i + self.k : i + self.k + self.r_max] # (r_max, ny, nx)
        return x, ys


# ── rollout loss ──────────────────────────────────────────────────────────────
def rollout_loss(model, x, ys, R):
    """Unroll the model R steps autoregressively, feeding each prediction back as the
    newest frame of the K-window; return the mean over the R steps of MSE vs the true
    future frames. Gradients flow through the full unrolled chain (BPTT).

      x  : (B, k, ny, nx)         input window
      ys : (B, r_max, ny, nx)     ground-truth future frames (r_max >= R)
    """
    window = x
    step_losses = []
    for step in range(R):
        pred = model(window)                 # (B, 1, ny, nx)
        tgt  = ys[:, step:step + 1]          # (B, 1, ny, nx)
        step_losses.append(F.mse_loss(pred, tgt))
        # feed prediction back: drop oldest frame, append prediction as newest
        window = torch.cat([window[:, 1:], pred], dim=1)   # (B, k, ny, nx)
    return torch.stack(step_losses).mean()


def run_rollout_epoch(model, loader, R, optimizer=None):
    training = optimizer is not None
    model.train(training)
    tot = n = 0.0
    with torch.set_grad_enabled(training):
        for x, ys in loader:
            x, ys = x.to(DEVICE), ys.to(DEVICE)
            loss = rollout_loss(model, x, ys, R)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = x.size(0)
            tot += loss.item() * bs
            n   += bs
    mse = tot / n
    return {"mse": mse, "rmse": mse ** 0.5}


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    horizons = parse_schedule(SCHEDULE)
    epochs   = len(horizons)
    r_max    = max(horizons)
    eval_r   = int(os.environ.get("GNN_ROLLOUT_EVAL_R", r_max))
    r_max    = max(r_max, eval_r)             # need enough future frames for val too

    train_ds = RolloutDataset(os.path.join(SPLIT_DIR, "train"), K, r_max)
    val_ds   = RolloutDataset(os.path.join(SPLIT_DIR, "val"),   K, r_max)
    ny, nx   = train_ds.segs[0].shape[1], train_ds.segs[0].shape[2]

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=N_WORKERS, pin_memory=True,
                              persistent_workers=N_WORKERS > 0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=N_WORKERS, pin_memory=True,
                              persistent_workers=N_WORKERS > 0)

    model = MeshGNN(ny=ny, nx=nx, k=K).to(DEVICE)
    if not (INIT_FROM and os.path.exists(INIT_FROM)):
        raise FileNotFoundError(
            f"rollout fine-tune requires a base checkpoint; GNN_ROLLOUT_INIT_FROM={INIT_FROM!r} "
            f"not found. Point it at checkpoints/overlap02/best.pt.")
    ck = torch.load(INIT_FROM, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: MeshGNN — R5 rollout fine-tune")
    print(f"  Device       : {DEVICE}")
    print(f"  Warm-start   : {INIT_FROM}  (epoch {ck.get('epoch')}, "
          f"val_rmse {ck.get('val_rmse'):.4f})")
    print(f"  Parameters   : {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  K / hidden   : {K} / {HIDDEN}   MP layers {N_MP}")
    print(f"  Split        : {SPLIT_DIR}")
    print(f"  Out ckpt dir : {CKPT_DIR}")
    print(f"  Schedule     : {SCHEDULE}  -> {epochs} epochs, horizons {horizons}")
    print(f"  Eval horizon : R={eval_r} (fixed, comparable epoch-to-epoch)")
    print(f"  LR / batch   : {LR:.1e} / {BATCH_SIZE}   seed {SEED}")
    print(f"  Train/Val    : {len(train_ds)} / {len(val_ds)} windows\n")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR / 20)

    best = float("inf"); history = []
    print(f"{'Ep':>4} {'R':>3} {'TrRMSE':>8} {'Vl1step':>8} {'VlRoll':>8} {'LR':>9}  Time")
    print("-" * 56)
    for ep in range(1, epochs + 1):
        R = horizons[ep - 1]
        t0 = time.time()
        tr    = run_rollout_epoch(model, train_loader, R, opt)
        vl1   = run_rollout_epoch(model, val_loader, 1)          # single-step (comparable to R1)
        vlR   = run_rollout_epoch(model, val_loader, eval_r)     # rollout horizon
        sched.step()

        row = {"epoch": ep, "R": R, "train_rmse": tr["rmse"],
               "val_rmse": vl1["rmse"], "val_rollout_rmse": vlR["rmse"],
               "lr": sched.get_last_lr()[0]}
        history.append(row)
        print(f"{ep:>4} {R:>3} {tr['rmse']:>8.4f} {vl1['rmse']:>8.4f} {vlR['rmse']:>8.4f} "
              f"{sched.get_last_lr()[0]:>9.2e}  {time.time()-t0:.1f}s")

        if vlR["rmse"] < best:
            best = vlR["rmse"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_rmse": vl1["rmse"], "val_rollout_rmse": best,
                        "eval_r": eval_r, "schedule": SCHEDULE},
                       os.path.join(CKPT_DIR, "best.pt"))

    np.save(os.path.join(CKPT_DIR, "history.npy"), history)
    print(f"\nBest val rollout RMSE (R={eval_r}) : {best:.4f}   ->  {CKPT_DIR}/best.pt")


# ── smoke test (CPU, synthetic) ────────────────────────────────────────────────
def smoke_test():
    torch.manual_seed(0)
    print("[smoke] R5 rollout fine-tune — CPU synthetic smoke test")
    print(f"[smoke] device={DEVICE}  cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    if DEVICE.type != "cpu":
        print("[smoke] WARNING: CUDA is visible; export CUDA_VISIBLE_DEVICES=\"\" for a CPU-only smoke test.")

    ny = nx = 50
    B, R = 2, 2
    model = MeshGNN(ny=ny, nx=nx, k=K).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] MeshGNN built: {n_params:,} params  (K={K}, hidden={HIDDEN}, MP={N_MP})")

    # tiny SYNTHETIC batch shaped like the real data
    x  = torch.randn(B, K, ny, nx, device=DEVICE)          # input window
    ys = torch.randn(B, R, ny, nx, device=DEVICE)          # R true future frames
    print(f"[smoke] synthetic x={tuple(x.shape)}  ys={tuple(ys.shape)}  horizon R={R}")

    # verify the autoregressive window feedback keeps shape (B, K, ny, nx)
    with torch.no_grad():
        pred0 = model(x)
        win1  = torch.cat([x[:, 1:], pred0], dim=1)
    assert pred0.shape == (B, 1, ny, nx), f"unexpected pred shape {tuple(pred0.shape)}"
    assert win1.shape  == (B, K, ny, nx), f"window feedback broke shape: {tuple(win1.shape)}"
    print(f"[smoke] step-0 pred={tuple(pred0.shape)} -> fed-back window={tuple(win1.shape)} OK")

    # forward + backward through the rollout loss
    model.train()
    loss = rollout_loss(model, x, ys, R)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    print(f"[smoke] rollout loss (mean over {R} steps) = {loss.item():.6f}  finite=OK")

    loss.backward()
    n_with_grad = sum(1 for p in model.parameters()
                      if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    n_total     = sum(1 for p in model.parameters() if p.requires_grad)
    total_gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert n_with_grad > 0, "no gradients flowed"
    assert np.isfinite(total_gnorm) and total_gnorm > 0, f"bad total grad norm {total_gnorm}"
    print(f"[smoke] gradients flow: {n_with_grad}/{n_total} params with nonzero finite grad; "
          f"total grad-norm = {total_gnorm:.4f}")
    print("[smoke] PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="CPU synthetic smoke test (no data, no training)")
    args = ap.parse_args()
    if args.smoke:
        smoke_test()
    else:
        os.makedirs(CKPT_DIR, exist_ok=True)
        main()
