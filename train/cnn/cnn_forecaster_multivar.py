"""
Phase 2 — Spatiotemporal CNN Forecaster (R3 MULTIVARIATE variant)

Channel-aware fork of cnn_forecaster.py. The ONLY architectural change is the
input/output channel dimension: the field is now C coupled observed channels
(temperature/pressure/humidity-like) instead of one scalar field, so Ŵ becomes
(channel × space).

Input  : (B, C, k, ny, nx)   — k past frames, C observed channels
Output : (B, C,    ny, nx)   — predicted next frame, all C channels

Everything else (3D conv → ResBlock×3 → 1×1 head, sizes, optimiser, schedule,
metric) is IDENTICAL to the single-channel forecaster; SpatioTemporalCNN already
parameterises in_ch, so C flows straight through. C is inferred from the data
(overridable with --channels).

Target parameter count: 2M–10M.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)   # flush every line even when piped

# ── config ───────────────────────────────────────────────────────────────────
K             = 3       # temporal window — k=3 per requirements §8
CHANNELS      = int(os.environ.get("MV_C", 2))   # C observed channels (inferred from data too)
BASE_CH       = 256     # channel width → ~3.2M params
BATCH_SIZE    = 64
LR            = 3e-4
EPOCHS        = 50
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR      = "checkpoints/multivar"
SPLIT_DIR     = os.path.join("data", "splits_multivar2")

os.makedirs(CKPT_DIR, exist_ok=True)

# ── dataset ──────────────────────────────────────────────────────────────────
class MultiRealisationDataset(Dataset):
    """
    Loads all realisations from a split directory.
    Each realisation contributes (T_split - k) sliding windows.
    obs files have shape (T_split, C, ny, nx) — the channel axis is kept.
    """
    def __init__(self, split_dir: str, k: int = K):
        import glob
        self.k    = k
        self.segs = []   # list of tensors (T_split, C, ny, nx)
        for path in sorted(glob.glob(os.path.join(split_dir, "realisation_*.npz"))):
            obs = np.load(path)["observations"]          # (T_split, C, ny, nx)
            t   = torch.from_numpy(obs).float()
            self.segs.append(t)
        self.C = self.segs[0].shape[1]
        # build flat index: (seg_idx, window_start)
        self.index = [(s, i) for s, seg in enumerate(self.segs)
                      for i in range(len(seg) - k)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, i  = self.index[idx]
        seg   = self.segs[s]
        x = seg[i : i + self.k].permute(1, 0, 2, 3)  # (C, k, ny, nx)
        y = seg[i + self.k]                           # (C, ny, nx)
        return x, y


# ── building blocks ──────────────────────────────────────────────────────────
class ResBlock2D(nn.Module):
    """Pre-norm residual block with optional channel projection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class SpatioTemporalCNN(nn.Module):
    """
    3D Conv to collapse temporal window  →  3 × ResBlock2D  →  1×1 head.
    in_ch = C observed channels (was 1 in the single-field forecaster).
    """
    def __init__(self, ny: int, nx: int, k: int = K,
                 in_ch: int = CHANNELS, base_ch: int = BASE_CH):
        super().__init__()

        # 3D conv: (B, C, k, ny, nx) → (B, base_ch//2, 1, ny, nx)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, base_ch // 2,
                      kernel_size=(k, 3, 3),
                      padding=(0, 1, 1),
                      bias=False),
            nn.BatchNorm3d(base_ch // 2),
            nn.GELU(),
        )

        # after squeeze: (B, base_ch//2, ny, nx) → residual blocks
        self.res1 = ResBlock2D(base_ch // 2, base_ch)
        self.res2 = ResBlock2D(base_ch,      base_ch)
        self.res3 = ResBlock2D(base_ch,      base_ch)

        # prediction head: channel → C fields
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(base_ch // 4, in_ch, 1),
        )

    def forward(self, x):
        # x: (B, C, k, ny, nx)
        h = self.conv3d(x)          # (B, base_ch//2, 1, ny, nx)
        h = h.squeeze(2)            # (B, base_ch//2,    ny, nx)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return self.head(h)         # (B, C, ny, nx)


# ── metrics ──────────────────────────────────────────────────────────────────
def forecast_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation averaged over batch (over all C·ny·nx elements)."""
    p = pred.flatten(1)
    t = target.flatten(1)
    p = p - p.mean(1, keepdim=True)
    t = t - t.mean(1, keepdim=True)
    num = (p * t).sum(1)
    den = p.norm(dim=1) * t.norm(dim=1) + 1e-8
    return (num / den).mean().item()


# ── train / eval loops ────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_mse = total_corr = n = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = F.mse_loss(pred, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = x.size(0)
            total_mse  += loss.item() * bs
            total_corr += forecast_corr(pred, y) * bs
            n          += bs

    mse  = total_mse  / n
    rmse = mse ** 0.5
    corr = total_corr / n
    return {"mse": mse, "rmse": rmse, "corr": corr}


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    train_ds = MultiRealisationDataset(os.path.join(SPLIT_DIR, "train"), K)
    val_ds   = MultiRealisationDataset(os.path.join(SPLIT_DIR, "val"),   K)

    # infer channel + spatial dims from first segment: (T, C, ny, nx)
    C  = train_ds.segs[0].shape[1]
    ny = train_ds.segs[0].shape[2]
    nx = train_ds.segs[0].shape[3]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = SpatioTemporalCNN(ny=ny, nx=nx, k=K, in_ch=C, base_ch=BASE_CH).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SpatioTemporalCNN (multivariate)")
    print(f"  Device     : {DEVICE}")
    print(f"  Parameters : {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Window k   : {K}")
    print(f"  Channels C : {C}")
    print(f"  Base ch    : {BASE_CH}")
    print(f"  Train set  : {len(train_ds)} windows")
    print(f"  Val set    : {len(val_ds)} windows\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 20)

    best_val_rmse = float("inf")
    history = []

    print(f"{'Epoch':>6} {'Train MSE':>10} {'Train RMSE':>11} "
          f"{'Val MSE':>9} {'Val RMSE':>10} {'Val Corr':>9}  {'LR':>8}  Time")
    print("-" * 85)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr = run_epoch(model, train_loader, optimizer)
        vl = run_epoch(model, val_loader)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "train_mse": tr["mse"], "train_rmse": tr["rmse"],
            "val_mse":   vl["mse"], "val_rmse":   vl["rmse"],
            "val_corr":  vl["corr"], "lr": lr_now,
        }
        history.append(row)

        print(f"{epoch:>6}   {tr['mse']:>9.4f}   {tr['rmse']:>10.4f}   "
              f"{vl['mse']:>8.4f}   {vl['rmse']:>9.4f}   "
              f"{vl['corr']:>8.4f}  {lr_now:>8.2e}  {elapsed:.1f}s")

        # save checkpoint every 10 epochs and on best val RMSE
        if vl["rmse"] < best_val_rmse:
            best_val_rmse = vl["rmse"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_rmse": best_val_rmse,
                "channels": C,
            }, os.path.join(CKPT_DIR, "best.pt"))

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "history": history,
                "channels": C,
            }, os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt"))

    np.save(os.path.join(CKPT_DIR, "history.npy"), history)
    print(f"\nBest val RMSE : {best_val_rmse:.4f}")
    print(f"Checkpoints   → {CKPT_DIR}/")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--split-dir", default=None,
                     help="Override split dir (default data/splits_multivar<C>)")
    _ap.add_argument("--ckpt-dir", default=None,
                     help="Override checkpoint dir (default checkpoints/multivar)")
    _ap.add_argument("--channels", type=int, default=None,
                     help="C observed channels (default: inferred from data)")
    _a = _ap.parse_args()
    if _a.split_dir:
        SPLIT_DIR = _a.split_dir
    if _a.ckpt_dir:
        CKPT_DIR  = _a.ckpt_dir
    if _a.channels:
        CHANNELS = _a.channels
    os.makedirs(CKPT_DIR, exist_ok=True)
    main()
