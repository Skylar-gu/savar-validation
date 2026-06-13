"""
Phase 7.2b — Per-mode TopK SAEs.

Why per-mode instead of one shared SAE
---------------------------------------
The mode-weighted activations live in a 256-dim space whose variance is almost
entirely explained by two principal components (PC0 ≈ 86%, PC1 ≈ 12%).  All
eight mode-weighted projections share the same dominant 'global activity'
direction, so a single SAE trained on all modes' mixed samples learns to
represent that shared direction and cannot distinguish modes.

Training one SAE per mode forces each SAE to find sparse structure *within*
that mode's activation cloud.  Some features will align with the optimal Z_j
direction in the PC0-PC1 subspace, where the theoretical ceiling is:

  X1 ≈ 0.57,  X3 ≈ 0.58,  X6 ≈ 0.58
  X0 ≈ 0.48,  X2 ≈ 0.47,  X4 ≈ 0.45,  X7 ≈ 0.47
  X5 ≈ 0.36   (limited by weak mode signal)

Alignment threshold is set to THRESH = 0.35 (below which a feature is noise).

Outputs (written to sae_data/)
-------------------------------
  sae_mode_{j}.pt   checkpoint for mode j (j = 0..7)
  sae_per_mode_history.npy   per-epoch logs for all modes
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--dy005", action="store_true")
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--deseason", action="store_true",
                 help="Use ensemble-mean-deseasonalized activations (sae_data_*_deseason/)")
_args = _ap.parse_args()

if _args.diurnal:
    DATA_DIR = Path("sae_data_diurnal")
elif _args.dy005:
    DATA_DIR = Path("sae_data_dy005")
else:
    DATA_DIR = Path("sae_data")
if _args.deseason:
    DATA_DIR = Path(str(DATA_DIR) + "_deseason")
OUT_DIR  = DATA_DIR

# ── hyperparameters ───────────────────────────────────────────────────────────

INPUT_DIM  = 256
N_FEATURES = 512    # 2× expansion
K_TOPK     = 25
LR         = 1e-3
EPOCHS     = 60
BATCH_SIZE = 256

RESAMPLE_INTERVAL = 1000   # steps between dead-feature checks (smaller: less data per mode)
DEAD_WINDOW       = 500    # steps of inactivity → feature considered dead

VAL_FRAC   = 0.15          # last 15% of realisations held out
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_MODES    = 8


# ── model ─────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, n_features: int, k: int):
        super().__init__()
        self.k          = k
        self.n_features = n_features
        self.encoder    = nn.Linear(input_dim, n_features, bias=True)
        self.decoder    = nn.Linear(n_features, input_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.decoder.weight, std=1.0 / INPUT_DIM ** 0.5)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
            self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre  = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    def forward(self, x: torch.Tensor):
        acts  = self.encode(x)
        recon = self.decoder(acts)
        return acts, recon

    @torch.no_grad()
    def normalise_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


# ── load activations ──────────────────────────────────────────────────────────

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, 497, 256)
n_real, n_modes_total, t_eff, d = acts_full.shape
assert d == INPUT_DIM

n_val   = max(1, int(n_real * VAL_FRAC))
n_train = n_real - n_val

all_history = {}


# ── train one SAE per mode ────────────────────────────────────────────────────

for j in range(N_MODES):

    print(f"\n{'═'*60}")
    print(f"  Mode X{j}  — training SAE")
    print(f"{'═'*60}")

    # Mode j's activations: (100, 497, 256) → split → flatten
    acts_j = acts_full[:, j, :, :]   # (100, 497, 256)

    train_j = acts_j[:n_train].reshape(-1, d).astype(np.float32)  # (~42K, 256)
    val_j   = acts_j[n_train:].reshape(-1, d).astype(np.float32)  # (~7K,  256)

    # Normalise (fit on train only)
    mean_j = train_j.mean(0)
    std_j  = train_j.std(0) + 1e-8

    def norm(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((x - mean_j) / std_j).to(DEVICE)

    X_train = norm(train_j)
    X_val   = norm(val_j)

    print(f"  train={len(X_train):,}  val={len(X_val):,}  "
          f"mean≈{X_train.mean().item():.3f}  std≈{X_train.std().item():.3f}")

    sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

    last_fired  = torch.zeros(N_FEATURES, dtype=torch.long, device=DEVICE)
    global_step = 0

    def resample_dead(dead_mask: torch.Tensor) -> int:
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        sample_idx = torch.randperm(len(X_train), device=DEVICE)[:min(2048, len(X_train))]
        X_s = X_train[sample_idx]
        with torch.no_grad():
            _, recon = sae(X_s)
            loss_s = F.mse_loss(recon, X_s, reduction="none").mean(dim=1)
            residuals = X_s - recon
        order     = loss_s.argsort(descending=True)
        residuals = residuals[order]
        if len(residuals) < n_dead:
            reps = (n_dead + len(residuals) - 1) // len(residuals)
            residuals = residuals.repeat(reps, 1)[:n_dead]
        else:
            residuals = residuals[:n_dead]
        res_norm = F.normalize(residuals, dim=1)
        with torch.no_grad():
            sae.encoder.weight.data[dead_idx] = res_norm
            sae.decoder.weight.data[:, dead_idx] = res_norm.T
            sae.encoder.bias.data[dead_idx] = 0.0
        enc_state = optimizer.state.get(sae.encoder.weight)
        if enc_state:
            enc_state["exp_avg"][dead_idx]    = 0.0
            enc_state["exp_avg_sq"][dead_idx] = 0.0
        dec_state = optimizer.state.get(sae.decoder.weight)
        if dec_state:
            dec_state["exp_avg"][:, dead_idx]    = 0.0
            dec_state["exp_avg_sq"][:, dead_idx] = 0.0
        return int(n_dead)

    steps_per_epoch = len(X_train) // BATCH_SIZE
    best_val_mse    = float("inf")
    history_j       = []
    total_resampled = 0

    print(f"\n  {'Ep':>4}  {'TrainMSE':>10}  {'ValMSE':>9}  {'L0':>5}  {'Dead':>5}  {'Resampled':>10}")
    print(f"  {'─'*56}")

    for epoch in range(1, EPOCHS + 1):
        sae.train()
        perm      = torch.randperm(len(X_train), device=DEVICE)
        epoch_mse = epoch_l0 = 0.0

        for i in range(steps_per_epoch):
            idx   = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch = X_train[idx]
            acts_b, recon = sae(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalise_decoder()

            fired = (acts_b.detach() > 0).any(dim=0)
            last_fired[fired] = global_step
            global_step += 1

            epoch_mse += loss.item()
            epoch_l0  += (acts_b.detach() > 0).float().sum(dim=1).mean().item()

            if global_step % RESAMPLE_INTERVAL == 0:
                dead_mask = (global_step - last_fired) > DEAD_WINDOW
                n = resample_dead(dead_mask)
                total_resampled += n

        train_mse = epoch_mse / steps_per_epoch
        train_l0  = epoch_l0  / steps_per_epoch

        sae.eval()
        with torch.no_grad():
            _, val_recon = sae(X_val)
            val_mse  = F.mse_loss(val_recon, X_val).item()
            dead_now = ((global_step - last_fired) > DEAD_WINDOW).sum().item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                "mode":        j,
                "epoch":       epoch,
                "model_state": sae.state_dict(),
                "n_features":  N_FEATURES,
                "k":           K_TOPK,
                "input_dim":   INPUT_DIM,
                "val_mse":     val_mse,
                "act_mean":    mean_j,
                "act_std":     std_j,
            }, OUT_DIR / f"sae_mode_{j}.pt")

        history_j.append({
            "epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
            "l0": train_l0, "dead": dead_now, "total_resampled": total_resampled,
        })

        print(f"  {epoch:>4}  {train_mse:>10.5f}  {val_mse:>9.5f}  "
              f"{train_l0:>5.1f}  {dead_now:>5}  {total_resampled:>10}")

    print(f"\n  Best val MSE : {best_val_mse:.5f}")
    print(f"  Checkpoint   → {OUT_DIR}/sae_mode_{j}.pt")
    all_history[f"mode_{j}"] = history_j


np.save(OUT_DIR / "sae_per_mode_history.npy", all_history)
print(f"\nAll done.  History → sae_data/sae_per_mode_history.npy")
