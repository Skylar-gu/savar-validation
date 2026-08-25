"""
Mixed-mode TopK SAE — one SAE trained on ALL modes' pooled activations together.

Counterpart to train_sae_per_mode.py (same architecture/hparams so results are
comparable): samples from all 8 mode-weighted streams are concatenated and a
single global normalisation is used (no mode label anywhere in training).
This is the setting the per-mode script's docstring argues fails — retrained
here on the het-dynamics data to test whether timescale heterogeneity changes
that, and as the substrate for mode-identity probes.

Outputs (written to --datadir, per-mode files untouched)
--------------------------------------------------------
  sae_mixed.pt            checkpoint (with global act_mean/act_std)
  sae_mixed_history.npy   per-epoch log
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--datadir", required=True,
                 help="Activation dir (e.g. sae_data/hetdynamics)")
_ap.add_argument("--seed", type=int, default=None,
                 help="If set: seed torch/numpy and write sae_mixed_<arch>_seed<N>.pt "
                      "instead of sae_mixed.pt (existing artifacts never overwritten)")
_args = _ap.parse_args()
DATA_DIR = OUT_DIR = Path(_args.datadir)

# Block C2: SAE_ARCH=kan adds per-feature learnable 1-D spline gates on the
# ENCODER PRE-ACTIVATIONS only; TopK sparsity + linear decoder unchanged.
import os
SAE_ARCH = os.environ.get("SAE_ARCH", "topk").lower()
assert SAE_ARCH in ("topk", "kan"), SAE_ARCH

if _args.seed is not None:
    torch.manual_seed(_args.seed)
    np.random.seed(_args.seed)
    _CKPT_NAME = f"sae_mixed_{SAE_ARCH}_seed{_args.seed}.pt"
    _HIST_NAME = f"sae_mixed_{SAE_ARCH}_seed{_args.seed}_history.npy"
else:
    assert SAE_ARCH == "topk", "kan runs must set --seed (keeps default files untouched)"
    _CKPT_NAME = "sae_mixed.pt"
    _HIST_NAME = "sae_mixed_history.npy"

# ── hyperparameters (identical to per-mode for comparability) ─────────────────

INPUT_DIM  = 256
N_FEATURES = 512
K_TOPK     = 25
LR         = 1e-3
EPOCHS     = int(os.environ.get("SAE_EPOCHS", 60))  # T6a movmech runs use 20 (stride-1 data has 5x the frames of eqvar; 20 ep = 127k steps > Block C's 76k)
BATCH_SIZE = 256

RESAMPLE_INTERVAL = 1000
DEAD_WINDOW       = 500

VAL_FRAC   = 0.15
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KANGate(nn.Module):
    """Per-feature learnable 1-D function on encoder pre-activations (KAN-SAE,
    arXiv 2605.17493 style): g_f(x) = w_base_f * x + sum_b c_{f,b} * rbf_b(x).
    Initialised to the identity (c=0, w_base=1) so seed-for-seed the model starts
    exactly at the TopK SAE. ~n_features*(n_basis+1) extra params."""

    def __init__(self, n_features: int, n_basis: int = 8, x_range: float = 4.0):
        super().__init__()
        centers = torch.linspace(-x_range, x_range, n_basis)
        self.register_buffer("centers", centers)
        self.h = 2.0 * x_range / (n_basis - 1)
        self.coef = nn.Parameter(torch.zeros(n_features, n_basis))
        self.base = nn.Parameter(torch.ones(n_features))

    def forward(self, pre):                       # pre: (B, F)
        phi = torch.exp(-((pre.unsqueeze(-1) - self.centers) / self.h) ** 2)  # (B, F, nb)
        return self.base * pre + (phi * self.coef).sum(-1)


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, n_features: int, k: int, arch: str = "topk"):
        super().__init__()
        self.k          = k
        self.n_features = n_features
        self.arch       = arch
        self.encoder    = nn.Linear(input_dim, n_features, bias=True)
        self.gate       = KANGate(n_features) if arch == "kan" else None
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
        if self.gate is not None:                 # KAN: per-feature 1-D gate on pre-acts
            pre = self.gate(pre)
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


# ── load: all modes concatenated, split by realisation ────────────────────────

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, T, 256)
n_real, n_modes, t_eff, d = acts_full.shape
assert d == INPUT_DIM

# Space-time SAE (v2 T6, arXiv:2604.03919): stack SAE_SPACETIME consecutive
# time-steps of each pooled per-mode activation so a MOVING pattern's temporal
# signature is one atom (a per-frame SAE cannot represent it). input_dim = ST*256.
ST = int(os.environ.get("SAE_SPACETIME", 1))

def _flatten(a):   # a: (R, 8, T, 256) -> (samples, ST*256)
    if ST == 1:
        return a.reshape(-1, d).astype(np.float32)
    R, M, T, D = a.shape
    w = np.stack([a[:, :, t:t+ST] for t in range(T - ST + 1)], axis=2)  # (R,M,T',ST,D)
    return w.reshape(-1, ST * D).astype(np.float32, copy=False)

n_val   = max(1, int(n_real * VAL_FRAC))
n_train = n_real - n_val
if ST > 1:
    INPUT_DIM = ST * d
    _CKPT_NAME = _CKPT_NAME.replace(".pt", f"_st{ST}.pt")
    _HIST_NAME = _HIST_NAME.replace(".npy", f"_st{ST}.npy")
    print(f"  SPACE-TIME SAE: ST={ST}  input_dim={INPUT_DIM}")

train_x = _flatten(acts_full[:n_train])
val_x   = _flatten(acts_full[n_train:])

# single GLOBAL normalisation — no mode information used
mean_g = train_x.mean(0)
std_g  = train_x.std(0) + 1e-8

# in-place normalisation (ST>1 windowed arrays are multi-GB; avoid broadcast copies)
train_x -= mean_g; train_x /= std_g
val_x   -= mean_g; val_x   /= std_g
X_train = torch.from_numpy(train_x).to(DEVICE)
X_val   = torch.from_numpy(val_x).to(DEVICE)

print(f"Mixed-mode SAE  —  {DATA_DIR}")
print(f"  train={len(X_train):,}  val={len(X_val):,}  (all {n_modes} modes pooled)")

sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK, arch=SAE_ARCH).to(DEVICE)
print(f"  arch={SAE_ARCH}  seed={_args.seed}  ckpt={_CKPT_NAME}")
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
history         = []
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
            total_resampled += resample_dead(dead_mask)

    train_mse = epoch_mse / steps_per_epoch
    train_l0  = epoch_l0  / steps_per_epoch

    sae.eval()
    with torch.no_grad():
        val_mse = 0.0
        for i in range(0, len(X_val), 65536):
            _, vr = sae(X_val[i:i+65536])
            val_mse += F.mse_loss(vr, X_val[i:i+65536], reduction="sum").item()
        val_mse /= X_val.numel()
        dead_now = ((global_step - last_fired) > DEAD_WINDOW).sum().item()

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        torch.save({
            "epoch":       epoch,
            "model_state": sae.state_dict(),
            "n_features":  N_FEATURES,
            "k":           K_TOPK,
            "input_dim":   INPUT_DIM,
            "arch":        SAE_ARCH,
            "val_mse":     val_mse,
            "act_mean":    mean_g,
            "act_std":     std_g,
        }, OUT_DIR / _CKPT_NAME)

    history.append({
        "epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
        "l0": train_l0, "dead": dead_now, "total_resampled": total_resampled,
    })
    print(f"  {epoch:>4}  {train_mse:>10.5f}  {val_mse:>9.5f}  "
          f"{train_l0:>5.1f}  {dead_now:>5}  {total_resampled:>10}")

np.save(OUT_DIR / _HIST_NAME, history)

# Final-epoch checkpoint (bake-off scoring uses THIS — literature rule: never
# select the model by reconstruction MSE; final weights for every arch/seed).
if _args.seed is not None:
    torch.save({
        "epoch":       EPOCHS,
        "model_state": sae.state_dict(),
        "n_features":  N_FEATURES,
        "k":           K_TOPK,
        "input_dim":   INPUT_DIM,
        "arch":        SAE_ARCH,
        "val_mse":     val_mse,
        "act_mean":    mean_g,
        "act_std":     std_g,
    }, OUT_DIR / _CKPT_NAME.replace(".pt", "_final.pt"))

print(f"\nBest val MSE : {best_val_mse:.5f}")
print(f"Checkpoint   → {OUT_DIR}/{_CKPT_NAME}")
