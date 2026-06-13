"""
Phase 7.3 — Feature–mode alignment
Phase 7.4 — Monosemanticity analysis

Loads the trained SAE and runs it on all 100 realisations' activations.
Computes a (n_features, 8) Pearson correlation matrix between SAE feature
time series and mode time series Z, then classifies each feature.

Thresholds
----------
  |r| >= 0.5  → "represents" a mode   (primary, used for success criterion)
  |r| >= 0.3  → "relates to" a mode   (secondary, used for polysemanticity)

Outputs (written to sae_data/)
-------------------------------
  alignment.npy   dict with correlation matrix and summary statistics
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, n_features: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    def forward(self, x: torch.Tensor):
        acts  = self.encode(x)
        recon = self.decoder(acts)
        return acts, recon

DATA_DIR = Path("sae_data")

THRESH_STRONG = 0.5   # "feature represents mode"
THRESH_WEAK   = 0.3   # "feature relates to mode"

N_MODES = 8
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── load ──────────────────────────────────────────────────────────────────────

ckpt = torch.load(DATA_DIR / "sae_best.pt", map_location=DEVICE, weights_only=False)

sae = TopKSAE(
    input_dim  = ckpt["input_dim"],
    n_features = ckpt["n_features"],
    k          = ckpt["k"],
).to(DEVICE)
sae.load_state_dict(ckpt["model_state"])
sae.eval()

act_mean = ckpt["act_mean"]   # (256,)
act_std  = ckpt["act_std"]    # (256,)

print(f"Loaded SAE  features={ckpt['n_features']}  k={ckpt['k']}  "
      f"val_MSE={ckpt['val_mse']:.5f}")

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, 497, 256)
Z_full    = np.load(DATA_DIR / "Z_full.npy")             # (100, 8, 497)

n_real, n_modes, t_eff, d = acts_full.shape
N_FEATURES = ckpt["n_features"]


# ── run SAE on all activations ────────────────────────────────────────────────
# Each sample corresponds to (realisation r, mode j, time t).
# We encode all samples, then for correlation purposes we keep the (mode, time)
# structure so we can compare feature_i(j, t) with Z_j(t).

print(f"\nRunning SAE on {n_real * n_modes * t_eff:,} samples ...")

acts_flat = acts_full.reshape(-1, d)                                    # (N, 256)
acts_norm = (acts_flat - act_mean) / act_std
X         = torch.from_numpy(acts_norm.astype(np.float32)).to(DEVICE)

BATCH = 2048
sae_acts_list = []

with torch.no_grad():
    for i in range(0, len(X), BATCH):
        batch_acts, _ = sae(X[i : i + BATCH])
        sae_acts_list.append(batch_acts.cpu().numpy())

# Reshape back: (100, 8, 497, N_FEATURES)
sae_acts = np.concatenate(sae_acts_list, axis=0).reshape(n_real, n_modes, t_eff, N_FEATURES)

print(f"SAE activations: shape={sae_acts.shape}  "
      f"mean L0={(sae_acts > 0).sum(-1).mean():.1f}")


# ── correlation matrix  C[feature, mode] ─────────────────────────────────────
# For each mode j: stack all (realisation, time) pairs → (100*497,) time series.
# Feature i's time series for mode j: sae_acts[:, j, :, i].flatten()  (100*497,)
# Z_j time series: Z_full[:, j, :].flatten()                          (100*497,)
# C[i, j] = Pearson( feature_i time series for mode j,  Z_j )

print("\nComputing correlation matrix C[features × modes] ...")

C = np.empty((N_FEATURES, N_MODES), dtype=np.float64)

for j in range(N_MODES):
    feat_j = sae_acts[:, j, :, :].reshape(-1, N_FEATURES).astype(np.float64)  # (100*497, M)
    Z_j    = Z_full[:, j, :].reshape(-1).astype(np.float64)                   # (100*497,)

    feat_c = feat_j - feat_j.mean(0)
    Z_c    = Z_j - Z_j.mean()
    feat_n = feat_c / (np.linalg.norm(feat_c, axis=0, keepdims=True) + 1e-8)
    Z_n    = Z_c   / (np.linalg.norm(Z_c) + 1e-8)

    C[:, j] = feat_n.T @ Z_n   # (M,)

print(f"Correlation matrix shape: {C.shape}")


# ── Phase 7.3 — per-mode summary ──────────────────────────────────────────────

print(f"\n{'─'*60}")
print("  Phase 7.3 — Feature–mode alignment")
print(f"{'─'*60}")
print(f"  {'Mode':<6}  {'Best feature':>13}  {'max |r|':>8}  {'# feats ≥0.5':>13}  Status")
print(f"  {'─'*58}")

mode_names = [f"X{j}" for j in range(N_MODES)]
mode_pass  = []

for j in range(N_MODES):
    col      = np.abs(C[:, j])
    best_f   = int(col.argmax())
    max_r    = col[best_f]
    n_strong = int((col >= THRESH_STRONG).sum())
    status   = "PASS" if max_r >= THRESH_STRONG else "FAIL"
    mode_pass.append(max_r >= THRESH_STRONG)
    print(f"  {mode_names[j]:<6}  {('feat '+str(best_f)):>13}  {max_r:>8.3f}  "
          f"{n_strong:>13}  {status}")

print()
n_pass = sum(mode_pass)
if n_pass == N_MODES:
    print(f"  SUCCESS: all {N_MODES} modes have max |r| ≥ {THRESH_STRONG}")
else:
    failed = [mode_names[j] for j, ok in enumerate(mode_pass) if not ok]
    print(f"  PARTIAL: {n_pass}/{N_MODES} modes pass. Failed: {failed}")
    print(f"  Consider increasing K or expansion factor.")


# ── Phase 7.4 — monosemanticity analysis ─────────────────────────────────────

print(f"\n{'─'*60}")
print("  Phase 7.4 — Monosemanticity analysis")
print(f"{'─'*60}")

categories = {"monosemantic": 0, "weakly_poly": 0, "strongly_poly": 0, "noise": 0}
# Also track per-mode monosemantic assignment
mono_features = {j: [] for j in range(N_MODES)}

for i in range(N_FEATURES):
    row        = np.abs(C[i])                         # (8,)
    n_strong   = int((row >= THRESH_STRONG).sum())    # modes above 0.5
    n_weak     = int((row >= THRESH_WEAK).sum())      # modes above 0.3

    if n_strong == 1 and n_weak == 1:
        categories["monosemantic"] += 1
        j = int(row.argmax())
        mono_features[j].append((i, float(row[j])))
    elif n_strong >= 2:
        categories["strongly_poly"] += 1
    elif n_strong == 0 and n_weak >= 2:
        categories["weakly_poly"] += 1
    elif n_strong == 1 and n_weak >= 2:
        # strong for one mode but weakly related to others
        categories["weakly_poly"] += 1
    else:
        categories["noise"] += 1

print(f"  Feature classification ({N_FEATURES} total):")
for cat, count in categories.items():
    pct = count / N_FEATURES * 100
    print(f"    {cat:<18} {count:>4}  ({pct:5.1f}%)")

print(f"\n  Monosemantic features per mode:")
print(f"  {'Mode':<6}  {'# mono feats':>13}  {'best |r|':>9}  Top features")
print(f"  {'─'*58}")
for j in range(N_MODES):
    feats   = sorted(mono_features[j], key=lambda x: -x[1])
    n_mono  = len(feats)
    best_r  = feats[0][1] if feats else 0.0
    top_str = ", ".join(f"f{fi}({r:.2f})" for fi, r in feats[:3])
    if not top_str:
        top_str = "none"
    print(f"  {mode_names[j]:<6}  {n_mono:>13}  {best_r:>9.3f}  {top_str}")

# Feature–feature redundancy check
print(f"\n  Feature–feature redundancy check:")
W   = F.normalize(
    torch.from_numpy(sae.encoder.weight.detach().cpu().numpy()), dim=1
).numpy()   # (M, D)
CC  = W @ W.T   # (M, M) cosine similarities
np.fill_diagonal(CC, 0.0)
max_sim     = np.abs(CC).max()
n_redundant = int((np.abs(CC) > 0.9).sum() // 2)  # pairs
print(f"    Max pairwise |cos_sim| : {max_sim:.3f}")
print(f"    Redundant pairs (>0.9) : {n_redundant}")
if n_redundant > N_FEATURES // 10:
    print("    WARNING: many redundant features — consider smaller expansion factor")


# ── save ──────────────────────────────────────────────────────────────────────

results = {
    "C":              C,              # (M, 8) correlation matrix
    "mode_max_r":     np.array([np.abs(C[:, j]).max() for j in range(N_MODES)]),
    "mode_best_feat": np.array([int(np.abs(C[:, j]).argmax()) for j in range(N_MODES)]),
    "categories":     categories,
    "mono_features":  mono_features,
    "thresh_strong":  THRESH_STRONG,
    "thresh_weak":    THRESH_WEAK,
}
np.save(DATA_DIR / "alignment.npy", results)
print(f"\nResults saved → sae_data/alignment.npy")
