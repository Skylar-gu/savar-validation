"""
Per-mode alignment of the MIXED-mode SAE (sae_mixed.pt from train_sae_mixed.py).

Same table as eval_sae_per_mode.py — for each mode j, the best-|r| feature of
the single shared SAE against Z_j, ceiling fraction, and cross-mode specificity
— so the mixed SAE is directly comparable with the per-mode SAEs.

Outputs: alignment_mixed.npy in --datadir (per-mode files untouched).
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--datadir", required=True)
_args = _ap.parse_args()
DATA_DIR = Path(_args.datadir)

INPUT_DIM     = 256
N_FEATURES    = 512
K_TOPK        = 25
N_MODES       = 8
THRESH_ALIGN  = 0.35
THRESH_STRONG = 0.50
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CEILINGS = list(np.load(DATA_DIR / "ceilings.npy"))


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, n_features: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x):
        pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    def forward(self, x):
        acts = self.encode(x)
        return acts, self.decoder(acts)


def encode_batch(sae, X_np, mean, std, batch_size=4096):
    X = torch.from_numpy(((X_np - mean) / std).astype(np.float32)).to(DEVICE)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            acts, _ = sae(X[i:i+batch_size])
            out.append(acts.cpu().numpy())
    return np.concatenate(out, axis=0)


def pearson_col(A, b):
    A = A.astype(np.float64); b = b.astype(np.float64)
    A_c = A - A.mean(0); b_c = b - b.mean()
    norms = np.linalg.norm(A_c, axis=0) + 1e-12
    return (A_c.T @ b_c) / (norms * (np.linalg.norm(b_c) + 1e-12))


ckpt = torch.load(DATA_DIR / "sae_mixed.pt", map_location=DEVICE, weights_only=False)
sae  = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
sae.load_state_dict(ckpt["model_state"])
sae.eval()
mean_g, std_g = ckpt["act_mean"], ckpt["act_std"]

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, T, 256)
Z_full    = np.load(DATA_DIR / "Z_full.npy")              # (100, 8, T)

# Pre-encode every mode's stream once (same SAE, same global normalisation)
sae_acts = [encode_batch(sae, acts_full[:, j].reshape(-1, INPUT_DIM), mean_g, std_g)
            for j in range(N_MODES)]
Z_flat   = [Z_full[:, j].reshape(-1) for j in range(N_MODES)]

# Full corr matrix: corr[j][k] = |r| of mode-j's best feature computed per stream
print("Mixed-mode SAE — per-mode alignment  (single shared SAE, global norm)")
print("=" * 70)
print()
print(f"  {'Mode':<5}  {'BestFeat':>9}  {'max|r|':>7}  {'Ceil':>6}  "
      f"{'Frac':>6}  {'Status':<10}  Cross-mode specificity")
print(f"  {'─'*70}")

results = {}
n_pass = n_strong = 0
for j in range(N_MODES):
    C_j    = pearson_col(sae_acts[j], Z_flat[j])
    best_f = int(np.abs(C_j).argmax())
    max_r  = float(np.abs(C_j)[best_f])
    ceil_j = CEILINGS[j]
    frac   = max_r / ceil_j

    cross_r = [float(np.abs(pearson_col(sae_acts[k], Z_flat[k])[best_f]))
               for k in range(N_MODES) if k != j]
    cross_max   = max(cross_r)
    specificity = max_r - cross_max

    status = ("STRONG" if max_r >= THRESH_STRONG
              else "ALIGN" if max_r >= THRESH_ALIGN else "FAIL")
    n_pass   += max_r >= THRESH_ALIGN
    n_strong += max_r >= THRESH_STRONG

    print(f"  X{j}     {('f'+str(best_f)):>9}  {max_r:>7.4f}  {ceil_j:>6.3f}  "
          f"{frac:>6.2f}  {status:<10}  j={max_r:.3f}  max_other={cross_max:.3f}  "
          f"spec={specificity:+.3f}")

    results[j] = {"best_feat": best_f, "max_r": max_r, "ceiling": ceil_j,
                  "frac_ceil": frac, "C_j": C_j, "cross_r": cross_r,
                  "cross_max": cross_max, "specificity": specificity}

print(f"\n  Aligned  (|r| ≥ {THRESH_ALIGN}): {n_pass}/{N_MODES} modes")
print(f"  Strong   (|r| ≥ {THRESH_STRONG}): {n_strong}/{N_MODES} modes")

shared = len(set(r["best_feat"] for r in results.values()))
print(f"  Distinct best features across modes: {shared}/{N_MODES}")

np.save(DATA_DIR / "alignment_mixed.npy", results)
print(f"\nResults → {DATA_DIR}/alignment_mixed.npy")
