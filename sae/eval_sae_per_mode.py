"""
Phase 7.3b / 7.4b — Alignment and monosemanticity for per-mode SAEs.

For each mode j:
  1. Load sae_mode_{j}.pt
  2. Run it on mode j's activations  → SAE feature activations (49700, 512)
  3. Compute Pearson correlation of each feature with Z_j  → C_j[feature]
  4. Find best feature and its |r|
  5. Cross-mode check: run SAE_j on other modes' activations;
     check whether best feature also fires for other Z_k

Thresholds
----------
  THRESH_ALIGN  = 0.35   (achievable for all modes given theoretical ceilings)
  THRESH_STRONG = 0.50   (modes X1, X3, X6 can reach this)

Theoretical ceilings (from linear regression on all 256 dims):
  X0≈0.49, X1≈0.57, X2≈0.48, X3≈0.59, X4≈0.45, X5≈0.36, X6≈0.58, X7≈0.46

Outputs
-------
  sae_data/alignment_per_mode.npy   dict with per-mode results
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

N_MODES      = 8
INPUT_DIM    = 256
N_FEATURES   = 512
K_TOPK       = 25
THRESH_ALIGN  = 0.35   # minimum to consider "aligned"
THRESH_STRONG = 0.50   # strong alignment (achievable for X1, X3, X6)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Theoretical ceilings from full-data ridge regression (per dataset)
_CEILINGS_DY1   = [0.489, 0.575, 0.479, 0.589, 0.450, 0.358, 0.584, 0.465]
_CEILINGS_DY005 = [0.490, 0.579, 0.477, 0.589, 0.450, 0.363, 0.584, 0.468]
# Diurnal ceilings not yet measured; reuse dy005 (same D_y) as a rough reference.
# The 'Frac' column is informational only — alignment pass/fail uses THRESH_ALIGN.
if _args.diurnal or _args.dy005:
    CEILINGS = _CEILINGS_DY005
else:
    CEILINGS = _CEILINGS_DY1


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, n_features: int, k: int):
        super().__init__()
        self.k          = k
        self.n_features = n_features
        self.encoder    = nn.Linear(input_dim, n_features, bias=True)
        self.decoder    = nn.Linear(n_features, input_dim, bias=True)

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


def encode_batch(sae, X_np: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 batch_size: int = 2048) -> np.ndarray:
    """Normalise X_np, run SAE encoder, return numpy (N, M)."""
    X_norm = torch.from_numpy(((X_np - mean) / std).astype(np.float32)).to(DEVICE)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            acts, _ = sae(X_norm[i : i + batch_size])
            out.append(acts.cpu().numpy())
    return np.concatenate(out, axis=0)


def pearson_col(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of A with vector b. Returns (A.shape[1],)."""
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    A_c = A - A.mean(0)
    b_c = b - b.mean()
    norms = np.linalg.norm(A_c, axis=0) + 1e-12
    return (A_c.T @ b_c) / (norms * (np.linalg.norm(b_c) + 1e-12))


acts_full = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, 497, 256)
Z_full    = np.load(DATA_DIR / "Z_full.npy")             # (100, 8, 497)

n_real, _, t_eff, _ = acts_full.shape

results = {}

print("Phase 7.3b — Per-mode SAE alignment")
print("=" * 70)
print()
print(f"  {'Mode':<5}  {'BestFeat':>9}  {'max|r|':>7}  {'Ceil':>6}  "
      f"{'Frac':>6}  {'Status':<10}  Cross-mode specificity")
print(f"  {'─'*70}")

mode_pass = []

for j in range(N_MODES):
    ckpt_path = DATA_DIR / f"sae_mode_{j}.pt"
    if not ckpt_path.exists():
        print(f"  X{j}     [MISSING checkpoint {ckpt_path}]")
        mode_pass.append(False)
        continue

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sae  = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    mean_j = ckpt["act_mean"]   # (256,)
    std_j  = ckpt["act_std"]    # (256,)

    # ── intra-mode: run SAE_j on mode j's activations ────────────────────────
    acts_j = acts_full[:, j, :, :].reshape(-1, INPUT_DIM)   # (49700, 256)
    Z_j    = Z_full[:, j, :].reshape(-1)                    # (49700,)

    sae_acts_j = encode_batch(sae, acts_j, mean_j, std_j)   # (49700, 512)

    C_j      = pearson_col(sae_acts_j, Z_j)                  # (512,)
    abs_C_j  = np.abs(C_j)
    best_f   = int(abs_C_j.argmax())
    max_r    = float(abs_C_j[best_f])
    ceil_j   = CEILINGS[j]
    frac     = max_r / ceil_j

    passed = max_r >= THRESH_ALIGN
    mode_pass.append(passed)
    status = "STRONG" if max_r >= THRESH_STRONG else ("ALIGN" if passed else "FAIL")

    # ── cross-mode: does this feature fire similarly for other modes? ─────────
    # Compute best feature's correlation with Z_k for k ≠ j
    cross_r = []
    for k in range(N_MODES):
        if k == j:
            continue
        acts_k   = acts_full[:, k, :, :].reshape(-1, INPUT_DIM)
        Z_k      = Z_full[:, k, :].reshape(-1)
        sae_ak   = encode_batch(sae, acts_k, mean_j, std_j)   # use SAE_j's normalization
        r_cross  = float(abs(pearson_col(sae_ak, Z_k)[best_f]))
        cross_r.append(r_cross)

    cross_max  = max(cross_r)
    specificity = max_r - cross_max   # positive = more correlated with j than others

    cross_str = f"j={max_r:.3f}  max_other={cross_max:.3f}  spec={specificity:+.3f}"

    print(f"  X{j}     {('f'+str(best_f)):>9}  {max_r:>7.4f}  {ceil_j:>6.3f}  "
          f"{frac:>6.2f}  {status:<10}  {cross_str}")

    results[j] = {
        "best_feat":   best_f,
        "max_r":       max_r,
        "ceiling":     ceil_j,
        "frac_ceil":   frac,
        "C_j":         C_j,
        "cross_r":     cross_r,
        "cross_max":   cross_max,
        "specificity": specificity,
    }

n_pass   = sum(mode_pass)
n_strong = sum(r["max_r"] >= THRESH_STRONG for r in results.values())

print(f"\n  Aligned  (|r| ≥ {THRESH_ALIGN}): {n_pass}/{N_MODES} modes")
print(f"  Strong   (|r| ≥ {THRESH_STRONG}): {n_strong}/{N_MODES} modes")

# ── Monosemanticity summary ───────────────────────────────────────────────────

print(f"\n{'─'*70}")
print("  Phase 7.4b — Monosemanticity summary")
print(f"{'─'*70}")
print()
print("  For each mode's best feature, 'specific' = it correlates more with")
print("  that mode's Z than with any other mode's Z (specificity > 0).")
print()

n_specific  = 0
n_global    = 0
n_weak      = 0

for j in range(N_MODES):
    if j not in results:
        continue
    r = results[j]
    if r["max_r"] < THRESH_ALIGN:
        n_weak += 1
        verdict = "WEAK (below alignment threshold)"
    elif r["specificity"] > 0.05:
        n_specific += 1
        verdict = "MONOSEMANTIC (mode-specific)"
    else:
        n_global += 1
        verdict = "GLOBAL (fires for multiple modes)"

    print(f"  X{j}: best feature r={r['max_r']:.3f}  spec={r['specificity']:+.3f}  "
          f"→ {verdict}")

print()
print(f"  Monosemantic (specific)  : {n_specific}/{N_MODES}")
print(f"  Global / polysemantic    : {n_global}/{N_MODES}")
print(f"  Weak (below threshold)   : {n_weak}/{N_MODES}")


# ── save ─────────────────────────────────────────────────────────────────────

np.save(DATA_DIR / "alignment_per_mode.npy", results)
print(f"\nResults → {DATA_DIR}/alignment_per_mode.npy")
