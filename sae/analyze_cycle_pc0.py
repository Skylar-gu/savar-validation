"""
Diurnal Phase 7 add-on — does the cyclic CNN push the diurnal/annual cycle into
the dominant activation direction (PC0), and how does that interact with the
PC0 'global-activity' collapse seen in the baseline (PC0 ≈ 86% variance)?

For each mode j (mode-weighted res3 activations, 256-dim):
  - PC0 var%            : variance share of the top PC (collapse indicator)
  - R²(PC0 ~ cycle)     : fraction of the PC0 time series explained by a
                          diurnal+annual harmonic regression (is PC0 a cycle?)
  - R²(Z_j ~ cycle)     : how cyclic the mode's latent state itself is (reference)
  - |r|(PC0, Z_j)       : alignment of PC0 with the mode

Window t aligns to Z_j(t+K), so absolute time = K + t.
Reads sae_data_diurnal/{activations_full,Z_full}.npy. Output: cycle_pc0.npy
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--dy005", action="store_true")
_ap.add_argument("--deseason", action="store_true",
                 help="Analyze deseasonalized activations; expect R²(PC0~cycle)≈0")
_args = _ap.parse_args()

_base = "sae_data_diurnal" if _args.diurnal else ("sae_data_dy005" if _args.dy005 else "sae_data")
if _args.deseason:
    _base += "_deseason"
DATA_DIR = Path(_base)
K        = 3
P_D      = 4         # diurnal period (steps) at dt=6h
P_A      = 1461      # annual period (steps)

acts = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, T_eff, 256)
Z    = np.load(DATA_DIR / "Z_full.npy")             # (100, 8, T_eff)
n_real, n_modes, t_eff, d = acts.shape
print(f"Loaded activations {acts.shape}, Z {Z.shape}")

# ── cycle design matrix on absolute time (identical per realisation) ──────────
t_abs = K + np.arange(t_eff)
cols  = [np.ones(t_eff)]
for P, nh in ((P_D, 1), (P_A, 2)):
    for h in range(1, nh + 1):
        w = 2 * np.pi * h * t_abs / P
        cols += [np.cos(w), np.sin(w)]
D1 = np.stack(cols, axis=1)            # (t_eff, n_reg)
D  = np.tile(D1, (n_real, 1))          # (n_real*t_eff, n_reg)
# precompute projector pieces
DtD_inv_Dt = np.linalg.pinv(D)         # (n_reg, n_real*t_eff)


def r2_on_cycle(y):
    """R² of regressing y (1-D) on the tiled cycle design matrix."""
    y = y.astype(np.float64)
    beta = DtD_inv_Dt @ y
    yhat = D @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


print(f"\n  {'Mode':<5} {'PC0 var%':>9} {'R2(PC0~cyc)':>12} {'R2(Z~cyc)':>11} {'|r|(PC0,Z)':>11}")
print(f"  {'-'*52}")

out = {}
for j in range(n_modes):
    Aj = acts[:, j, :, :].reshape(-1, d).astype(np.float64)   # (n_real*t_eff, 256)
    Zj = Z[:, j, :].reshape(-1).astype(np.float64)

    pca = PCA(n_components=4)
    scores = pca.fit_transform(Aj)            # (n_real*t_eff, 4)
    pc0 = scores[:, 0]
    pc0_var = pca.explained_variance_ratio_[0] * 100

    r2_pc0 = r2_on_cycle(pc0)
    r2_z   = r2_on_cycle(Zj)
    # |corr| PC0 vs Z_j
    a = pc0 - pc0.mean(); b = Zj - Zj.mean()
    r_pc0_z = abs((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"  X{j:<4} {pc0_var:>8.1f}% {r2_pc0:>12.3f} {r2_z:>11.3f} {r_pc0_z:>11.3f}")
    out[j] = {"pc0_var_pct": pc0_var, "r2_pc0_cycle": r2_pc0,
              "r2_z_cycle": r2_z, "r_pc0_z": r_pc0_z}

mean_pc0 = np.mean([v["pc0_var_pct"] for v in out.values()])
mean_r2  = np.mean([v["r2_pc0_cycle"] for v in out.values()])
print(f"\n  Mean PC0 var% = {mean_pc0:.1f}%   (baseline no-cycle ≈ 86%)")
print(f"  Mean R²(PC0 ~ cycle) = {mean_r2:.3f}")
print(f"  → High PC0 var% AND high R²(PC0~cycle) means the diurnal/annual cycle")
print(f"    is dominating the top activation direction (cycle-driven collapse).")

np.save(DATA_DIR / "cycle_pc0.npy", out)
print(f"\nSaved → {DATA_DIR}/cycle_pc0.npy")
