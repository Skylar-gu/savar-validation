"""
Measure the per-mode LINEAR CEILING on a pooled-activation dataset: the maximum
Pearson |r| between Z_j and any linear readout of the 256-dim mode-weighted
activations, estimated out-of-fold (5 folds over REALISATIONS, so no window of
a held-out realisation is seen in training) with ridge regression, alpha
chosen by inner RidgeCV on the training folds.

Also reports the PCA variance spectrum of the pooled activations (all modes
stacked), which is the "PC0 share" number.

Usage:  python3 sae/measure_ceilings.py --allknobs   (or --diurnal / --dy005)
Writes: <sae_data_dir>/ceilings.npy  (8,)  and  pca_spectrum.npy
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

_ap = argparse.ArgumentParser()
_ap.add_argument("--allknobs", action="store_true")
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--dy005", action="store_true")
_args = _ap.parse_args()

if _args.allknobs:
    DATA_DIR = Path("sae_data_allknobs")
elif _args.diurnal:
    DATA_DIR = Path("sae_data_diurnal")
elif _args.dy005:
    DATA_DIR = Path("sae_data_dy005")
else:
    DATA_DIR = Path("sae_data")

N_FOLDS = 5
ALPHAS  = np.logspace(-2, 3, 11)

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (R, 8, T_eff, 256)
Z_full    = np.load(DATA_DIR / "Z_full.npy")             # (R, 8, T_eff)
R, N_MODES, T_eff, D = acts_full.shape
assert np.isfinite(acts_full).all() and np.isfinite(Z_full).all()

folds = np.array_split(np.arange(R), N_FOLDS)

print(f"Ceilings: {DATA_DIR}  R={R} modes={N_MODES} T_eff={T_eff} D={D}")
print(f"  {N_FOLDS}-fold out-of-fold ridge over realisations, alpha grid 1e-2..1e3\n")
print(f"  {'Mode':<5} {'ceiling |r|':>12} {'alpha':>8}")

ceilings = np.zeros(N_MODES)
for j in range(N_MODES):
    X = acts_full[:, j].astype(np.float64)     # (R, T_eff, D)
    y = Z_full[:, j].astype(np.float64)        # (R, T_eff)
    pred = np.zeros_like(y)
    alphas_used = []
    for k in range(N_FOLDS):
        te = folds[k]
        tr = np.concatenate([folds[i] for i in range(N_FOLDS) if i != k])
        Xtr = X[tr].reshape(-1, D); ytr = y[tr].reshape(-1)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
        alphas_used.append(m.alpha_)
        pred[te] = m.predict((X[te].reshape(-1, D) - mu) / sd).reshape(len(te), T_eff)
    r = np.corrcoef(pred.reshape(-1), y.reshape(-1))[0, 1]
    ceilings[j] = abs(r)
    print(f"  X{j:<4} {abs(r):>12.3f} {np.median(alphas_used):>8.2g}")

np.save(DATA_DIR / "ceilings.npy", ceilings)

# PCA spectrum over all pooled samples (all modes stacked), as in Phase 7
Xall = acts_full.reshape(-1, D).astype(np.float32)
sub  = Xall[np.random.default_rng(0).choice(len(Xall), min(len(Xall), 400_000), replace=False)]
pca  = PCA(n_components=10).fit(sub)
evr  = pca.explained_variance_ratio_
np.save(DATA_DIR / "pca_spectrum.npy", evr)
print(f"\n  PCA of pooled activations (all modes): PC0 {evr[0]*100:.1f}%  PC1 {evr[1]*100:.1f}%  "
      f"PC2 {evr[2]*100:.1f}%  PC3+ {(1-evr[:3].sum())*100:.1f}%")
print(f"\nSaved {DATA_DIR}/ceilings.npy  {np.round(ceilings, 3)}")
