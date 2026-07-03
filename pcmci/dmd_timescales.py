"""
Block H — Koopman/DMD timescale cross-check (SINDy-SHRED/Koopman line).

DMD (rank ~20) on eqvar pixel data per realisation gives an architecture-free
linear-surrogate readout of the system's timescales:
  * eigenvalue e-folding times  τ = −1 / ln|λ|  vs the designed φ spectrum
    (τ_j = −1/ln φ_j, 22.8× spread);
  * leading DMD modes' spatial supports vs the W blob footprints
    (cosine of |mode| with each W row).

Output: results/dmd_timescales.npy
Env: DMD_RANK (20), DMD_NREAL (20)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from pydmd import DMD

DATA_DIR = Path("data/realisations_hetdynamics_eqvar")
RANK   = int(os.environ.get("DMD_RANK", 20))
N_REAL = int(os.environ.get("DMD_NREAL", 20))
PHI = np.array([0.15, 0.30, 0.42, 0.55, 0.68, 0.78, 0.86, 0.92])
TAU_DESIGN = -1.0 / np.log(PHI)
N_MODES = 8

paths = sorted(DATA_DIR.glob("realisation_*.npz"))[:N_REAL]
d0 = np.load(paths[0])
W = d0["W"].astype(np.float64)                       # (8, 2500)
Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

all_taus, all_best_cos, all_best_blob = [], [], []
per_real = []
for p in paths:
    X = np.load(p)["observations"].astype(np.float64)    # (2500, T)
    dmd = DMD(svd_rank=RANK)
    dmd.fit(X)
    lam = dmd.eigs
    amps = np.abs(dmd.amplitudes)
    taus = -1.0 / np.log(np.clip(np.abs(lam), 1e-9, 1 - 1e-9))
    modes = dmd.modes                                     # (2500, r)
    m = np.abs(modes)
    m = m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-12)
    cos = Wn @ m                                          # (8, r)
    best_blob = cos.argmax(0)
    best_cos = cos.max(0)
    order = np.argsort(-amps)
    per_real.append(dict(taus=taus, amps=amps, best_blob=best_blob,
                         best_cos=best_cos, order=order))
    all_taus.append(taus); all_best_cos.append(best_cos)
    all_best_blob.append(best_blob)

# summarize: for each designed mode, the DMD eigenvalue whose mode maps to its
# blob with highest cosine (weighted by amplitude), and its e-folding time
tau_hat = np.full(N_MODES, np.nan)
cos_hat = np.full(N_MODES, np.nan)
for j in range(N_MODES):
    cands = []
    for pr in per_real:
        sel = np.where(pr["best_blob"] == j)[0]
        if len(sel):
            k = sel[np.argmax(pr["best_cos"][sel])]
            cands.append((pr["taus"][k], pr["best_cos"][k]))
    if cands:
        t, c = np.array(cands).T
        tau_hat[j] = np.median(t)
        cos_hat[j] = np.median(c)

print(f"DMD rank {RANK} on {len(paths)} eqvar realisations")
print(f"  {'mode':<6} {'tau_design':>10} {'tau_DMD(med)':>13} {'blob cos(med)':>14}")
for j in range(N_MODES):
    print(f"  X{j:<5} {TAU_DESIGN[j]:>10.2f} {tau_hat[j]:>13.2f} {cos_hat[j]:>14.3f}")

fin = np.isfinite(tau_hat)
def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else np.nan
sp = spear(TAU_DESIGN[fin], tau_hat[fin]) if fin.sum() > 2 else np.nan
print(f"\n  Spearman(tau_design, tau_DMD) = {sp:.3f}  (n={int(fin.sum())})")

os.makedirs("results", exist_ok=True)
np.save("results/dmd_timescales.npy",
        dict(tau_design=TAU_DESIGN, tau_dmd=tau_hat, blob_cos=cos_hat,
             spearman=sp, rank=RANK, n_real=len(paths), per_real=per_real),
        allow_pickle=True)
print("saved -> results/dmd_timescales.npy")
