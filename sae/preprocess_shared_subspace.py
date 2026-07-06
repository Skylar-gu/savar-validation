"""
Follow-up 3 — Break the shared subspace before the SAE: diagnostics + datadirs.

Blocks C/D/F pointed at "one global amplitude direction absorbing everything".
Diagnostics here sharpen that picture on sae_data/hetdynamics_eqvar:

  1. mode-agnostic readout: ONE ridge fit on the MIXED stream (target = own-
     stream Z) reads out every mode's Z at |r| ~= the full per-mode readouts.
  2. BUT deflation is flat: project that direction out, refit, and an
     equivalent direction reappears (cross-mode mean |r| unchanged over 4
     deflations) -> the shared signal is HIGH-RANK, not one direction.
  3. cross-application: every per-mode ridge readout reads every OTHER stream
     essentially as well as its own (readout uniqueness <= 0.024) -> pooled
     activations contain ~no mode-unique linear structure. This is the
     representation-level ceiling behind SAE uniqueness ~ 0.

Interventions (each written as a full datadir so train_sae_mixed.py /
eval_sae_metrics.py run unchanged):

  sae_data/hetdynamics_eqvar_proj/    activations with the k=1 mixed-ridge
                                      shared direction projected out (the
                                      literal "project out the shared
                                      direction" intervention; diagnostics
                                      predict a null result).
  sae_data/hetdynamics_eqvar_whiten/  PCA-whitened activations (fit on train
                                      reals, eigenvalue floor 1e-6*lam_max).
                                      Rationale: mode-IDENTITY lives in
                                      low-variance per-blob channel
                                      fingerprints (Block D: 9 identity coders,
                                      id-R2 up to 0.896) that per-channel std
                                      normalization + TopK reconstruction
                                      deprioritize; whitening rebalances them.
                                      whiten_params.npz stores (mu, Wm, Wm_inv)
                                      for mapping decoder dirs back to
                                      activation space (steering).

Output: results/shared_subspace_diag.npy + the two datadirs.
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge

SRC = Path("sae_data/hetdynamics_eqvar")
N_TRAIN = 85          # match train_sae_mixed.py VAL_FRAC=0.15 split
N_MODES = 8

acts = np.load(SRC / "activations_full.npy")          # (100, 8, T, 256)
Z = np.load(SRC / "Z_full.npy")
n_real, n_modes, T, D = acts.shape
rng = np.random.default_rng(0)

# ── 1. mode-agnostic readout + deflation flatness ────────────────────────────
Xm = acts[:N_TRAIN].reshape(-1, D).astype(np.float64)
ym = Z[:N_TRAIN].reshape(-1)
sel = rng.choice(len(ym), 80000, replace=False)

def crossmode_r(coef):
    return np.array([abs(np.corrcoef(
        acts[N_TRAIN:, j].reshape(-1, D).astype(np.float64) @ coef,
        Z[N_TRAIN:, j].reshape(-1))[0, 1]) for j in range(N_MODES)])

defl_dirs, defl_r = [], []
Xtr = Xm[sel].copy()
for it in range(4):
    r = Ridge(alpha=10.0).fit(Xtr, ym[sel])
    w = r.coef_ / np.linalg.norm(r.coef_)
    coef = r.coef_.copy()
    for v in defl_dirs:                      # evaluate in deflated space
        coef = coef - (coef @ v) * v
    defl_dirs.append(w)
    rs = crossmode_r(r.coef_ if it == 0 else coef)
    defl_r.append(rs)
    print(f"deflation {it}: mean cross-mode |r| = {rs.mean():.3f}  "
          f"per-mode {np.round(rs, 2)}")
    Xtr = Xtr - np.outer(Xtr @ w, w)
w_shared = defl_dirs[0]

# ── 2. cross-application matrix of per-mode readouts ─────────────────────────
Wfit = []
for j in range(N_MODES):
    X = acts[:N_TRAIN, j].reshape(-1, D); y = Z[:N_TRAIN, j].reshape(-1)
    s2 = rng.choice(len(y), 40000, replace=False)
    Wfit.append(Ridge(alpha=10.0).fit(X[s2], y[s2]))
Rcross = np.zeros((N_MODES, N_MODES))
for j, w in enumerate(Wfit):
    for jp in range(N_MODES):
        pred = w.predict(acts[N_TRAIN:, jp].reshape(-1, D))
        Rcross[j, jp] = abs(np.corrcoef(pred, Z[N_TRAIN:, jp].reshape(-1))[0, 1])
uniq = np.array([Rcross[j, j] - np.delete(Rcross[j], j).max()
                 for j in range(N_MODES)])
print("readout uniqueness (own − best other):", np.round(uniq, 3))

# ── 3a. datadir: k=1 shared-direction projection ─────────────────────────────
proj_dir = Path("sae_data/hetdynamics_eqvar_proj")
proj_dir.mkdir(exist_ok=True)
Xp = acts.astype(np.float64)
Xp = Xp - (Xp @ w_shared)[..., None] * w_shared
np.save(proj_dir / "activations_full.npy", Xp.astype(np.float32))
if not (proj_dir / "Z_full.npy").exists():
    os.symlink(os.path.abspath(SRC / "Z_full.npy"), proj_dir / "Z_full.npy")
np.save(proj_dir / "shared_direction.npy", w_shared)
print(f"wrote {proj_dir}/ (k=1 projection)")

# ── 3b. datadir: PCA whitening (train-fit, eigenvalue floor) ─────────────────
wh_dir = Path("sae_data/hetdynamics_eqvar_whiten")
wh_dir.mkdir(exist_ok=True)
mu = Xm.mean(0)
cov = np.cov(Xm[rng.choice(len(Xm), 200000, replace=False)].T)
lam, V = np.linalg.eigh(cov)
lam_f = np.maximum(lam, 1e-6 * lam.max())
Wm = V @ np.diag(lam_f ** -0.5) @ V.T                # ZCA whitening (symmetric)
Wm_inv = V @ np.diag(lam_f ** 0.5) @ V.T
Xw = (acts.astype(np.float64) - mu) @ Wm
np.save(wh_dir / "activations_full.npy", Xw.astype(np.float32))
if not (wh_dir / "Z_full.npy").exists():
    os.symlink(os.path.abspath(SRC / "Z_full.npy"), wh_dir / "Z_full.npy")
np.savez(wh_dir / "whiten_params.npz", mu=mu, Wm=Wm, Wm_inv=Wm_inv, lam=lam)
print(f"wrote {wh_dir}/ (ZCA whitening; lam range {lam.min():.3e}..{lam.max():.3e})")

os.makedirs("results", exist_ok=True)
np.save("results/shared_subspace_diag.npy",
        dict(w_shared=w_shared, deflation_dirs=np.stack(defl_dirs),
             deflation_crossmode_r=np.stack(defl_r), cross_application=Rcross,
             readout_uniqueness=uniq, n_train=N_TRAIN,
             note="deflation flat => shared signal high-rank; readout "
                  "uniqueness <=0.024 => ~no mode-unique linear structure"),
        allow_pickle=True)
print("saved -> results/shared_subspace_diag.npy")
