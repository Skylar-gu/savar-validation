"""Deseasonalize a realisations dir by harmonic regression (robustness §4, seasonality row).

For every realisation, regresses out sin/cos components at the generating
periods from EVERY pixel series and every latent mode series, then writes a
parallel realisations dir with observations/latent_states replaced by the
residuals (all other npz keys passed through untouched).  This is the
"with deseasonalization" arm of the seasonality rung: same checkpoint, same
battery, preprocessed series.

Periods default to the npz's own `seas_periods` (saved by the seasonal
generators); DS_PERIODS overrides.

Env:
  DS_REAL_DIR   input realisations (default data/realisations_overlap02_seas)
  DS_OUT_DIR    output dir          (default <DS_REAL_DIR>_deseas)
  DS_PERIODS    comma list of periods in fine steps (default: from npz)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, glob, time
import numpy as np

REAL_DIR = os.environ.get("DS_REAL_DIR", os.path.join("data", "realisations_overlap02_seas"))
OUT_DIR  = os.environ.get("DS_OUT_DIR", REAL_DIR.rstrip("/") + "_deseas")
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(REAL_DIR, "realisation_*.npz")))
assert files, f"no realisations in {REAL_DIR}"

d0 = np.load(files[0])
if os.environ.get("DS_PERIODS"):
    periods = [float(x) for x in os.environ["DS_PERIODS"].split(",")]
elif "seas_periods" in d0:
    periods = [float(p) for p in np.atleast_1d(d0["seas_periods"])]
else:
    raise SystemExit("no seas_periods in npz and DS_PERIODS not set")

T = d0["observations"].shape[1]
t = np.arange(T, dtype=np.float64)
cols = [np.ones(T)]
for P in periods:
    cols += [np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P)]
X = np.column_stack(cols)                        # (T, 1+2P)
# hat-matrix pieces: resid = A - (A @ X) @ pinv(X'X) @ X'
XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)       # (k, T)

def deseason_rows(A):
    """A: (rows, T) -> residual after per-row harmonic regression."""
    beta = (XtX_inv_Xt @ A.T)                    # (k, rows)
    return A - (X @ beta).T

print(f"Deseasonalizing {len(files)} realisations  periods={periods}  T={T}")
print(f"  {REAL_DIR} -> {OUT_DIR}")

t0 = time.time()
for n, fpath in enumerate(files):
    d = np.load(fpath)
    obs = d["observations"].astype(np.float64)   # (L, T)
    Z   = d["latent_states"].astype(np.float64)  # (N, T)
    obs_ds = deseason_rows(obs)
    Z_ds   = deseason_rows(Z)
    rest = {k: d[k] for k in d.files if k not in ("observations", "latent_states")}
    np.savez_compressed(os.path.join(OUT_DIR, os.path.basename(fpath)),
                        observations=obs_ds.astype(np.float32),
                        latent_states=Z_ds.astype(np.float32),
                        **rest)
    if n == 0:
        rm = 1.0 - Z_ds.var(1) / Z.var(1)
        print("  variance removed per mode (real 0): "
              + " ".join(f"{v:+.3f}" for v in rm))
    if (n + 1) % 20 == 0:
        print(f"  [{n+1}/{len(files)}]  {time.time()-t0:.0f}s")

print(f"Done. {len(files)} realisations in {time.time()-t0:.0f}s -> {OUT_DIR}/")
