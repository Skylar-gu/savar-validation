"""Persistence (y_{t+1} = y_t) RMSE on the all-knobs val split, for context
against the CNN val RMSE. The oracle floor is not analytic for the nonlinear
generator, so only the persistence bracket is reported."""
import numpy as np, glob, os
files = sorted(glob.glob(os.path.join("data", "splits_allknobs", "val", "realisation_*.npz")))
se, n = 0.0, 0
for f in files:
    obs = np.load(f)["observations"].astype(np.float64)   # (T, ny, nx)
    d = obs[1:] - obs[:-1]
    se += (d ** 2).sum(); n += d.size
print(f"Persistence val RMSE (all-knobs): {np.sqrt(se / n):.4f}  over {len(files)} realisations")
