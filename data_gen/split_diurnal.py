"""Split realisations_diurnal 70/15/15 → data/splits_diurnal/

Mirrors split_dy005.py. Carries the cycle-forcing arrays through so the
ground-truth diurnal/annual forcing stays recoverable in each split.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import os, glob

REAL_DIR  = os.path.join("data", "realisations_diurnal")
SPLIT_DIR = os.path.join("data", "splits_diurnal")

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

files = sorted(glob.glob(os.path.join(REAL_DIR, "realisation_*.npz")))
assert len(files) > 0, f"No realisations found in {REAL_DIR}"

d0 = np.load(files[0])
T  = d0["observations"].shape[1]

n_train = int(0.70 * T)
n_val   = int(0.15 * T)
n_test  = T - n_train - n_val

split_idx = {
    "train": (0,               n_train),
    "val":   (n_train,         n_train + n_val),
    "test":  (n_train + n_val, T),
}

print(f"Splitting {len(files)} realisations  "
      f"(train={n_train}, val={n_val}, test={n_test})")

# arrays that are constant in time and copied verbatim into every split
PASSTHROUGH = ("ground_truth_graph", "W", "W_plus", "diurnal_amp",
               "diurnal_phase", "annual_amp", "annual_phase",
               "cycle_meta", "metadata")

for fpath in files:
    name = os.path.basename(fpath)
    d    = np.load(fpath)

    obs = d["observations"]                 # (L, T)
    Z   = d["latent_states"]                # (N, T)
    fl  = d["forcing_latent"]               # (N, T)

    L_size      = obs.shape[0]
    ny = nx     = int(L_size ** 0.5)
    obs_spatial = obs.T.reshape(T, ny, nx)
    Z_T         = Z.T                       # (T, N)
    fl_T        = fl.T                      # (T, N)

    const = {k: d[k] for k in PASSTHROUGH}

    for split, (a, b) in split_idx.items():
        np.savez_compressed(
            os.path.join(SPLIT_DIR, split, name),
            observations   = obs_spatial[a:b],
            latent_states  = Z_T[a:b],
            forcing_latent = fl_T[a:b],
            **const,
        )

print(f"Done → {SPLIT_DIR}/{{train,val,test}}/")
print(f"  CNN windows (k=3): train={len(files)*(n_train-3):,}  "
      f"val={len(files)*(n_val-3):,}  test={len(files)*(n_test-3):,}")
