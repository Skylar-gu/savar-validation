"""Split realisations_finecadence 70/15/15 (along time) → data/splits_finecadence/

Mirrors split_nonlinear.py. Carries nl_meta / ng_meta / fine_edges through so the
nonlinearity, non-Gaussianity, and fine-lag settings stay recoverable per split.
The GNN forecaster (train/gnn/gnn_forecaster.py) reads "observations" of shape
(T_split, ny, nx) from these splits.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import os, glob

REAL_DIR  = os.environ.get("FC_REAL_DIR", os.path.join("data", "realisations_finecadence"))
SPLIT_DIR = os.environ.get("FC_SPLIT_DIR", os.path.join("data", "splits_finecadence"))

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

files = sorted(glob.glob(os.path.join(REAL_DIR, "realisation_*.npz")))
assert files, f"No realisations found in {REAL_DIR}"

d0 = np.load(files[0])
T  = d0["observations"].shape[1]

n_train = int(0.70 * T)
n_val   = int(0.15 * T)
n_test  = T - n_train - n_val
split_idx = {"train": (0, n_train),
             "val":   (n_train, n_train + n_val),
             "test":  (n_train + n_val, T)}

print(f"Splitting {len(files)} realisations  (train={n_train}, val={n_val}, test={n_test})")

PASSTHROUGH = ("ground_truth_graph", "W", "W_plus", "fine_edges",
               "nl_meta", "ng_meta", "metadata")

for fpath in files:
    name = os.path.basename(fpath)
    d    = np.load(fpath)
    obs  = d["observations"]                       # (L, T)
    Z    = d["latent_states"]                      # (N, T)
    L_size = obs.shape[0]
    ny = nx = int(L_size ** 0.5)
    obs_spatial = obs.T.reshape(T, ny, nx)
    Z_T = Z.T
    const = {k: d[k] for k in PASSTHROUGH}
    for split, (a, b) in split_idx.items():
        np.savez_compressed(
            os.path.join(SPLIT_DIR, split, name),
            observations=obs_spatial[a:b].astype(np.float32),
            latent_states=Z_T[a:b].astype(np.float32),
            **const,
        )

print(f"Done → {SPLIT_DIR}/")
