"""Split realisations_dy005 70/15/15 → data/splits_dy005/"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import os, glob

REAL_DIR  = os.path.join("data", "realisations_dy005")
SPLIT_DIR = os.path.join("data", "splits_dy005")

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
    "train": (0,              n_train),
    "val":   (n_train,        n_train + n_val),
    "test":  (n_train + n_val, T),
}

print(f"Splitting {len(files)} realisations  "
      f"(train={n_train}, val={n_val}, test={n_test})")

for fpath in files:
    name = os.path.basename(fpath)
    d    = np.load(fpath)

    obs  = d["observations"]       # (L, T)
    Z    = d["latent_states"]
    G    = d["ground_truth_graph"]
    W    = d["W"]
    Wp   = d["W_plus"]
    meta = d["metadata"]

    L_size      = obs.shape[0]
    ny = nx     = int(L_size ** 0.5)
    obs_spatial = obs.T.reshape(T, ny, nx)
    Z_T         = Z.T

    for split, (a, b) in split_idx.items():
        np.savez_compressed(
            os.path.join(SPLIT_DIR, split, name),
            observations       = obs_spatial[a:b],
            latent_states      = Z_T[a:b],
            ground_truth_graph = G,
            W                  = W,
            W_plus             = Wp,
            metadata           = meta,
        )

print(f"Done → {SPLIT_DIR}/{{train,val,test}}/")
print(f"  CNN windows (k=3): train={len(files)*(n_train-3):,}  "
      f"val={len(files)*(n_val-3):,}  test={len(files)*(n_test-3):,}")
