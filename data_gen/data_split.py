"""
Phase 1 — Data Split
Splits each realisation 70/15/15 along the time axis.
Writes train/val/test npz files preserving latent_states and graph.

Per requirements: split only for CNN training.
Causal discovery uses the full 500 timestep observations.
"""

import numpy as np
import os, glob

REAL_DIR  = os.path.join("data", "realisations")
SPLIT_DIR = os.path.join("data", "splits")

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

files = sorted(glob.glob(os.path.join(REAL_DIR, "realisation_*.npz")))
assert len(files) > 0, f"No realisations found in {REAL_DIR}"

# read T from the first file to compute split indices
d0  = np.load(files[0])
T   = d0["observations"].shape[1]   # 500

n_train = int(0.70 * T)             # 350
n_val   = int(0.15 * T)             # 75
n_test  = T - n_train - n_val       # 75

split_idx = {
    "train": (0,              n_train),
    "val":   (n_train,        n_train + n_val),
    "test":  (n_train + n_val, T),
}

print(f"Splitting {len(files)} realisations  "
      f"(train={n_train}, val={n_val}, test={n_test})")

for fpath in files:
    name = os.path.basename(fpath)          # realisation_NNN.npz
    d    = np.load(fpath)

    obs  = d["observations"]                # (L, T)
    Z    = d["latent_states"]               # (N, T)
    G    = d["ground_truth_graph"]
    W    = d["W"]
    Wp   = d["W_plus"]
    meta = d["metadata"]

    # obs reshaped to (T, ny, nx) for CNN; ny=nx=50 inferred from L
    L_size = obs.shape[0]
    ny = nx = int(L_size ** 0.5)
    obs_spatial = obs.T.reshape(T, ny, nx)  # (T, 50, 50)
    Z_T         = Z.T                       # (T, N)

    for split, (a, b) in split_idx.items():
        np.savez_compressed(
            os.path.join(SPLIT_DIR, split, name),
            observations       = obs_spatial[a:b],   # (T_split, 50, 50)
            latent_states      = Z_T[a:b],            # (T_split, N)
            ground_truth_graph = G,
            W                  = W,
            W_plus             = Wp,
            metadata           = meta,
        )

print(f"Done. Splits written to {SPLIT_DIR}/{{train,val,test}}/")
print(f"  train obs shape: ({n_train}, 50, 50)")
print(f"  val   obs shape: ({n_val},  50, 50)")
print(f"  test  obs shape: ({n_test},  50, 50)")
print(f"  CNN windows per split (k=3): train={n_train-3}, val={n_val-3}, test={n_test-3}")
print(f"  × {len(files)} realisations = "
      f"train={len(files)*(n_train-3):,}  val={len(files)*(n_val-3):,}  test={len(files)*(n_test-3):,} total windows")
