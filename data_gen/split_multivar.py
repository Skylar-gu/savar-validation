"""
R3 MULTIVARIATE data split — channel-aware fork of data_split.py.

Splits each realisation 70/15/15 along the time axis, PRESERVING the channel
axis. Multivar observations are stored (C, L, T); this reshapes each channel to
(T, ny, nx) and stacks them as (T, C, ny, nx) — the layout the C-channel CNN
forecaster (train/cnn/cnn_forecaster_multivar.py) consumes.

Per requirements: split only for CNN training. Causal discovery / the battery
use the full-length (C, L, T) observations from the realisation dir directly.

Env: SPLIT_REAL_DIR (data/realisations_multivar2),
     SPLIT_OUT_DIR  (data/splits_multivar2)
"""

import numpy as np
import os, glob

REAL_DIR  = os.environ.get("SPLIT_REAL_DIR", os.path.join("data", "realisations_multivar2"))
SPLIT_DIR = os.environ.get("SPLIT_OUT_DIR", os.path.join("data", "splits_multivar2"))

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

files = sorted(glob.glob(os.path.join(REAL_DIR, "realisation_*.npz")))
assert len(files) > 0, f"No realisations found in {REAL_DIR}"

# read shapes from the first file to compute split indices
d0 = np.load(files[0])
C, L, T = d0["observations"].shape          # (C, L, T)
ny = nx = int(L ** 0.5)

n_train = int(0.70 * T)
n_val   = int(0.15 * T)
n_test  = T - n_train - n_val

split_idx = {
    "train": (0,               n_train),
    "val":   (n_train,         n_train + n_val),
    "test":  (n_train + n_val, T),
}

print(f"Splitting {len(files)} realisations  C={C}  T={T}  "
      f"(train={n_train}, val={n_val}, test={n_test})")

for fpath in files:
    name = os.path.basename(fpath)              # realisation_NNN.npz
    d    = np.load(fpath)

    obs  = d["observations"]                     # (C, L, T)
    Z    = d["latent_states"]                    # (NC, T)
    # (C, L, T) → (T, C, ny, nx): transpose to (T, C, L) then reshape spatial
    obs_spatial = np.transpose(obs, (2, 0, 1)).reshape(T, C, ny, nx)
    Z_T         = Z.T                            # (T, NC)

    for split, (a, b) in split_idx.items():
        np.savez_compressed(
            os.path.join(SPLIT_DIR, split, name),
            observations       = obs_spatial[a:b],   # (T_split, C, ny, nx)
            latent_states      = Z_T[a:b],           # (T_split, NC)
            ground_truth_graph = d["ground_truth_graph"],
            channel_edges      = d["channel_edges"],
            W                  = d["W"],
            W_plus             = d["W_plus"],
            mv_meta            = d["mv_meta"],
            metadata           = d["metadata"],
        )

print(f"Done. Splits written to {SPLIT_DIR}/{{train,val,test}}/")
print(f"  train obs shape: ({n_train}, {C}, {ny}, {nx})")
print(f"  val   obs shape: ({n_val},  {C}, {ny}, {nx})")
print(f"  test  obs shape: ({n_test},  {C}, {ny}, {nx})")
print(f"  CNN windows per split (k=3): train={n_train-3}, val={n_val-3}, test={n_test-3}")
print(f"  × {len(files)} realisations = "
      f"train={len(files)*(n_train-3):,}  val={len(files)*(n_val-3):,}  "
      f"test={len(files)*(n_test-3):,} total windows")
