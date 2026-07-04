"""Split realisations_movmech_* 70/15/15 (along time) -> data/splits_movmech_*.

Mirrors split_finecadence.py. Trains on the D_sub coarse observations by
default; set MM_OBS_KEY=observations_avg (+ its own MM_SPLIT_DIR) to build the
D_avg-control training splits without regenerating.
Carries Z_fine / centres / sub_meta / W maps through per split.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import os, glob

REAL_DIR  = os.environ.get("MM_REAL_DIR", os.path.join("data", "realisations_movmech_place"))
SPLIT_DIR = os.environ.get("MM_SPLIT_DIR", os.path.join("data", "splits_movmech_place"))
OBS_KEY   = os.environ.get("MM_OBS_KEY", "observations")

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

files = sorted(glob.glob(os.path.join(REAL_DIR, "realisation_*.npz")))
assert files, f"No realisations found in {REAL_DIR}"

d0 = np.load(files[0])
T  = d0[OBS_KEY].shape[1]

n_train = int(0.70 * T)
n_val   = int(0.15 * T)
split_idx = {"train": (0, n_train),
             "val":   (n_train, n_train + n_val),
             "test":  (n_train + n_val, T)}

print(f"Splitting {len(files)} realisations  (obs={OBS_KEY}; "
      f"train={n_train}, val={n_val}, test={T - n_train - n_val})")

PASSTHROUGH = ("ground_truth_graph", "W", "W_sub_coarse", "centres", "sub_pos",
               "sub_meta", "fine_edges", "nl_meta", "metadata")
TIMESLICED  = ("latent_states", "Z_fine")   # (n, T) arrays sliced along time

for fpath in files:
    name = os.path.basename(fpath)
    d    = np.load(fpath)
    obs  = d[OBS_KEY]                              # (2500, T)
    L_size = obs.shape[0]
    ny = nx = int(L_size ** 0.5)
    obs_spatial = obs.T.reshape(T, ny, nx)
    const = {k: d[k] for k in PASSTHROUGH}
    for split, (a, b) in split_idx.items():
        np.savez_compressed(
            os.path.join(SPLIT_DIR, split, name),
            observations=obs_spatial[a:b].astype(np.float32),
            **{k: d[k].T[a:b].astype(np.float32) for k in TIMESLICED},
            **const,
        )

print(f"Done -> {SPLIT_DIR}/")
