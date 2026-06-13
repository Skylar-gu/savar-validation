"""
Deseasonalize extracted SAE activations (and the aligned Z) by subtracting the
ensemble-mean climatology over realisations.

Every realisation shares the identical deterministic diurnal+annual forcing
while its noise is independent, so the mean over realisations at each
(mode, time, channel) is exactly the cycle-driven deterministic response.
Subtracting it leaves cycle-free anomalies — the activation-space analogue of
the PCMCI deseasonalization (Z − ensemble mean) that restored F1 to 0.825.

Cheap: operates on already-extracted activations, no CNN forward passes.

Reads  sae_data_diurnal/{activations_full,Z_full}.npy
Writes sae_data_diurnal_deseason/{activations_full,Z_full}.npy
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--dy005", action="store_true")
_args = _ap.parse_args()

base = "sae_data_diurnal" if _args.diurnal else ("sae_data_dy005" if _args.dy005 else "sae_data")
SRC = Path(base)
DST = Path(base + "_deseason")
DST.mkdir(exist_ok=True)

acts = np.load(SRC / "activations_full.npy")   # (R, M, T, C)
Z    = np.load(SRC / "Z_full.npy")             # (R, M, T)
print(f"Loaded activations {acts.shape}, Z {Z.shape} from {SRC}/")

acts_clim = acts.mean(axis=0, keepdims=True)   # (1, M, T, C) — cycle climatology
Z_clim    = Z.mean(axis=0, keepdims=True)      # (1, M, T)

acts_anom = (acts - acts_clim).astype(np.float32)
Z_anom    = (Z - Z_clim).astype(np.float32)

print(f"\n  {'Mode':<5} {'act var removed':>16} {'Z var removed':>15}")
print(f"  {'-'*38}")
for j in range(acts.shape[1]):
    a_rm = 1 - acts_anom[:, j].var() / (acts[:, j].var() + 1e-12)
    z_rm = 1 - Z_anom[:, j].var()    / (Z[:, j].var()    + 1e-12)
    print(f"  X{j:<4} {a_rm*100:>15.1f}% {z_rm*100:>14.1f}%")

np.save(DST / "activations_full.npy", acts_anom)
np.save(DST / "Z_full.npy", Z_anom)
print(f"\nSaved deseasonalized activations + Z → {DST}/")
print("  (cycle climatology removed; PC0~cycle R² should now be ≈ 0)")
