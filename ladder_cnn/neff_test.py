"""Are the ParCorr p-values anticonservative at this zero-inflation?

R0/R2/R3a series are ~68% exact zeros and a variable PAIR co-fires on only ~52
of the 497 timesteps, but ParCorr computes its p-value as if n = 497. Direct
test: circularly shift each variable independently (destroys ALL cross-
dependence, preserves autocorrelation, marginal distribution and the exact
zero-inflation pattern), then count detections. A calibrated test at alpha=0.05
over the 112 cross-slots should fire on ~5.6 of them.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from common import *
from common import _init_worker
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import skew

RN, DRAWS, WORKERS = 20, 30, 2
G, gt = load_gt()
acts = np.asarray(np.load(ROOT / "sae_data/base/activations_full.npy",
                          mmap_mode="r")[:RN])
align = np.load(ROOT / "sae_data/base/alignment_per_mode.npy",
                allow_pickle=True).item()
T = acts.shape[2]
NSLOT = N_MODES * (N_MODES - 1) * TAU_MAX
print(f"R={RN} T={T}  cross-slots tested per realisation = {NSLOT}")
print(f"expected detections under a CALIBRATED test at alpha=0.05: "
      f"{0.05*NSLOT:.1f}\n")

code = np.empty((RN, N_MODES, T, N_FEATURES), np.float32)
for j in range(N_MODES):
    s, mu, sd = load_sae(ROOT / f"sae_data/base/sae_mode_{j}.pt")
    code[:, j] = encode_block(s, acts[:, j].reshape(-1, INPUT_DIM), mu, sd
                              ).reshape(RN, T, N_FEATURES)
bf = [int(align[j]["best_feat"]) for j in range(N_MODES)]
S_R0 = np.stack([code[:, j, :, bf[j]] for j in range(N_MODES)], -1)

# a dense (non-sparse) reference with the SAME marginal autocorrelation:
Zf = np.asarray(np.load(ROOT / "sae_data/base/Z_full.npy")[:RN])
S_Z = Zf.transpose(0, 2, 1).astype(np.float64)

CFG = {"R0_SAE_features(68% zeros)": S_R0, "trueZ(0% zeros)": S_Z}


def _one(a):
    tag, seed = a
    rng = np.random.default_rng(seed)
    S = CFG[tag]
    out = []
    for r in range(S.shape[0]):
        x = np.stack([circ_shift(S[r, :, c], rng) for c in range(S.shape[-1])], -1)
        det, _, _ = pcmci_one(x)
        out.append(len(det))
    return tag, out


jobs = [(t, 700 + 31 * i) for t in CFG for i in range(DRAWS)]
acc = {t: [] for t in CFG}
with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as ex:
    for tag, out in ex.map(_one, jobs):
        acc[tag] += out
for tag, v in acc.items():
    v = np.array(v)
    print(f"{tag:<32} detections/realisation under independent circular shift:")
    print(f"    mean {v.mean():.2f}  sd {v.std():.2f}  median {np.median(v):.0f} "
          f"max {v.max()}   => empirical type-I rate {v.mean()/NSLOT:.4f} "
          f"(nominal 0.0500, inflation {v.mean()/(0.05*NSLOT):.2f}x)")
np.save(OUT / "neff_test.npy", {k: np.array(v) for k, v in acc.items()},
        allow_pickle=True)
print(f"\nsaved -> {OUT}/neff_test.npy")
