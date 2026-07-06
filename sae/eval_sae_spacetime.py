"""T6a — score space-time mixed SAEs (SAE_SPACETIME=ST) with the Block-C suite.

Same Hungarian-MCC / matched-F1 / uniqueness metrics as eval_sae_metrics.py, but
the per-mode streams are ST-frame windows (matching train_sae_mixed.py's
_flatten): stream_j[t] = acts[j, t:t+ST] flattened to ST*256. Z is aligned to
the CENTER frame of the window (most generous alignment for a moving-pattern
feature; identical to the ST=1 convention when ST=1). Scores FINAL checkpoints
(Block-C rule: never select on recon MSE); aggregates mean±sd across seeds.

Usage:
  python3 sae/eval_sae_spacetime.py --datadir sae_data_movmech_place \
      --st 3 --seeds 0,1,2 [--out results/foo.npy]
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_sae_metrics import load_sae, encode, metric_suite, print_suite, N_MODES

INPUT_DIM = 256

ap = argparse.ArgumentParser()
ap.add_argument("--datadir", required=True)
ap.add_argument("--st", type=int, required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--out", default=None)
a = ap.parse_args()
DATA_DIR = Path(a.datadir)
ST = a.st
SEEDS = [int(s) for s in a.seeds.split(",")]

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (R, 8, T, 256)
Z_full = np.load(DATA_DIR / "Z_full.npy")                 # (R, 8, T)
R, M, T, D = acts_full.shape
assert D == INPUT_DIM
Tp = T - ST + 1
off = ST // 2   # center-frame alignment


def mode_stream(j):
    """(R*Tp, ST*256) windowed stream for mode j (mirror of train _flatten)."""
    aj = acts_full[:, j]                                   # (R, T, 256)
    if ST == 1:
        return aj.reshape(-1, D)
    w = np.stack([aj[:, t:t+ST] for t in range(Tp)], axis=1)   # (R, Tp, ST, D)
    return w.reshape(-1, ST * D)


Zs = [Z_full[:, j, off:off + Tp].reshape(-1) for j in range(N_MODES)]

per_seed = {}
for seed in SEEDS:
    name = f"sae_mixed_topk_seed{seed}" + (f"_st{ST}" if ST > 1 else "") + "_final.pt"
    sae, m, s = load_sae(DATA_DIR / name)
    codes = [encode(sae, mode_stream(j), m, s) for j in range(N_MODES)]
    r = metric_suite(codes, Zs)
    print_suite(f"{DATA_DIR.name} ST={ST} seed={seed}", "mixed", r)
    per_seed[seed] = r
    del codes

mcc = np.array([per_seed[s]["mcc"] for s in SEEDS])
unq = np.array([per_seed[s]["mean_uniqueness"] for s in SEEDS])
f1 = np.array([per_seed[s]["mean_f1"] for s in SEEDS])
print(f"\n### {DATA_DIR.name}  ST={ST}  ({len(SEEDS)} seeds) ###")
print(f"  MCC        {mcc.mean():.4f} ± {mcc.std(ddof=1):.4f}   {np.round(mcc, 4).tolist()}")
print(f"  uniqueness {unq.mean():+.4f} ± {unq.std(ddof=1):.4f}   {np.round(unq, 4).tolist()}")
print(f"  matched F1 {f1.mean():.4f} ± {f1.std(ddof=1):.4f}   {np.round(f1, 4).tolist()}")

if a.out:
    np.save(a.out, dict(datadir=str(DATA_DIR), st=ST, seeds=SEEDS, per_seed=per_seed,
                        mcc=mcc, uniqueness=unq, f1=f1), allow_pickle=True)
    print(f"saved -> {a.out}")
