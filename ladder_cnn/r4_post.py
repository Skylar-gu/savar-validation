"""R4 null calibration, resumed from the cached CNN passes.

rung_r4.py completed both CNN passes and cached r4_series_mean.npy /
r4_footprints.npy / spatial_sae_base.pt, then died in its null stage on the same
missing-import bug that hit nulls.py. Nothing expensive is repeated here.

Optimisation vs the original: PCMCI is run ONCE per draw and the resulting edge
set is scored under BOTH matchers, halving the cost.
"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from common import *
from common import _init_worker, _pool_worker
from concurrent.futures import ProcessPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, default=100)
ap.add_argument("--workers", type=int, default=4)
A = ap.parse_args()

G, gt = load_gt()
S_all = np.load(OUT / "r4_series_mean.npy")             # (15, 497, 512)
foot = np.load(OUT / "r4_footprints.npy")               # (512, 2500)
Rn, T, _ = S_all.shape
W_true = np.load(ROOT / "data/realisations/realisation_000.npz")["W"].astype(np.float64)
Zf = np.asarray(np.load(ROOT / "sae_data/base/Z_full.npy")[:Rn])
flatZ = Zf.transpose(0, 2, 1).reshape(-1, N_MODES)
assert np.isfinite(S_all).all() and np.isfinite(foot).all(), "GATE FAIL"
print(f"R4 cache: series {S_all.shape}, footprints {foot.shape}")
print(f"  all-zero feature series: "
      f"{int((np.abs(S_all).sum((0,1))==0).sum())}/512")
fw = np.maximum(foot, 0)
fw = fw / np.maximum(fw.sum(1, keepdims=True), 1e-12)

chosen = sel_var(S_all, n_max=12)
S_obs = S_all[:, :, chosen]
flat = S_all.reshape(-1, N_FEATURES)
alive = np.where(((flat != 0).mean(0) >= 0.02) & (flat.var(0) > 1e-12))[0]
print(f"  live candidates {len(alive)}/512   chosen {chosen}")


def score_both(S, fp):
    """One PCMCI pass per realisation, scored under BOTH matchers."""
    mp_f, _ = map_foot(fp, W_true)
    mp_r, _ = map_r(S.reshape(-1, S.shape[-1]), flatZ)
    f1f, f1r = [], []
    for r in range(S.shape[0]):
        det, _, _ = pcmci_one(S[r])
        for mp, acc in ((mp_f, f1f), (mp_r, f1r)):
            tp, fp_, fn = hungarian_strict_score(det, mp, gt)
            acc.append(prf(tp, fp_, fn)[2])
    return float(np.mean(f1f)), float(np.mean(f1r)), len(mp_f), len(mp_r)


def _draw(a):
    kind, seed = a
    rng = np.random.default_rng(seed)
    if kind == "rand":
        cols = rng.choice(alive, size=len(chosen), replace=False)
        return kind, score_both(S_all[:, :, cols], fw[cols])
    S = np.empty_like(S_obs)
    for r in range(S_obs.shape[0]):
        for c in range(S_obs.shape[-1]):
            S[r, :, c] = phase_randomise(S_obs[r, :, c].astype(np.float64), rng)
    return kind, score_both(S, fw[chosen])


print("\n[observed]")
of, orr, nf, nr = score_both(S_obs, fw[chosen])
print(f"  R4 MAP-FOOT F1={of:.4f} matched={nf}/8    MAP-R F1={orr:.4f} matched={nr}/8")

jobs = [(k, 5000 + 13 * i) for k in ("rand", "phase") for i in range(A.draws)]
NULL = {("rand", "F"): [], ("rand", "R"): [], ("phase", "F"): [], ("phase", "R"): []}
t0 = time.time(); done = 0
with ProcessPoolExecutor(max_workers=A.workers, initializer=_init_worker) as ex:
    for kind, (f1f, f1r, _, _) in ex.map(_draw, jobs, chunksize=1):
        NULL[(kind, "F")].append(f1f); NULL[(kind, "R")].append(f1r)
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(jobs)} {el:.0f}s eta {el/done*(len(jobs)-done):.0f}s")

print("\n" + "=" * 92)
print(f"{'null':<8}{'matcher':<10}{'obs':>8}{'mean':>8}{'sd':>8}{'min':>8}"
      f"{'p50':>8}{'p95':>8}{'max':>8}{'p-value':>10}")
SUM = {}
for kind in ("rand", "phase"):
    for tagm, o in (("F", of), ("R", orr)):
        arr = np.array(NULL[(kind, tagm)])
        pv = (1 + int((arr >= o).sum())) / (1 + len(arr))
        SUM[f"{kind}|{tagm}"] = dict(obs=o, mean=arr.mean(), sd=arr.std(),
                                     mn=arr.min(), p50=np.percentile(arr, 50),
                                     p95=np.percentile(arr, 95), mx=arr.max(),
                                     pval=pv, draws=arr)
        s = SUM[f"{kind}|{tagm}"]
        lab = "MAP-FOOT" if tagm == "F" else "MAP-R"
        print(f"{kind:<8}{lab:<10}{o:>8.4f}{s['mean']:>8.4f}{s['sd']:>8.4f}"
              f"{s['mn']:>8.4f}{s['p50']:>8.4f}{s['p95']:>8.4f}{s['mx']:>8.4f}"
              f"{pv:>10.4f}")

np.save(OUT / "r4_null.npy", dict(summary=SUM, chosen=chosen,
                                  obs_foot=of, obs_r=orr,
                                  n_matched_foot=nf, n_matched_r=nr,
                                  n_real=Rn, alive=alive), allow_pickle=True)
print(f"\nsaved -> {OUT}/r4_null.npy   ({time.time()-t0:.0f}s)")
