"""Guardrail #9 (iii): the negative control that must FAIL.

`litext_e1_discovery.npy` scored shift5/diag8 at F1 0.000 — but on
data/realisations_hetdynamics_eqvar, tau_max=6, run_pcmciplus(tau_min=0).
Those numbers are NOT comparable to this ladder. Re-run here on the BASE
dataset under the ladder's exact protocol (tau_min=1, tau_max=2, alpha 0.05,
pc_alpha 0.2, per-realisation-mean F1), under BOTH matching rules.

If shift5/diag8 do not fail, the scoring protocol is broken and the whole
ladder is an instrument failure.
"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from scipy.ndimage import gaussian_filter
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--n_real", type=int, default=40)
ap.add_argument("--workers", type=int, default=4)
A = ap.parse_args()

G, gt = load_gt()
paths = sorted((ROOT / "data/realisations").glob("realisation_*.npz"))[:A.n_real]
d0 = np.load(paths[0])
W = d0["W"].astype(np.float64)
Zf = np.asarray(np.load(ROOT / "sae_data/base/Z_full.npy")[:A.n_real])  # (R,8,497)
T_EFF = Zf.shape[2]
K_LAG = 3   # Z_full = latent_states[:, 3:]  (verified in gate 4)


def norm(M):
    return M / np.maximum(M.sum(1, keepdims=True), 1e-12)


CANDS = {"oracle": norm(W)}
CANDS["shift5"] = norm(np.stack([np.roll(W[j].reshape(NY, NX), 5, axis=1).reshape(L)
                                 for j in range(N_MODES)]))
CANDS["diag8"] = norm(np.stack([np.roll(np.roll(W[j].reshape(NY, NX), 8, axis=0),
                                        8, axis=1).reshape(L)
                                for j in range(N_MODES)]))
CANDS["blur"] = norm(np.stack([gaussian_filter(W[j].reshape(NY, NX), sigma=6).reshape(L)
                               for j in range(N_MODES)]))
CANDS["random8"] = None   # filled below: 8 random disjoint square patches
rng = np.random.default_rng(0)
patches = []
for _ in range(N_MODES):
    m = np.zeros((NY, NX))
    y0, x0 = rng.integers(0, NY - 16, 2)
    m[y0:y0 + 16, x0:x0 + 16] = 1.0
    patches.append(m.reshape(L))
CANDS["random8"] = norm(np.stack(patches))

# observations, pooled through each W-hat; aligned to the same time window the
# activations/Z use so every rung sees the same T.
OBS = [np.load(p)["observations"].astype(np.float64) for p in paths]   # (2500, 500)

print("=" * 74)
print(f"NEGATIVE CONTROL — misplaced footprints, base dataset, ladder protocol")
print(f"  n_real={A.n_real}  T_eff={T_EFF}  gt edges={len(gt)}")
res = {}
for name, What in CANDS.items():
    series = [(What @ o).T[K_LAG:K_LAG + T_EFF] for o in OBS]     # (T,8)
    mp_f, Mf = map_foot(What, W)
    flatS = np.concatenate(series, 0)
    flatZ = Zf.transpose(0, 2, 1).reshape(-1, N_MODES)
    mp_r, Mr = map_r(flatS, flatZ)
    for tagm, mp, Mm in (("MAP-FOOT", mp_f, Mf), ("MAP-R", mp_r, Mr)):
        t0 = time.time()
        r = run_ladder_rung(series, mp, gt, workers=A.workers)
        res[(name, tagm)] = dict(
            f1=r["f1"], precision=r["precision"], recall=r["recall"],
            tp=r["tp"], fp=r["fp"], fn=r["fn"], n_matched=len(mp),
            mapping=mp, best_score=Mm.max(1))
        print(f"  {name:<9} {tagm:<9} F1={r['f1'].mean():.4f} "
              f"P={r['precision'].mean():.4f} R={r['recall'].mean():.4f} "
              f"matched={len(mp)}/8  best_score={np.round(Mm.max(1),2).tolist()} "
              f"[{time.time()-t0:.0f}s]")

np.save(OUT / "negcontrol.npy",
        {f"{k[0]}|{k[1]}": v for k, v in res.items()}, allow_pickle=True)
print(f"\nsaved -> {OUT}/negcontrol.npy")
print("\nVERDICT: protocol is usable only if shift5 and diag8 score ~0 under "
      "MAP-FOOT.")
