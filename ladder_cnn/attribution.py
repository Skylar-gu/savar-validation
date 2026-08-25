"""Attribution: WHY does R3b collapse?

R3b differs from R3a in three ways at once — the mode partition is gone, N is
not given (N_hat=12 > 8), and the matcher is MAP-R rather than MAP-ID. This
decomposes the drop with cheap controls, all under the frozen protocol.

  A1  true Z, 8 vars, MAP-ID                 — the ceiling (0.825)
  A2  true Z, 8 vars, MAP-R                  — cost of the matcher alone
  A3  true Z + 4 iid-noise vars, MAP-R       — cost of N_hat inflation alone
  A4  true Z + 4 global-mean vars, MAP-R     — N_hat inflation WITH redundancy
  A5  R3b restricted to its 8 MATCHED vars   — do the 4 unmatched vars cause it?
  A6  R3b with N_hat forced to 8 (oracle N)  — give N back, keep everything else
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from common import *

WORKERS = 2
G, gt = load_gt()
MAP_ID = {j: j for j in range(N_MODES)}
r3b = np.load(OUT / "r3b_series_base.npy", allow_pickle=True).item()
S3b = r3b["series"]                       # (100, 497, 12)
Rn, T, C = S3b.shape
Zf = np.asarray(np.load(ROOT / "sae_data/base/Z_full.npy")[:Rn]).astype(np.float64)
Zs = Zf.transpose(0, 2, 1)                # (R, T, 8)
flatZ = Zs.reshape(-1, N_MODES)
rng = np.random.default_rng(0)
OUTD = {}


def go(name, S, mp, note=""):
    r = run_ladder_rung(list(S), mp, gt, workers=WORKERS)
    OUTD[name] = dict(f1=r["f1"], P=r["precision"], R=r["recall"],
                      tp=r["tp"], fp=r["fp"], fn=r["fn"], n_matched=len(mp))
    print(f"  {name:<5} F1={r['f1'].mean():.4f} P={r['precision'].mean():.4f} "
          f"R={r['recall'].mean():.4f} TP/FP/FN="
          f"{r['tp'].mean():.1f}/{r['fp'].mean():.1f}/{r['fn'].mean():.1f} "
          f"matched={len(mp)}/8  {note}")


print("Attribution decomposition (100 realisations, frozen protocol)")
go("A1", Zs, MAP_ID, "true Z, MAP-ID  [ceiling]")

mp, M = map_r(Zs.reshape(-1, N_MODES), flatZ)
go("A2", Zs, mp, f"true Z, MAP-R  (matcher recovers identity: {mp == MAP_ID})")

noise = rng.standard_normal((Rn, T, 4))
S = np.concatenate([Zs, noise], -1)
mp, M = map_r(S.reshape(-1, 12), flatZ)
go("A3", S, mp, f"true Z + 4 iid noise, N_hat=12, matched={len(mp)}")

gm = Zs.mean(-1, keepdims=True)
S = np.concatenate([Zs, np.repeat(gm, 4, -1)
                    + 0.1 * rng.standard_normal((Rn, T, 4))], -1)
mp, M = map_r(S.reshape(-1, 12), flatZ)
go("A4", S, mp, f"true Z + 4 global-mean copies, matched={len(mp)}")

keep = sorted(r3b["mapping"].keys())
mp = {i: r3b["mapping"][c] for i, c in enumerate(keep)}
go("A5", S3b[:, :, keep], mp, f"R3b restricted to its 8 matched vars {keep}")

# A6: give N back — SEL-VAR top-8 from the same pooled candidate pool
acts = np.asarray(np.load(ROOT / "sae_data/base/activations_full.npy",
                          mmap_mode="r")[:Rn])
sae, mu, sd = load_sae(ROOT / "sae_data/base/sae_best.pt")
pooled = encode_block(sae, acts.reshape(-1, INPUT_DIM), mu, sd
                      ).reshape(Rn, N_MODES, T, N_FEATURES).mean(1)
del acts
ch8 = sel_var(pooled, n_max=8)
S = pooled[:, :, ch8]
mp, M = map_r(S.reshape(-1, 8), flatZ)
go("A6", S, mp, f"R3b with oracle N=8, chosen={ch8}, matched={len(mp)}, "
                f"maxr={np.round(M.max(1),2).tolist()}")

np.save(OUT / "attribution.npy", OUTD, allow_pickle=True)
print(f"\nsaved -> {OUT}/attribution.npy")
