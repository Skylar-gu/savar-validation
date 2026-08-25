"""GNN PORT (MeshGNN, hetdynamics_eqvar, stride-1 activations). Otherwise identical to ../savar_sae_pcmci/nulls.py.
"""
"""Guardrail #9 sides (i) and (ii): the null F1 DISTRIBUTION with the SAME
selection-and-matching freedom, and the observed rung's p-value inside it.

Nulls (PREREG §4):
  N-RAND  : variables drawn at random from the live candidate pool, then the
            identical discovery + Hungarian matching.
  N-PHASE : the observed selected series, each independently phase-randomised
            (autocorrelation + marginal spectrum preserved, cross-dependence
            destroyed), then the identical discovery + matching.
  N-SHIFT : the observed selected series, each independently circularly shifted.

Every null is scored at the SAME R_null as the observed value it is compared
against, so the comparison is like-for-like.
"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from scipy.stats import skew
from common import *
from common import _init_worker
from concurrent.futures import ProcessPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--sae_dir", default="sae_data/hetdynamics_eqvar")
ap.add_argument("--tag", default="gnn_eqvar")
ap.add_argument("--r_null", type=int, default=15)
ap.add_argument("--draws", type=int, default=200)
ap.add_argument("--workers", type=int, default=24)
A = ap.parse_args()

SAE_DIR = ROOT / A.sae_dir
Rn = A.r_null
G, gt = load_gt()
MAP_ID = {j: j for j in range(N_MODES)}

acts = np.asarray(np.load(ACTS_CACHE, mmap_mode="r")[:Rn])   # STRIDE 1
Zf = np.asarray(np.load(Z_CACHE)[:Rn])
align = np.load(SAE_DIR / "alignment_per_mode.npy", allow_pickle=True).item()
Rn, _, T_eff, _ = acts.shape
flatZ = Zf.transpose(0, 2, 1).reshape(-1, N_MODES)
print(f"# nulls  sae_dir={A.sae_dir}  R_null={Rn}  draws={A.draws}")

print("[encode] per-mode + mixed")
code_pm = np.empty((Rn, N_MODES, T_eff, N_FEATURES), np.float32)
for j in range(N_MODES):
    sae, mu, sd = load_sae(SAE_DIR / f"sae_mode_{j}.pt")
    code_pm[:, j] = encode_block(sae, acts[:, j].reshape(-1, INPUT_DIM), mu, sd
                                 ).reshape(Rn, T_eff, N_FEATURES)
mixed_path = SAE_DIR / "sae_mixed.pt"
sae, mu, sd = load_sae(mixed_path)
code_mx = encode_block(sae, acts.reshape(-1, INPUT_DIM), mu, sd
                       ).reshape(Rn, N_MODES, T_eff, N_FEATURES)
pooled = code_mx.mean(axis=1)
print(f"  mixed SAE: {mixed_path}")


def alive_cols(X):
    """X (R,T,C) -> indices with nz>=2% and var>0."""
    f = X.reshape(-1, X.shape[-1])
    return np.where(((f != 0).mean(0) >= 0.02) & (f.var(0) > 1e-12))[0]


def signflip(S):
    return S * np.array([float(np.sign(skew(S[..., j].ravel()))) or 1.0
                         for j in range(S.shape[-1])])


# ── observed configurations at R_null ────────────────────────────────────────
best_feat = [int(align[j]["best_feat"]) for j in range(N_MODES)]
sign_or = [float(np.sign(align[j]["C_j"][best_feat[j]])) or 1.0
           for j in range(N_MODES)]
OBS_CFG = {}
OBS_CFG["R0"] = (np.stack([sign_or[j] * code_pm[:, j, :, best_feat[j]]
                           for j in range(N_MODES)], -1), "ID")
picks2 = [sel_var(code_pm[:, j], n_max=1)[0] for j in range(N_MODES)]
OBS_CFG["R2"] = (signflip(np.stack([code_pm[:, j, :, picks2[j]]
                                    for j in range(N_MODES)], -1)), "ID")
picks3a = [sel_var(code_mx[:, j], n_max=1)[0] for j in range(N_MODES)]
OBS_CFG["R3a"] = (signflip(np.stack([code_mx[:, j, :, picks3a[j]]
                                     for j in range(N_MODES)], -1)), "ID")
chosen3b = sel_var(pooled, n_max=12)
OBS_CFG["R3b"] = (pooled[:, :, chosen3b], "R")
print(f"  picks R2={picks2}  R3a={picks3a}  R3b N_hat={len(chosen3b)} {chosen3b}")


def score_series(S, maprule):
    """S (R,T,C) -> mean F1 with the declared matching freedom."""
    if maprule == "ID":
        mp = MAP_ID
    else:
        mp, _ = map_r(S.reshape(-1, S.shape[-1]), flatZ)
    r = run_ladder_rung(list(S), mp, gt, workers=1)
    return r["f1"].mean(), r["precision"].mean(), r["recall"].mean(), len(mp)


def _draw(args):
    kind, rung, seed = args
    rng = np.random.default_rng(seed)
    S_obs, maprule = OBS_CFG[rung]
    if kind == "rand":
        if rung == "R2":
            cols = [int(rng.choice(alive_cols(code_pm[:, j])))
                    for j in range(N_MODES)]
            S = signflip(np.stack([code_pm[:, j, :, cols[j]]
                                   for j in range(N_MODES)], -1))
        elif rung == "R3a":
            cols = [int(rng.choice(alive_cols(code_mx[:, j])))
                    for j in range(N_MODES)]
            S = signflip(np.stack([code_mx[:, j, :, cols[j]]
                                   for j in range(N_MODES)], -1))
        elif rung == "R3b":
            av = alive_cols(pooled)
            k = S_obs.shape[-1]
            cols = rng.choice(av, size=min(k, len(av)), replace=False)
            S = pooled[:, :, cols]
        else:
            raise ValueError(rung)
    elif kind == "phase":
        S = np.empty_like(S_obs)
        for r in range(S_obs.shape[0]):
            for c in range(S_obs.shape[-1]):
                S[r, :, c] = phase_randomise(S_obs[r, :, c].astype(np.float64), rng)
    elif kind == "shift":
        S = np.empty_like(S_obs)
        for r in range(S_obs.shape[0]):
            for c in range(S_obs.shape[-1]):
                S[r, :, c] = circ_shift(S_obs[r, :, c], rng)
    else:
        raise ValueError(kind)
    f1, p, rc, nm = score_series(S, maprule)
    return kind, rung, seed, f1, p, rc, nm


RUNGS = ["R3b"]          # GNN port: R3b only (compute budget)
KINDS = ["rand", "phase", "shift"]
OBS = {}
print("\n[observed at R_null]")
for rung in ["R0"] + RUNGS:
    S, mr = OBS_CFG[rung]
    f1, p, rc, nm = score_series(S, mr)
    OBS[rung] = dict(f1=f1, P=p, R=rc, n_matched=nm)
    print(f"  {rung:<5} F1={f1:.4f} P={p:.4f} R={rc:.4f} matched={nm}/8")

jobs = [(k, r, 10000 + 97 * i) for k in KINDS for r in RUNGS
        for i in range(A.draws)]
print(f"\n[nulls] {len(jobs)} draws total, {A.workers} workers")
t0 = time.time()
NULL = {(k, r): [] for k in KINDS for r in RUNGS}
done = 0
with ProcessPoolExecutor(max_workers=A.workers, initializer=_init_worker) as ex:
    for kind, rung, seed, f1, p, rc, nm in ex.map(_draw, jobs, chunksize=1):
        NULL[(kind, rung)].append((f1, p, rc, nm))
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(jobs)}  {el:.0f}s elapsed, "
                  f"eta {el/done*(len(jobs)-done):.0f}s")

print("\n" + "=" * 86)
print(f"{'rung':<6}{'null':<7}{'obs F1':>8}{'mean':>8}{'sd':>7}{'min':>7}"
      f"{'p05':>7}{'p50':>7}{'p95':>7}{'max':>7}{'p-value':>9}")
SUM = {}
for rung in RUNGS:
    for kind in KINDS:
        arr = np.array([x[0] for x in NULL[(kind, rung)]])
        o = OBS[rung]["f1"]
        pv = (1 + int((arr >= o).sum())) / (1 + len(arr))
        SUM[(rung, kind)] = dict(obs=o, mean=arr.mean(), sd=arr.std(),
                                 mn=arr.min(), p05=np.percentile(arr, 5),
                                 p50=np.percentile(arr, 50),
                                 p95=np.percentile(arr, 95), mx=arr.max(),
                                 pval=pv, n=len(arr), draws=arr)
        s = SUM[(rung, kind)]
        print(f"{rung:<6}{kind:<7}{o:>8.4f}{s['mean']:>8.4f}{s['sd']:>7.4f}"
              f"{s['mn']:>7.4f}{s['p05']:>7.4f}{s['p50']:>7.4f}"
              f"{s['p95']:>7.4f}{s['mx']:>7.4f}{pv:>9.4f}")

np.save(OUT / f"nulls_{A.tag}.npy",
        dict(obs=OBS, summary={f"{k[0]}|{k[1]}": v for k, v in SUM.items()},
             r_null=Rn, draws=A.draws, chosen3b=chosen3b,
             picks2=picks2, picks3a=picks3a),
        allow_pickle=True)
print(f"\nsaved -> {OUT}/nulls_{A.tag}.npy   ({time.time()-t0:.0f}s)")
