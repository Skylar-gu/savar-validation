"""
E2 v2 — composite unsupervised aggregation selector
(litext plan, notes/literature_extension_experiments.md §3 E2).

v1 (`aggregation_selection.py`) operationalized pure Adag-style level-
consistency and FAILED in a diagnostic way: diag8 (truth-F1 0.000) scored
0.97 — an aggregation that destroys all signal has no dependencies at either
level and is therefore VACUOUSLY consistent; fine16 (duplicate halves,
F1 0.013) is consistent too, its faithfulness pathology invisible to
level-agreement. Consistency is necessary, not sufficient, for selection
across arbitrary candidate maps.

v2 scores (all ground-truth-free; micro-variables are PARTIAL AGGREGATES —
random half-splits of each group's support — for testing power):

  S_info   dependency density among aggregates (fraction of pairs with a
           significant conditional cross-lag dependence). Vacuous maps -> 0.
  S_dup    non-redundancy: 1 − fraction of aggregate pairs that are
           near-deterministic (max unconditioned cross-lag |corr| > DUP_R).
           Catches splits/duplicates (the fine16 pathology).
  S_agree  micro–macro agreement on pairs where EITHER level shows signal
           (both-null pairs excluded — no vacuous credit): aggregate verdict
           vs majority of the 4 half-pair verdicts.
  S_suff   sufficiency: half-aggregate of group i, residualized on the full
           aggregate i, should be independent of every other aggregate
           (merged modes fail: each half retains parents the merge hides).
  S_total  = S_info^0.5 · S_dup · S_agree · S_suff  (any hard failure tanks)

Output: results/litext_e2_adag_v2.npy (+ Spearman vs E1 truth-F1)
Env: E2_NREAL (6), E2_ALPHA (0.01), E2_DUP_R (0.9), E2_SEED (0)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/realisations_hetdynamics_eqvar"
RES_DIR  = ROOT / "results"

N_REAL  = int(os.environ.get("E2_NREAL", 6))
ALPHA   = float(os.environ.get("E2_ALPHA", 0.01))
DUP_R   = float(os.environ.get("E2_DUP_R", 0.9))
TAU_MAX = 6
SEED    = int(os.environ.get("E2_SEED", 0))

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
rng = np.random.default_rng(SEED)

src = np.load(RES_DIR / "litext_e1_discovery.npy", allow_pickle=True).item()
CANDS = src["cands"]
F1S = {k: float(v["F1"]) for k, v in src["graph"].items()}
print(f"candidates: {sorted(CANDS)}")

OBS = [np.load(paths[ri])["observations"].astype(np.float64)
       for ri in range(N_REAL)]


def lag_stack(x, lags):
    mx = max(lags)
    T = len(x)
    return np.stack([x[mx - l: T - l] for l in lags], axis=1)


def residualize(y, conds):
    X = np.column_stack([conds, np.ones(len(conds))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def dep_p(a, b, conds):
    """min Bonferroni p over cross-lags for residualized dependence."""
    ra = residualize(a, conds) if conds is not None else a - a.mean()
    rb = residualize(b, conds) if conds is not None else b - b.mean()
    T = len(ra)
    ps = []
    for tau in range(-TAU_MAX, TAU_MAX + 1):
        if tau >= 0:
            x, y = (ra[:T - tau], rb[tau:]) if tau else (ra, rb)
        else:
            x, y = ra[-tau:], rb[:T + tau]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = np.clip(float(np.corrcoef(x, y)[0, 1]), -0.999999, 0.999999)
        z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(len(x) - 3)
        ps.append(2 * (1 - stats.norm.cdf(abs(z))))
    return min(1.0, min(ps) * len(ps)) if ps else 1.0


def max_xcorr(a, b):
    """max unconditioned cross-lag |corr| (duplicate detector)."""
    T = len(a)
    best = 0.0
    aa = a - a.mean(); bb = b - b.mean()
    for tau in range(-TAU_MAX, TAU_MAX + 1):
        if tau >= 0:
            x, y = (aa[:T - tau], bb[tau:]) if tau else (aa, bb)
        else:
            x, y = aa[-tau:], bb[:T + tau]
        d = x.std() * y.std()
        if d > 1e-12:
            best = max(best, abs(float(np.mean(x * y) / d)))
    return best


def score_v2(What, tag):
    C = What.shape[0]
    lags_all = list(range(0, TAU_MAX + 1))
    lags_own = list(range(1, TAU_MAX + 1))
    # random half-splits of each group's support (fixed per candidate)
    halves = []
    for c in range(C):
        sup = np.where(What[c] > 0.01 * What[c].max())[0]
        perm = rng.permutation(sup)
        h1, h2 = perm[::2], perm[1::2]
        w1 = np.zeros_like(What[c]); w1[h1] = What[c][h1]
        w2 = np.zeros_like(What[c]); w2[h2] = What[c][h2]
        s1, s2 = w1.sum(), w2.sum()
        halves.append((w1 / s1 if s1 > 0 else w1, w2 / s2 if s2 > 0 else w2))

    n_pairs = dep_cnt = dup_cnt = 0
    agree_num = agree_den = 0
    suff_num = suff_den = 0
    for obs in OBS:
        Zh = What @ obs
        Hh = np.stack([np.stack([h1 @ obs, h2 @ obs])
                       for h1, h2 in halves])            # (C, 2, T)
        Lag = {k: lag_stack(Zh[k], lags_all) for k in range(C)}
        Own = {k: lag_stack(Zh[k], lags_own) for k in range(C)}
        cur = {k: Zh[k][TAU_MAX:] for k in range(C)}
        curH = Hh[:, :, TAU_MAX:]
        for i in range(C):
            for j in range(i + 1, C):
                # condition ONLY on the other aggregates — including the
                # pair's own lags conditions away the lagged dependence
                # under test (v1 bug: oracle showed n_dep=0)
                conds = np.column_stack(
                    [Lag[k] for k in range(C) if k not in (i, j)]) \
                    if C > 2 else None
                n_pairs += 1
                agg_dep = dep_p(cur[i], cur[j], conds) < ALPHA
                dep_cnt += agg_dep
                if max_xcorr(Zh[i], Zh[j]) > DUP_R:
                    dup_cnt += 1
                micro = [dep_p(curH[i, a], curH[j, b], conds) < ALPHA
                         for a in range(2) for b in range(2)]
                micro_dep = sum(micro) >= 2                # majority of 4
                if agg_dep or any(micro):                  # signal somewhere
                    agree_den += 1
                    agree_num += (agg_dep == micro_dep)
        for i in range(C):
            for a in range(2):
                # residualize the half on its own FULL aggregate's lags;
                # test remaining dependence on other aggregates' (lagged)
                # values directly — conditioning on k's own lags would strip
                # exactly the lagged influence being tested (v1 bug)
                resid = residualize(curH[i, a], Lag[i])
                dep_any = any(dep_p(resid, cur[k], None) < ALPHA
                              for k in range(C) if k != i)
                suff_den += 1
                suff_num += (not dep_any)
    s_info = dep_cnt / n_pairs if n_pairs else 0.0
    s_dup = 1.0 - (dup_cnt / n_pairs if n_pairs else 1.0)
    s_agree = agree_num / agree_den if agree_den else 0.0   # vacuous -> 0
    s_suff = suff_num / suff_den if suff_den else 0.0
    s_total = np.sqrt(s_info) * s_dup * s_agree * s_suff
    print(f"  {tag:<10} S_info={s_info:.3f} S_dup={s_dup:.3f} "
          f"S_agree={s_agree:.3f} (n={agree_den}) S_suff={s_suff:.3f} "
          f"-> S_total={s_total:.3f}   [F1={F1S.get(tag, float('nan')):.3f}]")
    return dict(S_info=s_info, S_dup=s_dup, S_agree=s_agree, S_suff=s_suff,
                S_total=s_total, n_signal_pairs=agree_den)


print(f"\n[E2 v2] composite scores ({N_REAL} reals, alpha={ALPHA}, "
      f"dup_r={DUP_R})")
SCORES = {}
for name in sorted(CANDS):
    if CANDS[name].shape[0] < 2:
        continue
    SCORES[name] = score_v2(CANDS[name], name)

common = [k for k in SCORES if k in F1S]
f1 = np.array([F1S[k] for k in common])
out = dict(scores=SCORES, f1_table={k: F1S[k] for k in common},
           n_real=N_REAL, alpha=ALPHA, dup_r=DUP_R,
           note="E2 v2: consistency alone is gameable by signal destruction "
                "(v1 result); composite adds informativeness + non-redundancy")
print(f"\n{'score':<9} Spearman(score, truth-F1) over {len(common)} candidates")
for s in ("S_info", "S_dup", "S_agree", "S_suff", "S_total"):
    v = np.array([SCORES[k][s] for k in common])
    rho = stats.spearmanr(v, f1).statistic
    out[f"spearman_{s}"] = float(rho)
    print(f"  {s:<8} {rho:+.3f}")

os.makedirs(RES_DIR, exist_ok=True)
np.save(RES_DIR / "litext_e2_adag_v2.npy", out, allow_pickle=True)
print("\nsaved -> results/litext_e2_adag_v2.npy")
