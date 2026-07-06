"""
E2 — Aggregation-consistency scores as an UNSUPERVISED selector
(litext plan, notes/literature_extension_experiments.md §3; operationalizes arXiv:2505.10476's
score family for our pixel->mode setting).

Question: do consistency scores computable WITHOUT ground truth rank the E1
candidate aggregation maps in the same order as their ground-truth PCMCI F1?
If yes, we possess the unsupervised model-selection criterion the GraphCast
rung needs (there is no W and no Phi there).

Scores per candidate W-hat (all from observational data + W-hat only):
  S_dep   dependence consistency — aggregate-level dependent pairs (i,j)
          should also be dependent at the micro level: fraction of
          aggregate-dependent pairs whose member-pixel pairs (residualized on
          the same conditioning set) are dependent too.
  S_indep independence consistency — aggregate-level independent pairs should
          have independent member pixels (a merged-away or fabricated edge
          betrays itself here).
  S_suff  partial-aggregation sufficiency — a pixel in group i, residualized
          on its OWN aggregate (lags 0..tau_max), should carry no remaining
          dependence on other groups' aggregates. Merging two true modes
          fails this (each sub-blob has parents its merged aggregate hides).
  S_joint harmonic mean of the three.

Conditioning protocol (pragmatic FullCI): for a pair (i,j), residualize both
series on {Z-hat_k lags 0..tau_max, k not in {i,j}} + {own lags 1..tau_max};
dependence = max-|corr| over cross-lags -tau..tau with Fisher-z + Bonferroni.

Output: results/litext_e2_adag.npy (scores per candidate; Spearman vs E1 F1
if results/litext_e1_discovery.npy exists)
Env: E2_NREAL (6 eval reals), E2_ALPHA (0.01), E2_NPIX (6 pixels/group)
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
N_PIX   = int(os.environ.get("E2_NPIX", 6))
TAU_MAX = 6
SEED    = 0

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
rng = np.random.default_rng(SEED)

# candidates from E1 (full battery file if present, else footprints-only file)
src = None
for f in ("litext_e1_discovery.npy", "litext_e1_footprints.npy"):
    p = RES_DIR / f
    if p.exists():
        src = np.load(p, allow_pickle=True).item()
        break
assert src is not None, "run sae/discover_modes.py first"
CANDS = src["cands"]
print(f"candidates: {sorted(CANDS)} (from {p.name})")

OBS = [np.load(paths[ri])["observations"].astype(np.float64)
       for ri in range(N_REAL)]                       # each (2500, T)


def lag_stack(x, lags):
    """x: (T,) -> (T - max(lags), len(lags)) matrix of x(t - lag)."""
    mx = max(lags) if lags else 0
    T = len(x)
    return np.stack([x[mx - l: T - l] for l in lags], axis=1)


def residualize(y, conds):
    """OLS-residualize y (T',) on conds (T', d)."""
    if conds.shape[1] == 0:
        return y - y.mean()
    X = np.column_stack([conds, np.ones(len(conds))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def dep_test(a, b, conds, alpha):
    """max-|corr| dependence of residualized a,b over cross-lags; Bonferroni.
    a, b, conds already time-aligned (same T'). Returns (dependent?, min_p)."""
    ra, rb = residualize(a, conds), residualize(b, conds)
    T = len(ra)
    ps = []
    for tau in range(-TAU_MAX, TAU_MAX + 1):
        if tau >= 0:
            x, y = ra[:T - tau] if tau else ra, rb[tau:] if tau else rb
        else:
            x, y = ra[-tau:], rb[:T + tau]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        r = np.clip(r, -0.999999, 0.999999)
        z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(len(x) - 3)
        ps.append(2 * (1 - stats.norm.cdf(abs(z))))
    if not ps:
        return False, 1.0
    p = min(1.0, min(ps) * len(ps))
    return p < alpha, p


def consistency_scores(What, tag):
    C = What.shape[0]
    MX = TAU_MAX
    lags_all = list(range(0, MX + 1))
    lags_own = list(range(1, MX + 1))
    dep_agree, dep_tot = 0, 0
    ind_agree, ind_tot = 0, 0
    suff_ok, suff_tot = 0, 0
    for obs in OBS:
        Zh = What @ obs                              # (C, T)
        T = Zh.shape[1]
        Tp = T - MX
        # pre-stack lagged aggregates
        Lag = {k: lag_stack(Zh[k], lags_all) for k in range(C)}     # (Tp, MX+1)
        Own = {k: lag_stack(Zh[k], lags_own) for k in range(C)}     # (Tp, MX)
        cur = {k: Zh[k][MX:] for k in range(C)}
        # pixel members per group (top-weight)
        members = {k: np.argsort(-What[k])[:N_PIX] for k in range(C)}
        pix = {k: obs[members[k]][:, MX:] for k in range(C)}        # (N_PIX, Tp)
        for i in range(C):
            for j in range(i + 1, C):
                conds = np.column_stack(
                    [Lag[k] for k in range(C) if k not in (i, j)] +
                    [Own[i], Own[j]]) if C > 2 else \
                    np.column_stack([Own[i], Own[j]])
                agg_dep, _ = dep_test(cur[i], cur[j], conds, ALPHA)
                # micro verdicts on sampled cross pixel pairs
                pairs = [(a, b) for a in range(min(3, N_PIX))
                         for b in range(min(3, N_PIX))]
                micro = [dep_test(pix[i][a], pix[j][b], conds, ALPHA)[0]
                         for a, b in pairs]
                micro_frac = float(np.mean(micro))
                if agg_dep:
                    dep_tot += 1
                    dep_agree += micro_frac        # graded, not thresholded
                else:
                    ind_tot += 1
                    ind_agree += 1.0 - micro_frac
        # sufficiency: pixel residualized on OWN aggregate should be
        # independent of every other aggregate
        for i in range(C):
            other = np.column_stack([Lag[k] for k in range(C) if k != i]) \
                if C > 1 else np.zeros((Tp, 0))
            for a in range(min(3, N_PIX)):
                resid = residualize(pix[i][a], Lag[i])
                dep_any = False
                for k in range(C):
                    if k == i:
                        continue
                    d, _ = dep_test(resid, cur[k],
                                    np.column_stack([Lag[i], Own[k]]), ALPHA)
                    if d:
                        dep_any = True
                        break
                suff_tot += 1
                suff_ok += 0.0 if dep_any else 1.0
    s_dep = dep_agree / dep_tot if dep_tot else np.nan
    s_ind = ind_agree / ind_tot if ind_tot else np.nan
    s_suf = suff_ok / suff_tot if suff_tot else np.nan
    parts = [s for s in (s_dep, s_ind, s_suf) if np.isfinite(s) and s > 0]
    s_joint = len(parts) / sum(1.0 / s for s in parts) if parts else 0.0
    print(f"  {tag:<10} S_dep={s_dep:.3f} (n={dep_tot}) "
          f"S_indep={s_ind:.3f} (n={ind_tot}) S_suff={s_suf:.3f} "
          f"-> S_joint={s_joint:.3f}")
    return dict(S_dep=s_dep, S_indep=s_ind, S_suff=s_suf, S_joint=s_joint,
                n_dep=dep_tot, n_ind=ind_tot)


print(f"\n[E2] consistency scores ({N_REAL} reals, alpha={ALPHA})")
SCORES = {}
for name in sorted(CANDS):
    What = CANDS[name]
    if What.shape[0] < 2:
        print(f"  {name}: skipped (C<2)")
        continue
    SCORES[name] = consistency_scores(What, name)

# join with E1 F1s if the battery has finished
out = dict(scores=SCORES, n_real=N_REAL, alpha=ALPHA,
           note="E2 litext: operationalized Adag-style consistency scores; "
                "S_suff is our partial-aggregation sufficiency variant")
e1 = RES_DIR / "litext_e1_discovery.npy"
if e1.exists():
    g = np.load(e1, allow_pickle=True).item()["graph"]
    common = [k for k in SCORES if k in g]
    f1 = np.array([g[k]["F1"] for k in common])
    for s in ("S_dep", "S_indep", "S_suff", "S_joint"):
        v = np.array([SCORES[k][s] for k in common])
        fin = np.isfinite(v)
        if fin.sum() > 2:
            rho = stats.spearmanr(v[fin], f1[fin]).statistic
            print(f"\nSpearman({s}, truth-F1) = {rho:.3f} "
                  f"(n={int(fin.sum())})")
            out[f"spearman_{s}"] = float(rho)
    out["f1_table"] = {k: float(g[k]["F1"]) for k in common}

os.makedirs(RES_DIR, exist_ok=True)
np.save(RES_DIR / "litext_e2_adag.npy", out, allow_pickle=True)
print("\nsaved -> results/litext_e2_adag.npy")
