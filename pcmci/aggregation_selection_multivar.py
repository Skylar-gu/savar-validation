"""
E2-MULTIVAR — Aggregation-consistency scores as an UNSUPERVISED selector, R3
multivariate rung (channel-aware fork of pcmci/aggregation_selection.py).

Identical score family (S_dep / S_indep / S_suff / S_joint) and conditioning
protocol as the parent. The ONLY change is the multivariate pixel field:
observations (C, L, T) are FLATTENED channel-major to (C*L, T); each candidate
W-hat (C_hat, C*L) from E1-multivar pools that flat field; pixel members are
taken over the (C*L) field. TAU_MAX and the true mode count come from mv_meta.

Corrupted footprint anchors are gated off in E1-multivar (block-diagonal setting),
so this selector runs over the pixel-side builders + oracle. If you later add
block-diagonal corrupted anchors to E1-multivar they flow through here unchanged.

Output: results/litext_e2_adag{E2_TAG}.npy  (run with E2_TAG=_multivar)
Env: E2_DATA_DIR (data/realisations_multivar2), E2_CANDS
     (results/litext_e1_discovery_multivar.npy), E2_NREAL (6), E2_ALPHA (0.01),
     E2_NPIX (6), E2_TAG (_multivar).
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("E2_DATA_DIR", ROOT / "data/realisations_multivar2"))
RES_DIR  = ROOT / "results"
E2_TAG   = os.environ.get("E2_TAG", "")

N_REAL  = int(os.environ.get("E2_NREAL", 6))
ALPHA   = float(os.environ.get("E2_ALPHA", 0.01))
N_PIX   = int(os.environ.get("E2_NPIX", 6))
SEED    = 0

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
rng = np.random.default_rng(SEED)

d0 = np.load(paths[0])
N_S, C, L_CH, NC, TAU_META = (int(x) for x in d0["mv_meta"])
LC = C * L_CH
# TAU_MAX from the ground-truth fine lags (matches E1/E4)
TAU_MAX = max(int(l) for _, _, l, _ in d0["fine_edges"])

# candidates from E1-multivar (full battery file if present, else footprints)
CANDS_SRC = os.environ.get("E2_CANDS", "")
src = None
if CANDS_SRC:
    src = np.load(CANDS_SRC, allow_pickle=True).item()
    src_name = Path(CANDS_SRC).name
else:
    for f in (f"litext_e1_discovery{E2_TAG}.npy", f"litext_e1_footprints{E2_TAG}.npy"):
        p = RES_DIR / f
        if p.exists():
            src = np.load(p, allow_pickle=True).item()
            src_name = p.name
            break
assert src is not None, "run sae/discover_modes_multivar.py first"
CANDS = src["cands"]
print(f"candidates: {sorted(CANDS)} (from {src_name}; NC={NC}, C={C}, tau_max={TAU_MAX})")

OBS = [np.load(paths[ri])["observations"].astype(np.float64).reshape(LC, -1)
       for ri in range(N_REAL)]                       # each (C*L, T)


def lag_stack(x, lags):
    mx = max(lags) if lags else 0
    T = len(x)
    return np.stack([x[mx - l: T - l] for l in lags], axis=1)


def residualize(y, conds):
    if conds.shape[1] == 0:
        return y - y.mean()
    X = np.column_stack([conds, np.ones(len(conds))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def dep_test(a, b, conds, alpha):
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
    Cn = What.shape[0]
    MX = TAU_MAX
    lags_all = list(range(0, MX + 1))
    lags_own = list(range(1, MX + 1))
    dep_agree, dep_tot = 0, 0
    ind_agree, ind_tot = 0, 0
    suff_ok, suff_tot = 0, 0
    for obs in OBS:
        Zh = What @ obs                              # (C_hat, T)
        T = Zh.shape[1]
        Tp = T - MX
        Lag = {k: lag_stack(Zh[k], lags_all) for k in range(Cn)}
        Own = {k: lag_stack(Zh[k], lags_own) for k in range(Cn)}
        cur = {k: Zh[k][MX:] for k in range(Cn)}
        members = {k: np.argsort(-What[k])[:N_PIX] for k in range(Cn)}
        pix = {k: obs[members[k]][:, MX:] for k in range(Cn)}
        for i in range(Cn):
            for j in range(i + 1, Cn):
                conds = np.column_stack(
                    [Lag[k] for k in range(Cn) if k not in (i, j)] +
                    [Own[i], Own[j]]) if Cn > 2 else \
                    np.column_stack([Own[i], Own[j]])
                agg_dep, _ = dep_test(cur[i], cur[j], conds, ALPHA)
                pairs = [(a, b) for a in range(min(3, N_PIX))
                         for b in range(min(3, N_PIX))]
                micro = [dep_test(pix[i][a], pix[j][b], conds, ALPHA)[0]
                         for a, b in pairs]
                micro_frac = float(np.mean(micro))
                if agg_dep:
                    dep_tot += 1
                    dep_agree += micro_frac
                else:
                    ind_tot += 1
                    ind_agree += 1.0 - micro_frac
        for i in range(Cn):
            for a in range(min(3, N_PIX)):
                resid = residualize(pix[i][a], Lag[i])
                dep_any = False
                for k in range(Cn):
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

out = dict(scores=SCORES, n_real=N_REAL, alpha=ALPHA,
           mv_meta=dict(N=N_S, C=C, L=L_CH, NC=NC),
           note="E2-multivar: Adag-style consistency scores over the "
                "block-diagonal (C*L) pixel field; S_suff = partial-aggregation "
                "sufficiency variant")
e1 = RES_DIR / f"litext_e1_discovery{E2_TAG}.npy"
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
np.save(RES_DIR / f"litext_e2_adag{E2_TAG}.npy", out, allow_pickle=True)
print(f"\nsaved -> results/litext_e2_adag{E2_TAG}.npy")
