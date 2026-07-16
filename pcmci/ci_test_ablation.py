"""CI-test ablation diagnostic (robustness rungs, notes/demo_notebook_plan.md §4).

Runs PCMCI+ on the TRUE mode series Z with different conditional-independence
tests and compares (a) each edge set against the true graph (F1) and (b) the
edge sets pairwise (Jaccard).  Reproduces the earlier ad-hoc comparison that
produced out/ci_compare.log (ParCorr vs CMIknn on (2000, 8) mode data).

Optionally (CIA_FPCAL=1) adds a false-positive calibration check on NULL pairs:
mode i from realisation r vs mode j from a DIFFERENT realisation r' are
independent by construction (same marginals / autocorrelation).  Each CI test
is run on lagged (X_{t-tau}, Y_t) with both series' own past in the
conditioning set, so the empirical rejection rate should sit near alpha if the
test's p-values are calibrated under this world's innovation distribution.

Env:
  CIA_DATA_DIR   realisations dir (default data/realisations_nl_gauss)
  CIA_TESTS      comma list of {parcorr,robustparcorr,gpdc,cmiknn}
                 (default "parcorr,robustparcorr")
  CIA_NSAMP      samples used per realisation (default 2000, as ci_compare)
  CIA_NREAL      realisations for the PCMCI+ comparison (default 1)
  CIA_FPCAL      "1" -> run the null-pair calibration block (default 0)
  CIA_NNULL      number of null-pair tests per CI test (default 200)
  CIA_TAG        results suffix -> results/litext_ci_ablation<TAG>.npy
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, time, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CIA_DATA_DIR", ROOT / "data/realisations_nl_gauss"))
TESTS    = [t.strip().lower() for t in
            os.environ.get("CIA_TESTS", "parcorr,robustparcorr").split(",") if t.strip()]
N_SAMP   = int(os.environ.get("CIA_NSAMP", 2000))
N_REAL   = int(os.environ.get("CIA_NREAL", 1))
FPCAL    = os.environ.get("CIA_FPCAL", "0") == "1"
N_NULL   = int(os.environ.get("CIA_NNULL", 200))
TAG      = os.environ.get("CIA_TAG", "")
PC_ALPHA = 0.05                      # same convention as E1/E4
ALPHAS   = (0.01, 0.05, 0.10)        # calibration grid
SEED     = 0
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert paths, f"no realisations in {DATA_DIR}"
d0 = np.load(paths[0])
gt = {(int(c), int(e), int(l)) for c, e, l, _ in d0["fine_edges"] if int(c) != int(e)}
TAU_MAX = max(l for _, _, l in gt)
N_MODES = d0["latent_states"].shape[0]

def make_test(name):
    if name == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        return ParCorr()
    if name == "robustparcorr":
        from tigramite.independence_tests.robust_parcorr import RobustParCorr
        return RobustParCorr()
    if name == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC
        return GPDC()
    if name == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        return CMIknn()
    raise ValueError(f"unknown CI test {name!r}")

def detect(graph):
    N, _, T1 = graph.shape
    return {(c, e, tau) for c in range(N) for e in range(N) if c != e
            for tau in range(1, T1) if graph[c, e, tau] == "-->"}

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

print(f"CI-test ablation on {DATA_DIR.name}: tests={TESTS} "
      f"n_samp={N_SAMP} n_real={N_REAL} tau_max={TAU_MAX} pc_alpha={PC_ALPHA}")
print(f"truth: {len(gt)} cross edges")

series = [np.load(paths[ri])["latent_states"].astype(np.float64).T[:N_SAMP]
          for ri in range(N_REAL)]                     # each (N_SAMP, N_MODES)
print(f"data: {series[0].shape} (samples, modes)\n")

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI

results = {"config": dict(data_dir=str(DATA_DIR), tests=TESTS, n_samp=N_SAMP,
                          n_real=N_REAL, tau_max=TAU_MAX, pc_alpha=PC_ALPHA,
                          gt=sorted(gt)),
           "per_test": {}, "jaccard": {}, "fpcal": {}}

edge_sets = {}
for name in TESTS:
    agg = dict(tp=0, fp=0, fn=0)
    per_real_edges = []
    t0 = time.time()
    for ri in range(N_REAL):
        ci = make_test(name)
        pc = PCMCI(dataframe=DataFrame(series[ri].copy()),
                   cond_ind_test=ci, verbosity=0)
        res = pc.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
        det = detect(res["graph"])
        per_real_edges.append(sorted(det))
        agg["tp"] += len(gt & det); agg["fp"] += len(det - gt)
        agg["fn"] += len(gt - det)
    dt = time.time() - t0
    p, r, f1 = prf(agg["tp"], agg["fp"], agg["fn"])
    edge_sets[name] = set().union(*[set(e) for e in per_real_edges])
    n_edges = sum(len(e) for e in per_real_edges) / N_REAL
    print(f"{name:<14} edges={n_edges:5.1f}  vs-truth F1={f1:.3f} "
          f"(tp{agg['tp']} fp{agg['fp']} fn{agg['fn']})  [{dt:.0f}s]")
    results["per_test"][name] = dict(P=p, R=r, F1=f1, **agg, seconds=dt,
                                     edges_per_real=per_real_edges)

print()
for a, b in itertools.combinations(TESTS, 2):
    A, B = edge_sets[a], edge_sets[b]
    jac = len(A & B) / len(A | B) if A | B else 1.0
    print(f"{a} vs {b}: Jaccard={jac:.3f}  agree={len(A & B)}  "
          f"only-{a}={len(A - B)}  only-{b}={len(B - A)}")
    print(f"  {a}\\{b}: {sorted(A - B)}")
    print(f"  {b}\\{a}: {sorted(B - A)}")
    results["jaccard"][(a, b)] = jac

if FPCAL:
    print(f"\nFP calibration on {N_NULL} null pairs (cross-realisation, "
          f"conditioned on both series' own past):")
    assert len(paths) >= 2, "need >= 2 realisations for null pairs"
    rng = np.random.default_rng(SEED)
    # pre-draw the null-pair specs once so every test sees the same pairs
    specs = []
    for _ in range(N_NULL):
        r1 = int(rng.integers(len(paths)))
        r2 = int(rng.integers(len(paths)))
        while r2 == r1:
            r2 = int(rng.integers(len(paths)))
        i = int(rng.integers(N_MODES)); j = int(rng.integers(N_MODES))
        tau = int(rng.integers(1, TAU_MAX + 1))
        specs.append((r1, r2, i, j, tau))
    cache = {}
    def zser(r, i):
        if (r, i) not in cache:
            cache[(r, i)] = np.load(paths[r])["latent_states"][i].astype(np.float64)[:N_SAMP + 8]
        return cache[(r, i)]
    for name in TESTS:
        if name in ("gpdc", "cmiknn") and N_NULL > 60:
            n_use = 60      # slow tests: cap the calibration sample
        else:
            n_use = N_NULL
        ci = make_test(name)
        pvals = []
        t0 = time.time()
        for (r1, r2, i, j, tau) in specs[:n_use]:
            X = zser(r1, i); Y = zser(r2, j)
            t_lo = tau + 2
            T_eff = min(len(X), len(Y)) - t_lo
            idx = np.arange(t_lo, t_lo + T_eff)
            x = X[idx - tau][:, None]                          # X_{t-tau}
            y = Y[idx][:, None]                                # Y_t
            z = np.column_stack([Y[idx - 1], Y[idx - 2],       # own past
                                 X[idx - tau - 1], X[idx - tau - 2]])
            val, pval = ci.run_test_raw(x, y, z)[:2]
            pvals.append(float(pval))
        pvals = np.array(pvals)
        rates = {a: float((pvals < a).mean()) for a in ALPHAS}
        print(f"  {name:<14} n={n_use:4d}  FP rate: " +
              "  ".join(f"alpha={a:.2f}->{rates[a]:.3f}" for a in ALPHAS) +
              f"  [{time.time()-t0:.0f}s]")
        results["fpcal"][name] = dict(alphas=list(ALPHAS), rates=rates,
                                      pvals=pvals, n=n_use)

out = RES_DIR / f"litext_ci_ablation{TAG}.npy"
np.save(out, results, allow_pickle=True)
print(f"\nsaved -> {out}")
