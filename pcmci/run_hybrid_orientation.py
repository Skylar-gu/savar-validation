"""
Block E (hybrid attempt) — PCMCI+ skeleton + DirectLiNGAM orientation.

VAR-LiNGAM's τ=0 orientation on the aliased edges came out at chance
(~0.53, results/orientation_benchmark.npy). Plan-sanctioned single follow-up:
use PCMCI+ for the skeleton and DirectLiNGAM only for what it is good at —
orienting a *given* contemporaneous adjacency from non-Gaussianity.

Per stride s ∈ {2,3,4}, per realisation:
 1. PCMCI+ (ParCorr, pc_alpha 0.05, same settings as run_pcmci_subsample.py)
    → graph; collect τ=0 adjacencies (o-o / --> / <--) and each variable's
    lagged parents (τ≥1 '-->' edges into it).
 2. Regress each Z_i on its lagged parents (OLS) → residual series e_i.
 3. DirectLiNGAM on the 8-variable residual matrix → causal order + B0.
 4. For every GT aliased pair recovered as a τ=0 adjacency by PCMCI+:
    orient by |B0| contest on the residuals (fallback: causal order).
    Score orientation accuracy against the true fine-edge direction.

Output: results/orientation_hybrid.npy
Env: OH_STRIDES="2,3,4"  OH_NREAL=40  OH_PCALPHA=0.05
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
import lingam
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

DATA_DIR = Path("data/realisations_finecadence")
STRIDES  = [int(s) for s in os.environ.get("OH_STRIDES", "2,3,4").split(",")]
N_REAL   = int(os.environ.get("OH_NREAL", 40))
PC_ALPHA = float(os.environ.get("OH_PCALPHA", 0.05))


def coarse_gt(fine_edges, s):
    contemp_dir = set()
    tau_max = 1
    for c, e, l, _ in fine_edges:
        c, e, l = int(c), int(e), int(l)
        tau = l // s
        if tau == 0:
            contemp_dir.add((c, e))
        else:
            tau_max = max(tau_max, tau)
    return contemp_dir, tau_max


paths = sorted(DATA_DIR.glob("realisation_*.npz"))[:N_REAL]
assert paths
print(f"Hybrid PCMCI+ skeleton + DirectLiNGAM orientation — {len(paths)} "
      f"realisations, strides {STRIDES}, pc_alpha {PC_ALPHA}")

results = {"strides": STRIDES, "n_real": len(paths), "pc_alpha": PC_ALPHA,
           "rows": []}

for s in STRIDES:
    con_gt_dir, tmax = coarse_gt(np.load(paths[0])["fine_edges"], s)
    n_ok_b0 = n_ok_order = n_scored = n_gt_recovered = 0
    n_gt_total = len(con_gt_dir) * len(paths)

    for p in paths:
        Z = np.load(p)["latent_states"].astype(np.float64)[:, ::s].T   # (T, N)
        T, N = Z.shape
        pc = PCMCI(dataframe=DataFrame(Z.copy()), cond_ind_test=ParCorr(),
                   verbosity=0)
        res = pc.run_pcmciplus(tau_min=0, tau_max=tmax, pc_alpha=PC_ALPHA)
        g = res["graph"]

        # tau=0 adjacencies and lagged parents
        adj0 = {frozenset((i, j)) for i in range(N) for j in range(N)
                if i != j and g[i, j, 0] not in ("", "x-x")}
        parents = {i: [(j, tau) for j in range(N)
                       for tau in range(1, tmax + 1) if g[j, i, tau] == "-->"]
                   for i in range(N)}

        # residualize on lagged parents
        E = np.empty((T - tmax, N))
        for i in range(N):
            y = Z[tmax:, i]
            if parents[i]:
                X = np.column_stack([Z[tmax - tau: T - tau, j]
                                     for j, tau in parents[i]])
                X = np.column_stack([X, np.ones(len(X))])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                E[:, i] = y - X @ beta
            else:
                E[:, i] = y - y.mean()

        dl = lingam.DirectLiNGAM()
        dl.fit(E)
        B0 = dl.adjacency_matrix_                    # B0[i, j]: x_j -> x_i
        order = list(dl.causal_order_)

        for (c, e) in con_gt_dir:
            if frozenset((c, e)) not in adj0:
                continue
            n_gt_recovered += 1
            fwd, rev = abs(B0[e, c]), abs(B0[c, e])
            if fwd > 0 or rev > 0:
                n_scored += 1
                n_ok_b0 += int(fwd > rev)
            n_ok_order += int(order.index(c) < order.index(e))

    acc_b0 = n_ok_b0 / n_scored if n_scored else float("nan")
    acc_order = n_ok_order / n_gt_recovered if n_gt_recovered else float("nan")
    row = dict(stride=s, n_con_gt=len(con_gt_dir), n_gt_total=n_gt_total,
               n_gt_recovered=n_gt_recovered, n_scored_b0=n_scored,
               orient_acc_B0=acc_b0, orient_acc_order=acc_order)
    results["rows"].append(row)
    print(f"stride {s}: GT aliased pairs recovered by PCMCI+ "
          f"{n_gt_recovered}/{n_gt_total}; DirectLiNGAM-on-residuals "
          f"orientation acc: B0 contest={acc_b0:.3f} ({n_ok_b0}/{n_scored}), "
          f"causal order={acc_order:.3f} ({n_ok_order}/{n_gt_recovered})")

os.makedirs("results", exist_ok=True)
np.save("results/orientation_hybrid.npy", results, allow_pickle=True)
print("saved -> results/orientation_hybrid.npy")
