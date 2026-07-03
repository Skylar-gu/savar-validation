"""
Block E — τ=0 orientation benchmark with VAR-LiNGAM (Gong et al. ICML 2015 line).

Subsampled non-Gaussian linear systems are identifiable in principle; our
fine-cadence generator uses skew-normal innovations, so the aliased τ=0 edges
(fine lag ℓ < stride s) that PCMCI+ leaves 'o-o' should be orientable.
Benchmark an existing estimator (lingam.VARLiNGAM) — no invented heuristics.

Per stride s ∈ {2,3,4} (same subsampling as pcmci/run_pcmci_subsample.py):
  * fit VARLiNGAM(lags=tau_max) on Z[:, ::s].T per realisation;
  * adjacency_matrices_[τ][i, j] = coefficient x_j(t−τ) → x_i(t);
  * (a) orientation accuracy on exactly the aliased GT edge set: among GT τ=0
        pairs, fraction where |B0[eff,cause]| > |B0[cause,eff]| (pure
        orientation, threshold-free), plus detected-and-oriented rate at the
        report threshold;
  * (b) full-graph directed F1 over a small |coef| threshold grid, split into
        lagged-F1 / contemp-F1 / overall for comparison with
        results/pcmci_subsample_sweep.npy (CON-F1 there is UNDIRECTED —
        we also report our undirected contemp adjacency F1 for apples-to-apples,
        and the stricter directed version).

Output: results/orientation_benchmark.npy
Env: OB_STRIDES="2,3,4"  OB_NREAL=40  OB_THRESH="0.01,0.05,0.10"  OB_REPORT=0.05
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
import lingam

DATA_DIR = Path("data/realisations_finecadence")
STRIDES  = [int(s) for s in os.environ.get("OB_STRIDES", "2,3,4").split(",")]
N_REAL   = int(os.environ.get("OB_NREAL", 40))
THRESH   = [float(t) for t in os.environ.get("OB_THRESH", "0.01,0.05,0.10").split(",")]
REPORT_T = float(os.environ.get("OB_REPORT", 0.05))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def coarse_gt(fine_edges, s):
    lagged, contemp_dir = set(), set()
    tau_max = 1
    for c, e, l, _ in fine_edges:
        c, e, l = int(c), int(e), int(l)
        tau = l // s
        if tau == 0:
            contemp_dir.add((c, e))
        else:
            lagged.add((c, e, tau))
            tau_max = max(tau_max, tau)
    return lagged, contemp_dir, tau_max


paths = sorted(DATA_DIR.glob("realisation_*.npz"))[:N_REAL]
assert paths, f"no realisations in {DATA_DIR}"
print(f"VAR-LiNGAM orientation benchmark — {len(paths)} realisations, "
      f"strides {STRIDES}, thresholds {THRESH} (report at {REPORT_T})")

results = {"strides": STRIDES, "n_real": len(paths), "thresh": THRESH,
           "report_thresh": REPORT_T, "rows": []}

for s in STRIDES:
    lag_gt, con_gt_dir, tmax = coarse_gt(np.load(paths[0])["fine_edges"], s)
    con_gt_pairs = {frozenset(p) for p in con_gt_dir}
    n_orient_correct = 0     # threshold-free |B0| direction contest, GT pairs
    n_orient_total = 0
    per_t = {t: dict(l_tp=0, l_fp=0, l_fn=0, cd_tp=0, cd_fp=0, cd_fn=0,
                     cu_tp=0, cu_fp=0, cu_fn=0, det_orient_ok=0, det_pairs=0)
             for t in THRESH}

    for p in paths:
        d = np.load(p)
        Z = d["latent_states"].astype(np.float64)[:, ::s].T      # (T, N)
        N = Z.shape[1]
        model = lingam.VARLiNGAM(lags=tmax, criterion=None)
        model.fit(Z)
        B = model.adjacency_matrices_                            # (tmax+1, N, N)

        # (a) threshold-free orientation contest on GT aliased pairs
        for (c, e) in con_gt_dir:
            fwd, rev = abs(B[0][e, c]), abs(B[0][c, e])
            if fwd > 0 or rev > 0:
                n_orient_total += 1
                n_orient_correct += int(fwd > rev)

        # (b) thresholded graphs
        for t in THRESH:
            lag_d = {(c, e, tau) for tau in range(1, tmax + 1)
                     for e in range(N) for c in range(N)
                     if c != e and abs(B[tau][e, c]) > t}
            con_d_dir = {(c, e) for e in range(N) for c in range(N)
                         if c != e and abs(B[0][e, c]) > t}
            con_d_pairs = {frozenset(p_) for p_ in con_d_dir}
            a = per_t[t]
            a["l_tp"] += len(lag_gt & lag_d); a["l_fp"] += len(lag_d - lag_gt)
            a["l_fn"] += len(lag_gt - lag_d)
            a["cd_tp"] += len(con_gt_dir & con_d_dir)
            a["cd_fp"] += len(con_d_dir - con_gt_dir)
            a["cd_fn"] += len(con_gt_dir - con_d_dir)
            a["cu_tp"] += len(con_gt_pairs & con_d_pairs)
            a["cu_fp"] += len(con_d_pairs - con_gt_pairs)
            a["cu_fn"] += len(con_gt_pairs - con_d_pairs)
            # of GT pairs detected as adjacency, correctly oriented?
            for (c, e) in con_gt_dir:
                if frozenset((c, e)) in con_d_pairs:
                    a["det_pairs"] += 1
                    a["det_orient_ok"] += int((c, e) in con_d_dir and
                                              abs(B[0][e, c]) > abs(B[0][c, e]))

    orient_acc = n_orient_correct / n_orient_total if n_orient_total else float("nan")
    row = dict(stride=s, tau_max=tmax, n_lag_gt=len(lag_gt),
               n_con_gt=len(con_gt_dir),
               orient_acc_threshfree=orient_acc, n_orient_total=n_orient_total,
               per_thresh={})
    print(f"\nstride {s}: GT lagged={len(lag_gt)} contemp={len(con_gt_dir)}  "
          f"threshold-free orientation acc = {orient_acc:.3f} "
          f"({n_orient_correct}/{n_orient_total})")
    print(f"  {'thr':>5} {'LAG-F1':>7} {'CON-F1dir':>10} {'CON-F1und':>10} "
          f"{'OVERALL':>8} {'orient|det':>10}")
    for t in THRESH:
        a = per_t[t]
        _, _, lf1 = prf(a["l_tp"], a["l_fp"], a["l_fn"])
        _, _, cdf1 = prf(a["cd_tp"], a["cd_fp"], a["cd_fn"])
        _, _, cuf1 = prf(a["cu_tp"], a["cu_fp"], a["cu_fn"])
        _, _, of1 = prf(a["l_tp"] + a["cu_tp"], a["l_fp"] + a["cu_fp"],
                        a["l_fn"] + a["cu_fn"])
        det_or = a["det_orient_ok"] / a["det_pairs"] if a["det_pairs"] else float("nan")
        row["per_thresh"][t] = dict(lag_f1=lf1, con_f1_dir=cdf1, con_f1_und=cuf1,
                                    overall_f1=of1, det_orient_rate=det_or,
                                    counts=dict(a))
        print(f"  {t:>5.2f} {lf1:>7.3f} {cdf1:>10.3f} {cuf1:>10.3f} "
              f"{of1:>8.3f} {det_or:>10.3f}")
    results["rows"].append(row)

os.makedirs("results", exist_ok=True)
np.save("results/orientation_benchmark.npy", results, allow_pickle=True)
print("\nsaved -> results/orientation_benchmark.npy")
print("(compare: results/pcmci_subsample_sweep.npy CON-F1 0.63/0.49/0.38 "
      "undirected, orient rate 0.08-0.30)")
