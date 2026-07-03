"""
PCMCI vs PCMCI+ under temporal SUBSAMPLING of a fine-cadence simulation.

Setup
-----
data/realisations_finecadence/ holds SAVAR runs simulated at a FINE cadence with
heterogeneous per-edge fine lags ℓ (see generate_finecadence.py). We subsample
the mode series Z at stride s (keep every s-th step) and ask how causal recovery
changes — in particular how fast couplings (ℓ < s) that ALIAS into contemporaneous
(τ=0) edges are recovered.

For each stride s, a fine edge (cause→eff, fine lag ℓ) maps to coarse lag
  τ = ℓ // s
so ℓ < s  ⇒  τ = 0 (contemporaneous);  ℓ ≥ s ⇒ lagged.

Two discoverers on the SAME subsampled data:
  • PCMCI  (run_pcmci,  tau_min=1) — lagged only; CANNOT represent τ=0 by
    construction, so every aliased edge is structurally invisible to it.
  • PCMCI+ (run_pcmciplus, tau_min=0) — contemporaneous + lagged.

Metrics, cross-mode edges only, averaged over realisations:
  • LAGGED  F1   — directed edges at τ ≥ 1.
  • CONTEMP F1   — UNDIRECTED adjacency at τ = 0 (PCMCI+ returns 'o-o': adjacency
    is recovered but the Gaussian ParCorr test does not orient it, even though
    the innovations are non-Gaussian — the orientation column reports how many
    of the recovered τ=0 edges PCMCI+ nonetheless oriented correctly).
  • OVERALL F1   — lagged (directed) + contemporaneous (undirected) combined.

Output: results/pcmci_subsample_sweep.npy
Env: SUB_STRIDES="1,2,3,4"  SUB_NREAL=40  SUB_PCALPHA=0.05  SUB_RUN_PLAIN=1
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

DATA_DIR  = Path("data/realisations_finecadence")
STRIDES   = [int(s) for s in os.environ.get("SUB_STRIDES", "1,2,3,4").split(",")]
N_REAL    = int(os.environ.get("SUB_NREAL", 40))
PC_ALPHA  = float(os.environ.get("SUB_PCALPHA", 0.05))
RUN_PLAIN = os.environ.get("SUB_RUN_PLAIN", "1") == "1"


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def coarse_ground_truth(fine_edges, s):
    """Map fine edges to coarse (lagged directed + contemporaneous undirected) GT."""
    lagged = set()        # (cause, eff, tau), tau >= 1, directed
    contemp = set()       # frozenset({cause, eff}), undirected
    contemp_dir = set()   # (cause, eff) true direction, for orientation scoring
    tau_max = 1
    for cause, eff, ell, _ in fine_edges:
        cause, eff, ell = int(cause), int(eff), int(ell)
        tau = ell // s
        if tau == 0:
            contemp.add(frozenset((cause, eff)))
            contemp_dir.add((cause, eff))
        else:
            lagged.add((cause, eff, tau))
            tau_max = max(tau_max, tau)
    return lagged, contemp, contemp_dir, tau_max


def detect_from_graph(graph):
    """Read directed-lagged and undirected-contemporaneous edges from a PCMCI(+)
    graph array (shape N,N,tau_max+1 of strings)."""
    N = graph.shape[0]
    T = graph.shape[2] - 1
    lagged = set()
    contemp = set()
    contemp_dir = set()
    for c in range(N):
        for e in range(N):
            if c == e:
                continue
            for tau in range(1, T + 1):
                if graph[c, e, tau] == '-->':
                    lagged.add((c, e, tau))
            g0 = graph[c, e, 0]
            if g0 not in ('', 'x-x'):           # any recovered adjacency at tau=0
                contemp.add(frozenset((c, e)))
            if g0 == '-->':                      # oriented this way
                contemp_dir.add((c, e))
    return lagged, contemp, contemp_dir


def eval_one(Zs, fine_edges, s, plus):
    """Run one discoverer on one subsampled realisation; return per-edge-type counts."""
    lag_gt, con_gt, con_gt_dir, tmax = coarse_ground_truth(fine_edges, s)
    df = DataFrame(Zs)
    pc = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    if plus:
        res = pc.run_pcmciplus(tau_min=0, tau_max=tmax, pc_alpha=PC_ALPHA)
        lag_d, con_d, con_d_dir = detect_from_graph(res["graph"])
    else:
        # plain PCMCI: lagged only (tau_min=1). Contemporaneous structurally absent.
        res = pc.run_pcmci(tau_min=1, tau_max=tmax, pc_alpha=PC_ALPHA, alpha_level=PC_ALPHA)
        N = res["p_matrix"].shape[0]
        lag_d = {(c, e, tau) for c in range(N) for e in range(N) if c != e
                 for tau in range(1, tmax + 1) if res["p_matrix"][c, e, tau] < PC_ALPHA}
        con_d, con_d_dir = set(), set()

    # lagged (directed)
    l_tp = len(lag_gt & lag_d); l_fp = len(lag_d - lag_gt); l_fn = len(lag_gt - lag_d)
    # contemporaneous (undirected adjacency)
    c_tp = len(con_gt & con_d); c_fp = len(con_d - con_gt); c_fn = len(con_gt - con_d)
    # orientation: of the GT contemp pairs that were recovered as adjacency, how many
    # did PCMCI+ orient in the correct direction?
    recovered_pairs = con_gt & con_d
    n_orient_ok = sum(1 for (a, b) in con_gt_dir
                      if frozenset((a, b)) in recovered_pairs and (a, b) in con_d_dir)
    return dict(l_tp=l_tp, l_fp=l_fp, l_fn=l_fn,
                c_tp=c_tp, c_fp=c_fp, c_fn=c_fn,
                n_con_gt=len(con_gt), n_recovered=len(recovered_pairs),
                n_orient_ok=n_orient_ok, lag_d=lag_d, con_d=con_d,
                lag_gt=lag_gt, con_gt=con_gt)


def sweep(paths, plus):
    label = "PCMCI+" if plus else "PCMCI "
    rows = []
    for s in STRIDES:
        recs = []
        for p in paths:
            Z = np.load(p)["latent_states"].astype(np.float64)   # (N, T_fine)
            fe = np.load(p)["fine_edges"]
            Zs = Z[:, ::s].T                                      # (T_coarse, N)
            recs.append(eval_one(Zs, fe, s, plus))
        agg = {k: np.array([r[k] for r in recs]) for k in
               ("l_tp", "l_fp", "l_fn", "c_tp", "c_fp", "c_fn",
                "n_con_gt", "n_recovered", "n_orient_ok")}
        l_p, l_r, l_f1 = prf(agg["l_tp"].sum(), agg["l_fp"].sum(), agg["l_fn"].sum())
        c_p, c_r, c_f1 = prf(agg["c_tp"].sum(), agg["c_fp"].sum(), agg["c_fn"].sum())
        o_p, o_r, o_f1 = prf(agg["l_tp"].sum() + agg["c_tp"].sum(),
                             agg["l_fp"].sum() + agg["c_fp"].sum(),
                             agg["l_fn"].sum() + agg["c_fn"].sum())
        lag_gt, con_gt, _, tmax = coarse_ground_truth(np.load(paths[0])["fine_edges"], s)
        orient_rate = (agg["n_orient_ok"].sum() / agg["n_recovered"].sum()
                       if agg["n_recovered"].sum() else 0.0)
        row = dict(stride=s, T_coarse=Z[:, ::s].shape[1], tau_max=tmax,
                   n_lag_gt=len(lag_gt), n_con_gt=len(con_gt),
                   lag_f1=l_f1, lag_p=l_p, lag_r=l_r,
                   con_f1=c_f1, con_p=c_p, con_r=c_r,
                   overall_f1=o_f1, orient_rate=orient_rate)
        rows.append(row)
        print(f"  [{label}] s={s}  T={row['T_coarse']:4d}  τmax={tmax}  "
              f"GT(lag={len(lag_gt):2d},con={len(con_gt):2d})  "
              f"LAG F1={l_f1:.3f}(P{l_p:.2f}/R{l_r:.2f})  "
              f"CON F1={c_f1:.3f}(P{c_p:.2f}/R{c_r:.2f})  "
              f"OVERALL F1={o_f1:.3f}  con-orient={orient_rate:.2f}")
    return rows


paths = sorted(DATA_DIR.glob("realisation_*.npz"))[:N_REAL]
assert paths, f"no realisations in {DATA_DIR}"

print("PCMCI vs PCMCI+ under subsampling of a fine-cadence simulation")
print("=" * 78)
print(f"  realisations : {len(paths)}   strides : {STRIDES}   pc_alpha : {PC_ALPHA}")
print(f"  fine lags    : {sorted(set(int(l) for _,_,l,_ in np.load(paths[0])['fine_edges']))}")
print(f"  rule         : coarse τ = fine_lag // stride   (τ=0 ⇒ contemporaneous)")
print()

results = {"strides": STRIDES, "n_real": len(paths), "pc_alpha": PC_ALPHA}

print("── PCMCI+ (run_pcmciplus, tau_min=0: contemporaneous + lagged) ──")
results["pcmciplus"] = sweep(paths, plus=True)

if RUN_PLAIN:
    print("\n── PCMCI (run_pcmci, tau_min=1: lagged only) ──")
    results["pcmci"] = sweep(paths, plus=False)

os.makedirs("results", exist_ok=True)
np.save("results/pcmci_subsample_sweep.npy", results)

print("\n" + "═" * 78)
print("SUMMARY — contemporaneous-edge recovery as more fast couplings alias to τ=0")
print(f"{'stride':>6} {'n_con_GT':>9} {'PCMCI+ CON-F1':>14} {'PCMCI+ OVERALL':>15} "
      f"{'PCMCI CON-F1':>13} {'PCMCI OVERALL':>14}")
for i, s in enumerate(STRIDES):
    pp = results["pcmciplus"][i]
    pl = results["pcmci"][i] if RUN_PLAIN else {"con_f1": float('nan'), "overall_f1": float('nan')}
    print(f"{s:>6} {pp['n_con_gt']:>9} {pp['con_f1']:>14.3f} {pp['overall_f1']:>15.3f} "
          f"{pl['con_f1']:>13.3f} {pl['overall_f1']:>14.3f}")
print("═" * 78)
print("Note: PCMCI (tau_min=1) has contemporaneous recall ≡ 0 by construction — it")
print("cannot represent τ=0 edges, so its OVERALL F1 degrades as aliasing increases.")
