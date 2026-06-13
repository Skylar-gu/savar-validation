"""
Phase 6 — PCMCI causal discovery on mode time series Z.

Runs PCMCI+ParCorr on Z from all 100 realisations and evaluates
edge recovery against the ground truth graph G.

Index conventions
-----------------
Ground truth G:
  G[eff, cause, tau-1] = coefficient of X_cause(t-tau) in X_eff(t)

tigramite (this version):
  p_matrix[cause, eff, tau]   — p-value for  X_cause(t-tau) -> X_eff(t)
  val_matrix[cause, eff, tau] — partial corr for the same link

All edge tuples in this script are stored as (cause, eff, tau).
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

_ap = argparse.ArgumentParser()
_ap.add_argument("--dy005", action="store_true", help="Use D_y=0.05·I_L dataset")
_args = _ap.parse_args()

TAU_MAX  = 2
ALPHA    = 0.05   # MCI significance threshold
PC_ALPHA = 0.2    # PC skeleton step (standard for PCMCI)
if _args.dy005:
    DATA_DIR  = Path("data/realisations_dy005")
    SAVE_PATH = Path("results/pcmci_results_dy005.npy")
else:
    DATA_DIR  = Path("data/realisations")
    SAVE_PATH = Path("results/pcmci_results.npy")


# ── helpers ───────────────────────────────────────────────────────────────────

def gt_edges(G, cross_only=True):
    """Set of (cause, eff, tau) for nonzero entries in G; tau in [1, tau_max]."""
    N, _, tau_max = G.shape
    out = set()
    for eff in range(N):
        for cause in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tau_max + 1):
                if G[eff, cause, tau - 1] != 0:
                    out.add((cause, eff, tau))
    return out


def detected_edges(p_matrix, alpha, cross_only=True):
    """Set of (cause, eff, tau) where p_matrix[cause, eff, tau] < alpha."""
    N = p_matrix.shape[0]
    tau_max = p_matrix.shape[2] - 1
    out = set()
    for cause in range(N):
        for eff in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tau_max + 1):
                if p_matrix[cause, eff, tau] < alpha:
                    out.add((cause, eff, tau))
    return out


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


# ── load ground truth ─────────────────────────────────────────────────────────

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert len(paths) == 100, f"Expected 100 realisations, found {len(paths)}"

G  = np.load(paths[0])["ground_truth_graph"].astype(np.float64)  # (N, N, tau_max)
gt = gt_edges(G, cross_only=True)
N  = G.shape[0]

print("PCMCI Phase 6 — Mode Time Series Causal Discovery")
print("=" * 60)
print(f"  Realisations : {len(paths)}")
print(f"  Modes (N)    : {N}")
print(f"  tau_max      : {TAU_MAX}")
print(f"  alpha (MCI)  : {ALPHA}")
print(f"  pc_alpha     : {PC_ALPHA}")
print(f"  Ground truth cross-mode edges: {len(gt)}")
print()
for cause, eff, tau in sorted(gt):
    print(f"    X{cause}(t-{tau}) -> X{eff}(t)  coeff={G[eff, cause, tau-1]:+.2f}")
print()


# ── run PCMCI on each realisation ─────────────────────────────────────────────

records = []

for k, path in enumerate(paths):
    d = np.load(path)
    Z = d["latent_states"].astype(np.float64).T   # (T, N) — tigramite convention

    df    = DataFrame(Z)
    pcobj = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res   = pcobj.run_pcmci(
        tau_min=1, tau_max=TAU_MAX,
        pc_alpha=PC_ALPHA, alpha_level=ALPHA,
    )

    det        = detected_edges(res["p_matrix"], ALPHA, cross_only=True)
    tp, fp, fn = len(gt & det), len(det - gt), len(gt - det)
    prec, rec, f1 = prf(tp, fp, fn)

    records.append({
        "tp": tp, "fp": fp, "fn": fn,
        "prec": prec, "rec": rec, "f1": f1,
        "detected": det,
        "p_matrix":   res["p_matrix"],
        "val_matrix": res["val_matrix"],
    })

    if (k + 1) % 10 == 0:
        f1s = [r["f1"] for r in records]
        print(f"  [{k+1:3d}/100]  running mean F1 = {np.mean(f1s):.3f}")


# ── aggregate ─────────────────────────────────────────────────────────────────

precs = np.array([r["prec"] for r in records])
recs  = np.array([r["rec"]  for r in records])
f1s   = np.array([r["f1"]   for r in records])
tps   = np.array([r["tp"]   for r in records])
fps   = np.array([r["fp"]   for r in records])
fns   = np.array([r["fn"]   for r in records])

print(f"\n{'─'*60}")
print("  Aggregate edge-recovery metrics  (cross-mode edges only)")
print(f"{'─'*60}")
print(f"  Ground truth edges : {len(gt)}")
print(f"  Precision          : {precs.mean():.3f}  ±  {precs.std():.3f}")
print(f"  Recall             : {recs.mean():.3f}  ±  {recs.std():.3f}")
print(f"  F1                 : {f1s.mean():.3f}  ±  {f1s.std():.3f}")
print(f"  Mean TP / FP / FN  : {tps.mean():.1f} / {fps.mean():.1f} / {fns.mean():.1f}")


# ── per-edge recovery rate ────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print("  Per-edge recovery rate")
print(f"{'─'*60}")

edge_recovery = {}
for cause, eff, tau in sorted(gt):
    hits = sum(1 for r in records if (cause, eff, tau) in r["detected"])
    rate = hits / len(records)
    edge_recovery[(cause, eff, tau)] = rate
    coeff = G[eff, cause, tau - 1]
    flag  = "  <-- lag-2 only" if tau == 2 and G[eff, cause, 0] == 0 else ""
    print(f"  X{cause}(t-{tau}) -> X{eff}(t)  coeff={coeff:+.2f}  recovery={rate*100:5.1f}%{flag}")


# ── sign accuracy on true positives ──────────────────────────────────────────

sign_correct = sign_total = 0
for r in records:
    for cause, eff, tau in gt & r["detected"]:
        sign_total   += 1
        sign_correct += int(
            np.sign(G[eff, cause, tau - 1]) == np.sign(r["val_matrix"][cause, eff, tau])
        )

print(f"\n  Sign accuracy on TP edges: {sign_correct}/{sign_total} = {sign_correct/sign_total*100:.1f}%")


# ── false positive analysis ───────────────────────────────────────────────────

fp_counts: dict = {}
for r in records:
    for edge in r["detected"] - gt:
        fp_counts[edge] = fp_counts.get(edge, 0) + 1

if fp_counts:
    print(f"\n{'─'*60}")
    print("  Most common false positives (top 10)")
    print(f"{'─'*60}")
    for (cause, eff, tau), count in sorted(fp_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  X{cause}(t-{tau}) -> X{eff}(t)  in {count}/100 realisations")
else:
    print("\n  No false positives detected.")


# ── save ──────────────────────────────────────────────────────────────────────

np.save(SAVE_PATH, {
    "precision":     precs,
    "recall":        recs,
    "f1":            f1s,
    "tp":            tps,
    "fp":            fps,
    "fn":            fns,
    "edge_recovery": edge_recovery,
    "fp_counts":     fp_counts,
    "ground_truth":  gt,
})
print(f"\nResults saved → {SAVE_PATH}")
