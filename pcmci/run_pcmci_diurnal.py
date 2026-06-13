"""
Phase 6 (diurnal) — PCMCI on the diurnal/annual dataset, RAW vs DESEASONALIZED.

The diurnal Z carries a deterministic diurnal+annual cycle shared across all 100
realisations. Because the SAVAR dynamics are LINEAR, each realisation decomposes
exactly as  Z = Z_forced + Z_noise,  where Z_forced (the propagated cycle) is
identical across realisations and Z_noise has zero ensemble mean. Therefore the
ensemble mean over realisations IS the forced climatology, and subtracting it
recovers the pure cycle-free dynamics.

We run PCMCI two ways and compare edge recovery against the same ground-truth G:
  RAW           — PCMCI on Z (cycles present; expect cycle-driven confounding/FPs)
  DESEASONALIZED— PCMCI on Z − ensemble_mean(Z)  (cycles removed; expect ~baseline)

Outputs:
  pcmci_results_diurnal_raw.npy
  pcmci_results_diurnal_deseason.npy
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

TAU_MAX  = 2
ALPHA    = 0.05
PC_ALPHA = 0.2
DATA_DIR = Path("data/realisations_diurnal")


def gt_edges(G, cross_only=True):
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


# ── load all Z and build the ensemble-mean climatology ────────────────────────
paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert len(paths) == 100, f"Expected 100, found {len(paths)}"

G  = np.load(paths[0])["ground_truth_graph"].astype(np.float64)
gt = gt_edges(G, cross_only=True)
N  = G.shape[0]

Z_all = np.stack([np.load(p)["latent_states"].astype(np.float64) for p in paths])  # (100, N, T)
climatology = Z_all.mean(axis=0)                                                    # (N, T)

print("PCMCI (diurnal) — RAW vs DESEASONALIZED")
print("=" * 60)
print(f"  Realisations : {len(paths)}   Modes : {N}   T : {Z_all.shape[2]}")
print(f"  Ground-truth cross-mode edges : {len(gt)}")
# how much of the variance the cycle (ensemble mean) accounts for, per mode
clim_var = climatology.var(axis=1)
tot_var  = Z_all.mean(0).var(axis=1) * 0 + Z_all.reshape(-1, N, Z_all.shape[2]).var(axis=(0, 2))
print(f"  Climatology variance share per mode: "
      f"{np.round(clim_var / Z_all.var(axis=(0,2)) * 100, 1)} %")
print()


def run_mode(name, deseason):
    records = []
    for k, p in enumerate(paths):
        Z = Z_all[k]
        if deseason:
            Z = Z - climatology
        Zt = Z.T  # (T, N)
        pcobj = PCMCI(dataframe=DataFrame(Zt), cond_ind_test=ParCorr(), verbosity=0)
        res = pcobj.run_pcmci(tau_min=1, tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA)
        det = detected_edges(res["p_matrix"], ALPHA, cross_only=True)
        tp, fp, fn = len(gt & det), len(det - gt), len(gt - det)
        prec, rec, f1 = prf(tp, fp, fn)
        records.append({"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec,
                        "f1": f1, "detected": det, "val_matrix": res["val_matrix"]})
        if (k + 1) % 25 == 0:
            print(f"  [{name}] [{k+1:3d}/100]  running mean F1 = "
                  f"{np.mean([r['f1'] for r in records]):.3f}")
    return records


def report(name, records, save_path):
    precs = np.array([r["prec"] for r in records])
    recs  = np.array([r["rec"]  for r in records])
    f1s   = np.array([r["f1"]   for r in records])
    fps   = np.array([r["fp"]   for r in records])
    print(f"\n  ── {name} ──")
    print(f"    Precision : {precs.mean():.3f} ± {precs.std():.3f}")
    print(f"    Recall    : {recs.mean():.3f} ± {recs.std():.3f}")
    print(f"    F1        : {f1s.mean():.3f} ± {f1s.std():.3f}")
    print(f"    Mean FP   : {fps.mean():.1f}")
    # top false positives
    fp_counts = {}
    for r in records:
        for e in r["detected"] - gt:
            fp_counts[e] = fp_counts.get(e, 0) + 1
    top = sorted(fp_counts.items(), key=lambda x: -x[1])[:5]
    if top:
        print(f"    Top FPs   : " + ", ".join(
            f"X{c}->X{e}(t-{t}):{n}" for (c, e, t), n in top))
    np.save(save_path, {"precision": precs, "recall": recs, "f1": f1s,
                        "fp_counts": fp_counts, "ground_truth": gt})
    return f1s.mean(), fps.mean()


print("Running RAW ...")
raw = run_mode("raw", deseason=False)
f1_raw, fp_raw = report("RAW (cycles present)", raw, "results/pcmci_results_diurnal_raw.npy")

print("\nRunning DESEASONALIZED ...")
des = run_mode("deseason", deseason=True)
f1_des, fp_des = report("DESEASONALIZED (Z − ensemble mean)", des, "results/pcmci_results_diurnal_deseason.npy")

print(f"\n{'═'*60}")
print(f"  SUMMARY   raw F1={f1_raw:.3f} (FP {fp_raw:.1f})  →  "
      f"deseason F1={f1_des:.3f} (FP {fp_des:.1f})")
print(f"  baseline reference (no cycles): F1≈0.825")
print(f"{'═'*60}")
