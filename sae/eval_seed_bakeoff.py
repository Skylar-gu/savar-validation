"""
Blocks C + C2 — seed-level scoring and the TopK vs KAN-SAE bake-off.

Loads the mixed eqvar activation streams once, scores every seed checkpoint
(sae_mixed_{arch}_seed{N}_final.pt — FINAL weights per the literature rule:
never select the model by reconstruction MSE) with the Block-C metric suite
(sae/eval_sae_metrics.py: Hungarian-matched MCC, per-mode matched F1, feature
uniqueness), and writes:

  results/sae_metrics_suite.npy   {existing-artifact metrics for all 3 data
                                   dirs} + {topk seed mean/std on eqvar}
                                   + old best-|r| numbers side-by-side
  results/kan_sae_bakeoff.npy     paired per-seed TopK vs KAN table + verdict
                                   (win must exceed seed error bars to gate
                                   Block D/F propagation)

Usage: python3 sae/eval_seed_bakeoff.py
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path

from eval_sae_metrics import (INPUT_DIM, N_MODES, load_sae, encode,
                              metric_suite, print_suite)

DATADIR = Path("sae_data/hetdynamics_eqvar")
SEEDS   = (0, 1, 2)
ARCHS   = ("topk", "kan")

acts_full = np.load(DATADIR / "activations_full.npy")
Z_full    = np.load(DATADIR / "Z_full.npy")
streams = [acts_full[:, j].reshape(-1, INPUT_DIM) for j in range(N_MODES)]
Zs      = [Z_full[:, j].reshape(-1) for j in range(N_MODES)]

seed_results = {a: {} for a in ARCHS}
for arch in ARCHS:
    for s in SEEDS:
        ckpt = DATADIR / f"sae_mixed_{arch}_seed{s}_final.pt"
        if not ckpt.exists():
            print(f"MISSING {ckpt} — skipped"); continue
        sae, m, sd = load_sae(ckpt)
        codes = [encode(sae, streams[j], m, sd) for j in range(N_MODES)]
        r = metric_suite(codes, Zs)
        seed_results[arch][s] = r
        print_suite(f"eqvar mixed {arch}", f"seed{s}", r)

# ── paired bake-off table ─────────────────────────────────────────────────────
def col(arch, key):
    return np.array([seed_results[arch][s][key] for s in SEEDS
                     if s in seed_results[arch]])

print("\n" + "=" * 72)
print("KAN-SAE bake-off (eqvar mixed, final ckpts, paired seeds)")
print(f"  {'metric':<18} {'topk mean±std':>18} {'kan mean±std':>18} "
      f"{'paired Δ (kan−topk)':>22}")
verdict = {}
for key, name in [("mcc", "Hungarian MCC"), ("mean_f1", "matched F1"),
                  ("mean_uniqueness", "uniqueness")]:
    t, k = col("topk", key), col("kan", key)
    d = k - t
    win = bool(d.mean() > 0 and abs(d.mean()) > d.std(ddof=1) if len(d) > 1
               else d.mean() > 0)
    verdict[key] = dict(topk_mean=float(t.mean()), topk_std=float(t.std(ddof=1)),
                        kan_mean=float(k.mean()), kan_std=float(k.std(ddof=1)),
                        paired_delta=d.tolist(), delta_mean=float(d.mean()),
                        delta_std=float(d.std(ddof=1)), kan_wins_beyond_err=win)
    print(f"  {name:<18} {t.mean():>10.4f}±{t.std(ddof=1):.4f} "
          f"{k.mean():>11.4f}±{k.std(ddof=1):.4f} "
          f"{d.mean():>+12.4f}±{d.std(ddof=1):.4f}  {'WIN' if win else ''}")

gate = verdict["mcc"]["kan_wins_beyond_err"]
print(f"\nGate (propagate KAN codes to Blocks D/F): "
      f"{'YES — KAN wins MCC beyond seed error bars' if gate else 'NO — TopK stays primary'}")

os.makedirs("results", exist_ok=True)
np.save("results/kan_sae_bakeoff.npy",
        dict(seed_results=seed_results, verdict=verdict, gate_kan_propagates=gate,
             seeds=SEEDS, note="final ckpts, never selected on recon MSE"),
        allow_pickle=True)
print("saved -> results/kan_sae_bakeoff.npy")

# ── aggregate metric suite (Block C deliverable) ──────────────────────────────
suite = {}
for tag, f in [("finecadence", "scratchpad/blockC/metrics_finecadence.npy"),
               ("hetdynamics", "scratchpad/blockC/metrics_hetdynamics.npy"),
               ("eqvar", "scratchpad/blockC/metrics_eqvar.npy")]:
    if os.path.exists(f):
        suite[tag] = np.load(f, allow_pickle=True).item()[tag]

# old best-|r| numbers side-by-side
old = {}
for tag, d in [("finecadence", "sae_data/gnn"), ("hetdynamics", "sae_data/hetdynamics"),
               ("eqvar", "sae_data/hetdynamics_eqvar")]:
    old[tag] = {}
    for variant, f in [("permode", "alignment_per_mode"), ("mixed", "alignment_mixed")]:
        p = Path(d) / f"{f}.npy"
        if p.exists():
            a = np.load(p, allow_pickle=True).item()
            old[tag][variant] = dict(
                best_r=[float(abs(a[j]["max_r"])) for j in range(N_MODES)],
                mean_best_r=float(np.mean([abs(a[j]["max_r"]) for j in range(N_MODES)])),
                frac_ceil=[float(a[j].get("frac_ceil", np.nan)) for j in range(N_MODES)])

suite["eqvar_topk_seeds"] = dict(
    per_seed={s: seed_results["topk"][s] for s in seed_results["topk"]},
    mcc_mean=float(col("topk", "mcc").mean()),
    mcc_std=float(col("topk", "mcc").std(ddof=1)),
    f1_mean=float(col("topk", "mean_f1").mean()),
    f1_std=float(col("topk", "mean_f1").std(ddof=1)))
suite["old_best_r"] = old
np.save("results/sae_metrics_suite.npy", suite, allow_pickle=True)
print("saved -> results/sae_metrics_suite.npy")
