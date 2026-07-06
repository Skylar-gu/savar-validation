"""
Follow-up 3 — Score the shared-subspace-break SAEs with the Block-C suite.

For each variant datadir (proj = k=1 shared-direction projection, whiten = ZCA)
and each seed {0,1,2}, score the FINAL checkpoint (never recon-MSE-selected)
with sae/eval_sae_metrics.py's suite (Hungarian MCC, uniqueness, matched F1),
paired against the Block-C baseline TopK seeds trained on the raw activations
(sae_data/hetdynamics_eqvar/sae_mixed_topk_seed{n}_final.pt).

Output: results/sae_shared_subspace_break.npy
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sae_metrics import eval_dir

VARIANTS = {
    "baseline": "sae_data/hetdynamics_eqvar",
    "proj": "sae_data/hetdynamics_eqvar_proj",
    "whiten": "sae_data/hetdynamics_eqvar_whiten",
}
SEEDS = (0, 1, 2)

out = {}
for var, d in VARIANTS.items():
    rows = []
    for s in SEEDS:
        ck = f"sae_mixed_topk_seed{s}_final.pt"
        r = eval_dir(d, variants=("mixed",), mixed_ckpt=ck)["mixed"]
        rows.append(r)
        print(f"{var} seed{s}: MCC={r['mcc']:.4f} uniq={r['mean_uniqueness']:+.4f} "
              f"F1={r['mean_f1']:.4f}  matched_r={np.round(r['matched_r'], 2)}")
    out[var] = dict(
        seeds=rows,
        mcc=np.array([r["mcc"] for r in rows]),
        uniq=np.array([r["mean_uniqueness"] for r in rows]),
        f1=np.array([r["mean_f1"] for r in rows]),
    )
    print(f"{var}: MCC {out[var]['mcc'].mean():.4f}±{out[var]['mcc'].std():.4f}  "
          f"uniq {out[var]['uniq'].mean():+.4f}±{out[var]['uniq'].std():.4f}  "
          f"F1 {out[var]['f1'].mean():.4f}±{out[var]['f1'].std():.4f}\n")

np.save("results/sae_shared_subspace_break.npy", out, allow_pickle=True)
print("saved -> results/sae_shared_subspace_break.npy")
