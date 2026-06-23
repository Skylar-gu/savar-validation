"""
Diagnostic: do 1-2 SAE features dominate the variance?

Re-uses the trade-off-sweep machinery (import-safe now that its driver is under
__main__). Trains one CoordCNN (strength, seed below), runs the spatial probe
with return_full=True, and reports how concentrated the per-feature variance is
(st ∝ variance), plus whether the highest-variance features are grid-locked or
content. ~18 min on GPU.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics_vs_position_tradeoff_sweep as S

STRENGTH = 1.0
SEED = 0

print(f"Loading windows ... (strength={STRENGTH}, seed={SEED})")
Xtr, Ytr = S.load_windows("train", S.N_REAL_TRAIN)
Xva, Yva = S.load_windows("val", S.N_REAL_VAL)
S.set_seed(SEED)
model, rmse = S.train_cnn(STRENGTH, Xtr, Ytr, Xva, Yva)
r = S.spatial_probe(model, SEED, return_full=True)
print(f"valRMSE={rmse:.4f}  #grid-locked={r['n_gridlocked']}  #content={r['n_content']}")

var, order, alive = r["var"], r["order"], r["alive"]
tv = var.sum(); n_alive = int(alive.sum())
cum = np.cumsum(var[order]) / tv
print(f"\nAlive features: {n_alive}/512   participation ratio (effective #feats): {r['part_ratio']:.1f}")
print("Variance share captured by top-k features:")
for k in [1, 2, 5, 10, 20, 50]:
    print(f"  top-{k:>3}: {cum[k-1]*100:5.1f}%")

print("\nTop-10 features by variance (share | pos_R2 | cont_R2 | bucket):")
for j in order[:10]:
    bucket = "grid-locked" if r["gl"][j] else ("content" if r["ct"][j] else "—")
    print(f"  feat {j:>3}: {var[j]/tv*100:5.1f}%  pos={r['pos_R2'][j]:.2f}  "
          f"cont={r['cont_R2'][j]:.2f}  {bucket}")

# how much of total variance lives in grid-locked vs content vs other
gl_share = var[r["gl"]].sum() / tv
ct_share = var[r["ct"]].sum() / tv
print(f"\nVariance by bucket:  grid-locked={gl_share*100:.1f}%  "
      f"content={ct_share*100:.1f}%  other={ (1-gl_share-ct_share)*100:.1f}%")
np.save("results/probe_variance_diag.npy",
        dict(strength=STRENGTH, seed=SEED, val_rmse=rmse,
             var=var, order=order, pos_R2=r["pos_R2"], cont_R2=r["cont_R2"],
             gl=r["gl"], ct=r["ct"], alive=alive, part_ratio=r["part_ratio"]))
print("Saved → results/probe_variance_diag.npy")
