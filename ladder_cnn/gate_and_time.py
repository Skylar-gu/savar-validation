"""Guardrail #6 data gate + timing probe. Run BEFORE anything at scale."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from common import *

print("=" * 74)
print("GATE 1 — ground truth")
G, gt = load_gt()
ref = np.load(ROOT / "results/pcmci_results.npy", allow_pickle=True).item()
gt_ref = set(map(tuple, ref["ground_truth"]))
print(f"  G shape {G.shape} finite={np.isfinite(G).all()}")
print(f"  cross edges derived : {len(gt)}")
print(f"  cross edges in results/pcmci_results.npy : {len(gt_ref)}")
assert gt == gt_ref, (sorted(gt), sorted(gt_ref))
print(f"  MATCH: {sorted(gt)}")
print(f"  true-Z ceiling on THIS dataset: F1={ref['f1'].mean():.4f} "
      f"P={ref['precision'].mean():.4f} R={ref['recall'].mean():.4f}")

print("\nGATE 2 — G identical across realisations")
paths = sorted((ROOT / "data/realisations").glob("realisation_*.npz"))
print(f"  n realisations: {len(paths)}")
bad = 0
for p in paths[::10]:
    d = np.load(p)
    if not np.array_equal(d["ground_truth_graph"].astype(np.float64), G):
        bad += 1
print(f"  spot-checked {len(paths[::10])} files, mismatches={bad}")
assert bad == 0

print("\nGATE 3 — activations / Z")
A = np.load(ROOT / "sae_data/base/activations_full.npy", mmap_mode="r")
Z = np.load(ROOT / "sae_data/base/Z_full.npy")
print(f"  activations {A.shape} {A.dtype}")
print(f"  Z_full      {Z.shape} {Z.dtype}")
sub = np.asarray(A[:10])
print(f"  finite (first 10 real): {np.isfinite(sub).all()}")
print(f"  all-zero rows (r,j,t): {(np.abs(sub).sum(-1) == 0).sum()} / {sub[...,0].size}")
print(f"  Z finite: {np.isfinite(Z).all()}  all-zero series: "
      f"{int((np.abs(Z).sum(-1)==0).sum())}")
print(f"  activation mean|.| {np.abs(sub).mean():.4f}  frac-zero {(sub==0).mean():.4f}")

print("\nGATE 4 — Z_full vs latent_states time alignment (T_eff = T - k)")
d0 = np.load(paths[0])
Zr = d0["latent_states"].astype(np.float64)  # (8, 500)
print(f"  raw latent_states {Zr.shape}, Z_full[0] {Z[0].shape}")
for off in (0, 1, 2, 3):
    seg = Zr[:, off:off + Z.shape[2]]
    if seg.shape[1] != Z.shape[2]:
        continue
    r = np.mean([np.corrcoef(seg[j], Z[0, j])[0, 1] for j in range(8)])
    print(f"    offset {off}: mean r(Z_full, latent_states) = {r:+.4f}")

print("\nGATE 5 — W")
W = d0["W"].astype(np.float64)
print(f"  W {W.shape} finite={np.isfinite(W).all()} rowsums={W.sum(1).round(3)}")
print(f"  nonzero per row: {(W!=0).sum(1)}")

print("\n" + "=" * 74)
print("TIMING PROBE — one PCMCI unit (T=497, C=8, tau_max=2)")
rng = np.random.default_rng(0)
x = rng.standard_normal((497, 8))
t0 = time.time(); pcmci_one(x); t1 = time.time()
print(f"  random 8-var series : {t1-t0:.3f} s")
Zt = Z[0].T  # (497, 8)
t0 = time.time(); pcmci_one(Zt); t1 = time.time()
t_z = t1 - t0
print(f"  true-Z 8-var series : {t_z:.3f} s")
x12 = rng.standard_normal((497, 12))
t0 = time.time(); pcmci_one(x12); t1 = time.time()
print(f"  random 12-var series: {t1-t0:.3f} s")
print(f"  => 100 realisations, 4 workers, 8 vars ~ {t_z*100/4:.0f} s per rung")
print(f"  => 200 null draws x 20 reals, 4 workers ~ {t_z*200*20/4/60:.1f} min per null")

print("\nTIMING PROBE — CNN forward, one realisation on CPU")
import torch
sys.path.insert(0, str(ROOT / "train" / "cnn"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH
torch.set_num_threads(2)
model = SpatioTemporalCNN(ny=NY, nx=NX, k=K, base_ch=BASE_CH)
ck = torch.load(ROOT / "checkpoints/base/best.pt", map_location="cpu")
model.load_state_dict(ck["model_state"]); model.eval()
print(f"  params {sum(p.numel() for p in model.parameters())/1e6:.2f} M  "
      f"val_rmse {ck['val_rmse']:.4f}")
cap = {}
model.res3.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o))
obs = d0["observations"].astype(np.float32).T.reshape(-1, NY, NX)  # (500,50,50)
frames = torch.from_numpy(obs)
B = 32
t0 = time.time()
with torch.no_grad():
    w = torch.stack([frames[t:t + K] for t in range(0, B)]).unsqueeze(1)
    model(w)
t1 = time.time()
print(f"  res3 shape {tuple(cap['a'].shape)}")
per_frame = (t1 - t0) / B
print(f"  {B} windows in {t1-t0:.2f} s -> {per_frame*1000:.1f} ms/window")
T_eff = obs.shape[0] - K
print(f"  one realisation (T_eff={T_eff}): {per_frame*T_eff:.0f} s")
print(f"  => 12 realisations, 1 pass : {per_frame*T_eff*12/60:.1f} min")
print(f"  => 12 realisations, 2 passes: {per_frame*T_eff*12*2/60:.1f} min")
print(f"  per-realisation per-pixel act tensor: "
      f"{T_eff*L*256*2/1e9:.2f} GB in float16")
print("=" * 74)
