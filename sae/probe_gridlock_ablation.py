"""
Task 2 — Causal ablation of grid-locked SAE directions in res3.

Everything so far has been *decodability*: grid-locked position structure can be
read out of res3. This script asks the causal question — is that structure
actually USED by the forecaster, or is it epiphenomenal?

Method (standard activation patching, restricted to a subspace):
  1. Run the trained CNN on val windows, hook res3 → a (B,256,50,50).
  2. At every pixel, normalize the 256-vector with the SAE's train mean/std,
     encode through the cached spatial SAE, and subtract ONLY the grid-locked
     features' decoder contribution:
         a_norm_ablated = a_norm − Σ_{f∈gl} feat_f · dec_f
     (all non-grid-locked signal, including SAE reconstruction error, is kept).
  3. Un-normalize, run model.head on the patched res3, measure forecast RMSE.
  4. ΔRMSE = RMSE_ablated − RMSE_baseline.

Controls (same machinery, different feature set):
  • content features  — the handful of cont_R2-dominant features
  • random features   — a random alive set of the SAME count as grid-locked
  • all features      — subtract the full SAE reconstruction residual direction
                        (sanity: how much the head even relies on res3 detail)

If grid-locked directions are epiphenomenal, ΔRMSE(grid-locked) ≈ 0 while content
ablation costs more. Predicted by the flat RMSE sweep + 8.8% variance share.
"""

import sys, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH

CKPT = "checkpoints_diurnal/best.pt"
DATA = "data/realisations_diurnal"
OUT = Path("sae_data_diurnal")
INPUT_DIM, N_FEATURES, K_TOPK = 256, 512, 25
NY = NX = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REAL_VAL = 12
T_STRIDE = 10
rng = np.random.default_rng(0)


class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d, n)
        self.decoder = nn.Linear(n, d)

    def encode(self, x):
        pre = self.encoder(x)
        v, i = torch.topk(pre, self.k, dim=-1)
        a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v))
        return a


# ── CNN ─────────────────────────────────────────────────────────────────────
model = SpatioTemporalCNN(ny=NY, nx=NX, k=K, base_ch=BASE_CH).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"]); model.eval()
print(f"Loaded {CKPT}  val RMSE={ckpt['val_rmse']:.4f}")
_cap = {}
model.res3.register_forward_hook(lambda m, i, o: _cap.__setitem__("a", o))

# ── SAE + train-normalization (must match how the SAE was trained) ───────────
d = np.load(OUT / "spatial_acts.npz")
Xtr = d["X"]
mean = torch.from_numpy(Xtr.mean(0).astype(np.float32)).to(DEVICE)        # (256,)
std = torch.from_numpy((Xtr.std(0) + 1e-8).astype(np.float32)).to(DEVICE)  # (256,)
sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
sae.load_state_dict(torch.load(OUT / "spatial_sae.pt", map_location=DEVICE))
sae.eval()
Wdec = sae.decoder.weight.detach()                                         # (256,512)

# ── feature sets (use the Task-1 no-one-hot taxonomy; identical to original) ──
sc = np.load(OUT / "content_r2_refit.npz")
alive = sc["alive"]; gl = sc["gl_no_oh"]; ct = sc["ct_no_oh"]
gl_idx = np.where(gl)[0]
ct_idx = np.where(ct)[0]
alive_idx = np.where(alive)[0]
rand_idx = rng.choice(alive_idx, size=len(gl_idx), replace=False)
print(f"feature sets:  grid-locked={len(gl_idx)}  content={len(ct_idx)}  "
      f"random(alive)={len(rand_idx)}  alive={len(alive_idx)}")

SETS = {
    "grid-locked": torch.from_numpy(gl_idx).to(DEVICE),
    "content": torch.from_numpy(ct_idx).to(DEVICE),
    "random-same-n": torch.from_numpy(rand_idx).to(DEVICE),
    "all-alive": torch.from_numpy(alive_idx).to(DEVICE),
}


def head_rmse_se(a):
    """squared-error sum and count for model.head on res3 acts a (B,256,H,W)."""
    pred = model.head(a)                                  # (B,1,H,W)
    return pred


# ── load val windows ─────────────────────────────────────────────────────────
paths = sorted(glob.glob(str(Path(DATA) / "realisation_*.npz")))[:N_REAL_VAL]
T_TOTAL = np.load(paths[0])["observations"].shape[1]

# accumulators
acc = {name: 0.0 for name in SETS}
acc["baseline"] = 0.0
removed_norm = {name: 0.0 for name in SETS}   # mean fraction of ||a_norm|| removed
tot = 0
nwin = 0

with torch.no_grad():
    for p in paths:
        obs = np.load(p)["observations"].astype(np.float32)              # (L,T)? -> shape (T,50,50)?
        # observations stored as (T, 50, 50) in diurnal set; handle both
        if obs.ndim == 2:                                                # (L, T)
            frames = obs.T.reshape(T_TOTAL, 1, NY, NX)
            targets_all = obs.T.reshape(T_TOTAL, NY, NX)
        else:                                                           # (T,50,50)
            frames = obs.reshape(T_TOTAL, 1, NY, NX)
            targets_all = obs.reshape(T_TOTAL, NY, NX)
        tlist = list(range(0, T_TOTAL - K, T_STRIDE))
        wins = np.stack([frames[t:t+K] for t in tlist]).transpose(0, 2, 1, 3, 4)
        tgt = targets_all[[t + K for t in tlist]]
        for i in range(0, len(wins), 64):
            xb = torch.from_numpy(wins[i:i+64]).float().to(DEVICE)
            yb = torch.from_numpy(tgt[i:i+64]).float().to(DEVICE).unsqueeze(1)
            model(xb)
            a = _cap["a"]                                               # (B,256,50,50)
            B = a.shape[0]
            # baseline
            pred0 = model.head(a)
            acc["baseline"] += ((pred0 - yb) ** 2).sum().item()
            tot += yb.numel(); nwin += B

            # normalize per-pixel: (B,256,H,W) -> (B*H*W,256)
            av = a.permute(0, 2, 3, 1).reshape(-1, INPUT_DIM)
            an = (av - mean) / std
            feats = sae.encode(an)                                      # (M,512)
            base_norm = an.norm(dim=1) + 1e-8
            for name, idx in SETS.items():
                contrib = feats[:, idx] @ Wdec[:, idx].T                # (M,256)
                an_ab = an - contrib
                removed_norm[name] += (contrib.norm(dim=1) / base_norm).sum().item()
                a_ab = (an_ab * std + mean).reshape(B, NY, NX, INPUT_DIM).permute(0, 3, 1, 2)
                pred = model.head(a_ab)
                acc[name] += ((pred - yb) ** 2).sum().item()

base_rmse = (acc["baseline"] / tot) ** 0.5
print(f"\nval windows={nwin}   pixels={tot:,}")
print(f"\n{'feature set':>14}  {'RMSE':>8}  {'ΔRMSE':>9}  {'Δ%':>7}  {'mean‖removed‖/‖a‖':>18}")
print(f"{'baseline':>14}  {base_rmse:8.4f}  {'—':>9}  {'—':>7}  {'—':>18}")
rows = {"baseline_rmse": base_rmse, "nwin": nwin}
for name in SETS:
    rmse = (acc[name] / tot) ** 0.5
    d_abs = rmse - base_rmse
    d_pct = 100 * d_abs / base_rmse
    frac = removed_norm[name] / nwin / (NY * NX)  # mean over pixels
    print(f"{name:>14}  {rmse:8.4f}  {d_abs:+9.5f}  {d_pct:+6.2f}%  {frac:18.4f}")
    rows[name] = dict(rmse=rmse, d_abs=d_abs, d_pct=d_pct, removed_frac=frac)

np.save("results/gridlock_ablation.npy", rows)
print("\nSaved → results/gridlock_ablation.npy")
print("\nInterpretation: ΔRMSE(grid-locked) ≈ 0 with ΔRMSE(content) larger ⇒ "
      "grid-locked directions are decodable but causally epiphenomenal.")
