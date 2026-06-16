"""
Phase 5 (PDF) — full per-feature forecast_importance[i] sweep.

The gridlock ablation (`probe_gridlock_ablation.py`) measured ΔRMSE for *buckets*
of spatial-SAE features (grid-locked / content / random / all). This script runs
the per-feature version: for every alive SAE feature i, subtract ONLY that feature's
decoder contribution from res3, re-run model.head, and record

    forecast_importance[i] = RMSE_ablated(i) − RMSE_baseline

i.e. how much the forecaster's accuracy degrades when feature i is removed. The
result is a forecast-influence ranking over the whole dictionary, which we then
cross-reference against the position/content taxonomy from content_r2_refit.npz.

Same activation-patching machinery as the gridlock script (per-pixel encode with
the SAE's train mean/std, subtract feat_i·dec_i, un-normalize, head, RMSE).

Prediction (from the gridlock thread): forecast_importance concentrates on the
handful of content features; grid-locked features sit at ≈0.
"""

import sys, glob, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH

_ap = argparse.ArgumentParser()
_ap.add_argument("--split", choices=["raw", "train", "val", "test"], default="raw",
                 help="raw = full timeline (NOT held out: CNN trained on the first 70%% of it); "
                      "test/val = chronological held-out segment, genuinely out-of-sample for the CNN.")
_ap.add_argument("--n_real", type=int, default=8, help="realisations to evaluate over")
_ap.add_argument("--t_stride", type=int, default=20, help="temporal window stride")
_a = _ap.parse_args()

CKPT = "checkpoints_diurnal/best.pt"
DATA = ("data/realisations_diurnal" if _a.split == "raw"
        else f"data/splits_diurnal/{_a.split}")
OUT = Path("sae_data_diurnal")
INPUT_DIM, N_FEATURES, K_TOPK = 256, 512, 25
NY = NX = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REAL_VAL = _a.n_real
T_STRIDE = _a.t_stride
SUFFIX = "" if _a.split == "raw" else f"_{_a.split}"


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
print(f"Evaluating on split='{_a.split}'  ({DATA})  "
      f"{'[HELD-OUT, out-of-sample for the CNN]' if _a.split in ('val','test') else '[RAW full timeline — ~70%% overlaps CNN training steps]'}")
_cap = {}
model.res3.register_forward_hook(lambda m, i, o: _cap.__setitem__("a", o))

# ── SAE + train normalization (must match how the SAE was trained) ───────────
d = np.load(OUT / "spatial_acts.npz")
Xtr = d["X"]
mean = torch.from_numpy(Xtr.mean(0).astype(np.float32)).to(DEVICE)         # (256,)
std = torch.from_numpy((Xtr.std(0) + 1e-8).astype(np.float32)).to(DEVICE)  # (256,)
sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
sae.load_state_dict(torch.load(OUT / "spatial_sae.pt", map_location=DEVICE))
sae.eval()
Wdec = sae.decoder.weight.detach()                                          # (256,512)

# ── taxonomy (no-one-hot, identical to the gridlock ablation) ────────────────
sc = np.load(OUT / "content_r2_refit.npz")
alive = sc["alive"]                                  # (512,) bool
gl = sc["gl_no_oh"]; ct = sc["ct_no_oh"]
position_R2 = sc["position_R2"]; content_R2 = sc["content_R2_no_oh"]
alive_idx = np.where(alive)[0]
print(f"alive features: {len(alive_idx)}/{N_FEATURES}  "
      f"(grid-locked={gl.sum()}, content={ct.sum()})")

# ── load windows ─────────────────────────────────────────────────────────────
paths = sorted(glob.glob(str(Path(DATA) / "realisation_*.npz")))[:N_REAL_VAL]

# accumulators: summed squared error over all pixels
se = np.zeros(N_FEATURES, dtype=np.float64)           # per-feature ablated SE
se_base = 0.0
removed_norm = np.zeros(N_FEATURES, dtype=np.float64)  # mean ‖removed‖/‖a‖ accumulator
tot = 0
nwin = 0

alive_t = torch.from_numpy(alive_idx).to(DEVICE)

with torch.no_grad():
    for p in paths:
        obs = np.load(p)["observations"].astype(np.float32)
        if obs.ndim == 2:                              # (L, T) — raw realisations
            T_cur = obs.shape[1]
            frames = obs.T.reshape(T_cur, 1, NY, NX)
            targets_all = obs.T.reshape(T_cur, NY, NX)
        else:                                          # (T, 50, 50) — split files
            T_cur = obs.shape[0]
            frames = obs.reshape(T_cur, 1, NY, NX)
            targets_all = obs.reshape(T_cur, NY, NX)
        tlist = list(range(0, T_cur - K, T_STRIDE))
        wins = np.stack([frames[t:t+K] for t in tlist]).transpose(0, 2, 1, 3, 4)
        tgt = targets_all[[t + K for t in tlist]]
        for i in range(0, len(wins), 64):
            xb = torch.from_numpy(wins[i:i+64]).float().to(DEVICE)
            yb = torch.from_numpy(tgt[i:i+64]).float().to(DEVICE).unsqueeze(1)
            model(xb)
            a = _cap["a"]                              # (B,256,50,50)
            B = a.shape[0]

            pred0 = model.head(a)
            se_base += ((pred0 - yb) ** 2).sum().item()
            tot += yb.numel(); nwin += B

            av = a.permute(0, 2, 3, 1).reshape(-1, INPUT_DIM)   # (M,256)
            an = (av - mean) / std
            feats = sae.encode(an)                              # (M,512)
            base_norm = an.norm(dim=1) + 1e-8

            # per-feature ablation over alive features only
            for fi in alive_idx:
                contrib = torch.outer(feats[:, fi], Wdec[:, fi])  # (M,256)
                an_ab = an - contrib
                removed_norm[fi] += (contrib.norm(dim=1) / base_norm).sum().item()
                a_ab = (an_ab * std + mean).reshape(B, NY, NX, INPUT_DIM).permute(0, 3, 1, 2)
                pred = model.head(a_ab)
                se[fi] += ((pred - yb) ** 2).sum().item()
        print(f"  {Path(p).name}: cumulative windows={nwin}")

base_rmse = (se_base / tot) ** 0.5
rmse = np.full(N_FEATURES, np.nan)
rmse[alive_idx] = (se[alive_idx] / tot) ** 0.5
forecast_importance = rmse - base_rmse                 # ΔRMSE (nan for dead)
removed_frac = removed_norm / max(nwin, 1) / (NY * NX)

print(f"\nval windows={nwin}   pixels={tot:,}   baseline RMSE={base_rmse:.4f}")

# ── ranking ───────────────────────────────────────────────────────────────────
order = alive_idx[np.argsort(-forecast_importance[alive_idx])]
def tag(i):
    return "GRID" if gl[i] else ("CONT" if ct[i] else "  · ")

print(f"\nTop-15 features by forecast_importance (ΔRMSE):")
print(f"  {'feat':>5}  {'ΔRMSE':>9}  {'Δ%':>7}  {'posR2':>6}  {'contR2':>6}  "
      f"{'rmNorm':>6}  tax")
for i in order[:15]:
    print(f"  {i:>5}  {forecast_importance[i]:+9.5f}  "
          f"{100*forecast_importance[i]/base_rmse:+6.2f}%  "
          f"{position_R2[i]:>6.3f}  {content_R2[i]:>6.3f}  "
          f"{removed_frac[i]:>6.3f}  {tag(i)}")

# ── taxonomy-level summary (mean importance per bucket) ───────────────────────
def bucket_mean(mask):
    idx = np.where(mask & alive)[0]
    return forecast_importance[idx].mean() if len(idx) else np.nan, len(idx)

gl_mean, gl_n = bucket_mean(gl)
ct_mean, ct_n = bucket_mean(ct)
other = alive & ~gl & ~ct
ot_mean, ot_n = bucket_mean(other)
print(f"\nMean forecast_importance by taxonomy bucket:")
print(f"  grid-locked (n={gl_n:>3}): {gl_mean:+.5f}")
print(f"  content     (n={ct_n:>3}): {ct_mean:+.5f}")
print(f"  other-alive (n={ot_n:>3}): {ot_mean:+.5f}")

# correlation of importance with content/position decodability
fi_alive = forecast_importance[alive_idx]
c_cont = np.corrcoef(fi_alive, content_R2[alive_idx])[0, 1]
c_pos = np.corrcoef(fi_alive, position_R2[alive_idx])[0, 1]
print(f"\ncorr(forecast_importance, content_R2) = {c_cont:+.3f}")
print(f"corr(forecast_importance, position_R2) = {c_pos:+.3f}")

# ── save ──────────────────────────────────────────────────────────────────────
out = {
    "forecast_importance": forecast_importance,   # (512,) ΔRMSE, nan for dead
    "rmse": rmse,
    "baseline_rmse": base_rmse,
    "removed_frac": removed_frac,
    "alive": alive,
    "gl_no_oh": gl, "ct_no_oh": ct,
    "position_R2": position_R2, "content_R2": content_R2,
    "ranking": order,
    "nwin": nwin,
    "corr_content_R2": c_cont, "corr_position_R2": c_pos,
}
out["split"] = _a.split
np.save(f"results/forecast_importance_sweep{SUFFIX}.npy", out)
print(f"\nSaved → results/forecast_importance_sweep{SUFFIX}.npy")
if _a.split in ("val", "test"):
    print("Caveat: forecaster RMSE here is genuinely out-of-sample (CNN never trained on "
          "these timesteps). The spatial-SAE dictionary/taxonomy, however, were extracted "
          "from the full timeline (incl. test) — that affects feature *definitions*, not the "
          "CNN's forecast, and the SAE is unsupervised wrt the forecast target.")
