"""
Step 2a — Extract res3 activations WITHOUT spatial pooling.

The per-mode pipeline W-pools the 50x50 map into one vector per mode, destroying
the spatial axis, so it cannot host a grid-locked (position) feature. Here we
keep space: each SAMPLE is the 256-dim res3 vector at one pixel (y,x) of one
window. We tag every sample with its (y,x), the local field value (content), and
its mode id (which blob it sits in, or -1 for a dead-zone pixel), so a spatial
SAE feature can later be scored for position- vs content-selectivity.

Output: sae_data_diurnal/spatial_acts.npz
  X        (Nsamp, 256)  res3 activation vectors
  ys, xs   (Nsamp,)      pixel coordinates
  content  (Nsamp,)      target-frame field value at that pixel
  mode_id  (Nsamp,)      blob index 0..7 at that pixel, -1 if dead zone
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH

_ap = argparse.ArgumentParser()
_ap.add_argument("--dy005", action="store_true")
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--n_real", type=int, default=40)
_ap.add_argument("--t_stride", type=int, default=30)
_ap.add_argument("--pix_per_frame", type=int, default=100)
_a = _ap.parse_args()

if _a.diurnal:
    CKPT, DATA, OUT = "checkpoints_diurnal/best.pt", "data/realisations_diurnal", "sae_data_diurnal"
elif _a.dy005:
    CKPT, DATA, OUT = "checkpoints_dy005/best.pt", "data/realisations_dy005", "sae_data_dy005"
else:
    CKPT, DATA, OUT = "checkpoints/best.pt", "data/realisations", "sae_data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY = NX = 50
B = 128

model = SpatioTemporalCNN(ny=NY, nx=NX, k=K, base_ch=BASE_CH).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"]); model.eval()
print(f"Loaded {CKPT}  val RMSE={ckpt['val_rmse']:.4f}")
_cap = {}
model.res3.register_forward_hook(lambda m, i, o: _cap.__setitem__("a", o))

paths = sorted(Path(DATA).glob("realisation_*.npz"))[: _a.n_real]
W0 = np.load(paths[0])["W"].astype(np.float32).reshape(8, NY, NX)
mode_map = np.where(np.abs(W0).sum(0) > 0, np.abs(W0).argmax(0), -1).astype(np.int64)  # (50,50)
T_TOTAL = np.load(paths[0])["observations"].shape[1]

rng = np.random.default_rng(0)
X, YS, XS, CONT, MODE = [], [], [], [], []

with torch.no_grad():
    for r, p in enumerate(paths):
        obs = np.load(p)["observations"].astype(np.float32)          # (L,T)
        frames = obs.T.reshape(T_TOTAL, 1, NY, NX)
        tlist = list(range(0, T_TOTAL - K, _a.t_stride))
        wins = np.stack([frames[t:t+K] for t in tlist]).transpose(0, 2, 1, 3, 4)  # (nt,1,K,50,50)
        targets = obs.T.reshape(T_TOTAL, NY, NX)[[t + K for t in tlist]]          # (nt,50,50)
        for i in range(0, len(wins), B):
            xb = torch.from_numpy(wins[i:i+B]).float().to(DEVICE)
            model(xb)
            act = _cap["a"].permute(0, 2, 3, 1).cpu().numpy()        # (b,50,50,256)
            tb = targets[i:i+B]
            for b in range(act.shape[0]):
                yy = rng.integers(0, NY, _a.pix_per_frame)
                xx = rng.integers(0, NX, _a.pix_per_frame)
                X.append(act[b, yy, xx])                              # (P,256)
                YS.append(yy); XS.append(xx)
                CONT.append(tb[b, yy, xx]); MODE.append(mode_map[yy, xx])
        if (r + 1) % 10 == 0:
            print(f"  [{r+1}/{len(paths)}]  samples so far: {sum(len(a) for a in X):,}")

X = np.concatenate(X).astype(np.float32)
YS = np.concatenate(YS).astype(np.int64); XS = np.concatenate(XS).astype(np.int64)
CONT = np.concatenate(CONT).astype(np.float32); MODE = np.concatenate(MODE).astype(np.int64)
outp = Path(OUT) / "spatial_acts.npz"
np.savez(outp, X=X, ys=YS, xs=XS, content=CONT, mode_id=MODE, mode_map=mode_map)
print(f"\nSaved {outp}   X={X.shape}  dead-zone frac={np.mean(MODE<0):.2f}")
