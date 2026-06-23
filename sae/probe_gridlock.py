"""
Step 1 — Grid-lock diagnostic on res3 activation MAPS (no spatial pooling).

Question: does the CNN develop position-sensitive ("grid-locked") activation
structure even though the SAVAR data has no privileged spatial location?

Method
------
The CNN is translation-equivariant by construction (Conv2d/Conv3d weights are
spatially shared; BatchNorm affine is per-channel; the head is 1x1). The ONLY
spatial-symmetry breaker is zero-padding at the borders. So:

  * Feed SPATIALLY STRUCTURELESS input (white noise, std-matched to the data).
    The input distribution is translation-invariant, so under a perfectly
    equivariant network the time-mean activation map would be spatially flat
    (up to sampling noise). ANY spatial structure that survives is therefore
    100% architecture-induced — the grid-locked artifact.
  * Feed REAL windows for comparison (structure should sit on the blobs).

For each res3 channel c we accumulate the time-mean map M_c(y,x) and the
temporal-variance map V_c(y,x) over many samples, then score:

  S_c     = spatial std of M_c over the 2500 pixels        (how non-flat)
  T_c     = mean temporal std  = mean_pixels sqrt(V_c)     (input-driven jitter)
  floor_c = T_c / sqrt(N)                                  (sampling-noise floor)
  gridlock_c = S_c / floor_c   (>>1 ⇒ real fixed spatial structure, not noise)
  border_ratio_c = (energy of M_c in border ring) / (chance fraction)

A channel with high gridlock_c AND border_ratio_c>1 under STRUCTURELESS input is
a genuine grid-locked (architecture) feature.
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH

_ap = argparse.ArgumentParser()
_ap.add_argument("--dy005", action="store_true")
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--baseline", action="store_true")
_ap.add_argument("--n_real", type=int, default=20, help="realisations to sample real windows from")
_ap.add_argument("--max_samples", type=int, default=4000)
_ap.add_argument("--border", type=int, default=3, help="border ring width in pixels")
_a = _ap.parse_args()

if _a.diurnal:
    CKPT, DATA, OUT = "checkpoints_diurnal/best.pt", "data/realisations_diurnal", "sae_data_diurnal"
elif _a.dy005:
    CKPT, DATA, OUT = "checkpoints_dy005/best.pt", "data/realisations_dy005", "sae_data_dy005"
else:
    CKPT, DATA, OUT = "checkpoints/best.pt", "data/realisations", "sae_data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY = NX = 50
L = NY * NX
B = 128

# ── load model + hook res3 ────────────────────────────────────────────────────
model = SpatioTemporalCNN(ny=NY, nx=NX, k=K, base_ch=BASE_CH).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded {CKPT}  val RMSE={ckpt['val_rmse']:.4f}  device={DEVICE}")

_cap = {}
model.res3.register_forward_hook(lambda m, i, o: _cap.__setitem__("a", o))

# ── gather REAL windows + data std + blob mask ────────────────────────────────
paths = sorted(Path(DATA).glob("realisation_*.npz"))[: _a.n_real]
d0 = np.load(paths[0])
W0 = d0["W"].astype(np.float32).reshape(8, NY, NX)
blob_mask = (np.abs(W0).sum(0) > 0)                       # (50,50) bool: blob support
T_TOTAL = d0["observations"].shape[1]

real_windows = []
obs_vals = []
for p in paths:
    obs = np.load(p)["observations"].astype(np.float32)  # (L, T)
    obs_vals.append(obs.ravel())
    frames = obs.T.reshape(T_TOTAL, 1, NY, NX)
    for t in range(0, T_TOTAL - K, 3):                   # stride 3 to decorrelate
        real_windows.append(frames[t : t + K])           # (K,1,50,50)? -> fix below
real_windows = np.stack(real_windows)                    # (M, K, 1, 50, 50)
real_windows = real_windows.transpose(0, 2, 1, 3, 4)     # (M, 1, K, 50, 50)
obs_std = float(np.concatenate(obs_vals).std())
rng = np.random.default_rng(0)
if len(real_windows) > _a.max_samples:
    sel = rng.choice(len(real_windows), _a.max_samples, replace=False)
    real_windows = real_windows[sel]
print(f"Real windows: {real_windows.shape}   obs_std={obs_std:.4f}   blob pixels={blob_mask.sum()}/{L}")


# ── accumulator over activation maps ──────────────────────────────────────────
class MapAccum:
    def __init__(self):
        self.n = 0
        self.s = torch.zeros(BASE_CH, NY, NX, dtype=torch.float64, device=DEVICE)
        self.s2 = torch.zeros(BASE_CH, NY, NX, dtype=torch.float64, device=DEVICE)

    def add(self, act):                                   # act (b,256,50,50)
        a = act.double()
        self.s += a.sum(0)
        self.s2 += (a * a).sum(0)
        self.n += act.shape[0]

    def finalize(self):
        mean = (self.s / self.n)                          # (256,50,50)
        var = (self.s2 / self.n) - mean * mean
        return mean.cpu().numpy(), var.clamp_min(0).cpu().numpy()


def run_regime(get_batch, n_total, label):
    acc = MapAccum()
    with torch.no_grad():
        for i in range(0, n_total, B):
            x = get_batch(i, min(B, n_total - i)).to(DEVICE)
            model(x)
            acc.add(_cap["a"])
    mean, var = acc.finalize()
    print(f"  {label}: accumulated {acc.n} samples")
    return mean, var, acc.n


def real_batch(i, b):
    return torch.from_numpy(real_windows[i : i + b]).float()

def noise_batch(i, b):
    return torch.from_numpy(rng.standard_normal((b, 1, K, NY, NX)).astype(np.float32) * obs_std)


print("\nForward passes:")
M_real, V_real, n_real = run_regime(real_batch, len(real_windows), "REAL       ")
M_rand, V_rand, n_rand = run_regime(noise_batch, _a.max_samples, "STRUCTURELESS")


# ── scoring ───────────────────────────────────────────────────────────────────
border = np.zeros((NY, NX), bool)
border[: _a.border, :] = border[-_a.border :, :] = True
border[:, : _a.border] = border[:, -_a.border :] = True
chance_border = border.mean()
corner = np.zeros((NY, NX), bool)
cs = _a.border + 2
corner[:cs, :cs] = corner[:cs, -cs:] = corner[-cs:, :cs] = corner[-cs:, -cs:] = True
chance_corner = corner.mean()


def score(M, V, n):
    # per-channel spatial structure of the time-mean map vs sampling-noise floor
    S = M.reshape(BASE_CH, -1).std(1)                      # spatial std of mean map
    T = np.sqrt(V).reshape(BASE_CH, -1).mean(1)            # mean temporal std
    floor = T / np.sqrt(n) + 1e-9
    gridlock = S / floor                                   # >>1 ⇒ real fixed structure
    # energy localisation of the (centered) mean map
    Mc = M - M.reshape(BASE_CH, -1).mean(1)[:, None, None]
    e = Mc ** 2
    etot = e.reshape(BASE_CH, -1).sum(1) + 1e-12
    border_ratio = (e[:, border].sum(1) / etot) / chance_border
    corner_ratio = (e[:, corner].sum(1) / etot) / chance_corner
    blob_ratio = (e[:, blob_mask].sum(1) / etot) / blob_mask.mean()
    return dict(S=S, T=T, gridlock=gridlock,
                border_ratio=border_ratio, corner_ratio=corner_ratio, blob_ratio=blob_ratio)


sr = score(M_real, V_real, n_real)
sn = score(M_rand, V_rand, n_rand)


def summarize(s, tag):
    gl = s["gridlock"]
    print(f"\n── {tag} ──")
    print(f"  gridlock S/floor : median {np.median(gl):.1f}   p90 {np.percentile(gl,90):.1f}   max {gl.max():.1f}")
    print(f"  border energy ratio (chance=1): mean {s['border_ratio'].mean():.2f}  "
          f"frac channels >1.5: {(s['border_ratio']>1.5).mean():.2f}")
    print(f"  corner energy ratio (chance=1): mean {s['corner_ratio'].mean():.2f}")
    print(f"  blob   energy ratio (chance=1): mean {s['blob_ratio'].mean():.2f}")


summarize(sr, "REAL input")
summarize(sn, "STRUCTURELESS input (any structure = pure architecture)")

# headline numbers
ch_mean_rand = np.abs(M_rand).mean(0)                      # (50,50) channel-averaged |mean|
border_vs_interior = ch_mean_rand[border].mean() / ch_mean_rand[~border].mean()
print("\n══ HEADLINE ══")
print(f"  Structureless input, channel-avg |time-mean map|:")
print(f"    border / interior energy = {border_vs_interior:.2f}x   (1.0 = no grid-lock)")
n_gridlocked = int(((sn["gridlock"] > 5) & (sn["border_ratio"] > 1.5)).sum())
print(f"    grid-locked channels (S/floor>5 AND border>1.5x): {n_gridlocked}/{BASE_CH}")
print(f"  → CNN {'DOES' if border_vs_interior > 1.15 or n_gridlocked > 0 else 'does NOT'} "
      f"invent position structure on symmetric data.")

# ── save maps + figure ────────────────────────────────────────────────────────
outp = Path(OUT) / "gridlock_maps.npz"
np.savez_compressed(outp, M_real=M_real.astype(np.float32), V_real=V_real.astype(np.float32),
                    M_rand=M_rand.astype(np.float32), V_rand=V_rand.astype(np.float32),
                    blob_mask=blob_mask, border=border,
                    gridlock_rand=sn["gridlock"], border_ratio_rand=sn["border_ratio"])
print(f"\nSaved maps → {outp}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    im0 = ax[0].imshow(np.abs(M_real).mean(0)); ax[0].set_title("REAL: channel-avg |mean map|")
    im1 = ax[1].imshow(ch_mean_rand); ax[1].set_title("STRUCTURELESS: channel-avg |mean map|")
    ax[2].imshow(blob_mask, cmap="gray"); ax[2].set_title("blob support")
    for a, im in ((ax[0], im0), (ax[1], im1)):
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    figp = Path("figures") / "gridlock_step1.png"
    fig.savefig(figp, dpi=120)
    print(f"Saved figure → {figp}")
except Exception as e:
    print(f"(figure skipped: {e})")
