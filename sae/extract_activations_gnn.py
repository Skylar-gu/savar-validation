"""
Phase 7.1-GNN — Extract MeshGNN message-passing activations (mode-weighted pooling).

Analog of extract_activations.py, but for the GNN forecaster instead of the CNN.

Where the activations come from
-------------------------------
The CNN's res3 gives a (B, 256, 50, 50) per-spatial-location feature map that we
mode-weight pool.  The GNN's exact analog is the node hidden state H after the
last message-passing layer, shape (B, L=2500, hidden=256) — one 256-dim vector
per mesh node, immediately before the decoder.  So the same mode-weighted pool

    feat_j(t, c) = W[j, :] @ H(t, :, c)        (256-dim, one per mode)

applies verbatim (W[j,:] is the mode weight map, L1-normalised, from each .npz).
The gate-layer to read is the output of layers[N_MP-1] (default; overridable with
--layer).

Window alignment: window t = frames[t:t+K] predicts frame t+K, so the feature is
aligned to Z_j(t+K) — identical to the CNN pipeline.

Outputs (written to sae_data/gnn/)
----------------------------------
  activations_full.npy  (100, 8, T_eff, 256)   W[j,:] @ H(t) per realisation/mode
  Z_full.npy            (100, 8, T_eff)         Z_j(t+K) aligned to above
  ceilings.npy          (8,)                    per-mode ridge |r| ceiling (out-of-fold)

The per-mode SAE train/eval scripts consume sae_data/gnn/ via their --gnn flag.
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "gnn"))
from gnn_forecaster import MeshGNN, K, HIDDEN, N_MP

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default="checkpoints/finecadence/best.pt",
                 help="GNN checkpoint (model_state) to hook")
_ap.add_argument("--data", default="data/realisations_finecadence",
                 help="raw realisation_*.npz dir (observations, latent_states, W)")
_ap.add_argument("--out", default="sae_data/gnn", help="output dir")
_ap.add_argument("--layer", type=int, default=N_MP - 1,
                 help="which MP layer's output to read (default: last)")
_ap.add_argument("--stride", type=int, default=1,
                 help="temporal stride between windows (1 = every window)")
_ap.add_argument("--variant", default="plain",
                 help="plain|blurpool|refframe|slot (architecture sweep, v2)")
_ap.add_argument("--n-real", type=int, default=100, help="expected #realisations")
_ap.add_argument("--hook", default="auto",
                 help="mplayer (layers[LAYER] out) | decoder (final node H, pre-decoder) | auto")
_args = _ap.parse_args()

CKPT_PATH = Path(_args.ckpt)
DATA_DIR  = Path(_args.data)
OUT_DIR   = Path(_args.out)
LAYER     = _args.layer
STRIDE    = _args.stride

EXTRACT_BS = 64    # windows per GNN forward pass (GNN is heavier than the CNN)

OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY, NX = 50, 50
BASE_CH = HIDDEN


# ── load GNN ──────────────────────────────────────────────────────────────────

if _args.variant == "plain":
    model = MeshGNN(ny=NY, nx=NX, k=K).to(DEVICE)
else:
    from mesh_gnn_variants import VariantMeshGNN
    model = VariantMeshGNN(ny=NY, nx=NX, variant=_args.variant, k=K).to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])   # strict
model.eval()

# hook: for variants (esp. slot, where binding happens AFTER the MP stack) read
# the FINAL node states the model actually decodes = the input to decoder[0].
# For plain/blurpool/refframe this equals layers[-1] output; slot adds binding.
HOOK = _args.hook
if HOOK == "auto":
    HOOK = "decoder" if _args.variant != "plain" else "mplayer"
print(f"Loaded {_args.variant}  val RMSE = {ckpt.get('val_rmse', float('nan')):.4f}  "
      f"device = {DEVICE}  hook={HOOK}"
      + (f" layers[{LAYER}]" if HOOK == "mplayer" else " decoder-input"))

_captured = {}

if HOOK == "mplayer":
    def _mp_hook(module, inp, out):
        _captured["act"] = out   # (B, 2500, 256)
    model.layers[LAYER].register_forward_hook(_mp_hook)
else:
    def _dec_hook(module, inp, out):
        _captured["act"] = inp[0]   # (B, 2500, 256) final node H
    model.decoder[0].register_forward_hook(_dec_hook)


# ── extract ──────────────────────────────────────────────────────────────────

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert len(paths) == _args.n_real, f"expected {_args.n_real} realisations, found {len(paths)}"
N_REAL = len(paths)

T_TOTAL = int(np.load(paths[0])["observations"].shape[1])   # (L, T)
starts  = list(range(0, T_TOTAL - K, STRIDE))               # window start indices
T_eff   = len(starts)
N_MODES = 8

activations_full = np.empty((N_REAL, N_MODES, T_eff, BASE_CH), dtype=np.float32)
Z_full           = np.empty((N_REAL, N_MODES, T_eff),           dtype=np.float32)

print(f"\nExtracting activations  (MP layer H → mode-weighted pool → {BASE_CH}-dim per mode)")
print(f"  T_total={T_TOTAL}  stride={STRIDE}  windows/realisation={T_eff}")
print(f"  100 realisations × {N_MODES} modes × {T_eff} windows = {100*N_MODES*T_eff:,} samples\n")

with torch.no_grad():
    for r, path in enumerate(paths):
        d   = np.load(path)
        obs = d["observations"].astype(np.float32)    # (2500, T) = (L, T)
        Z   = d["latent_states"].astype(np.float32)   # (8, T)
        W   = d["W"].astype(np.float32)               # (8, 2500), L1-normalised

        W_t    = torch.from_numpy(W).to(DEVICE)                        # (8, 2500)
        frames = torch.from_numpy(obs.T.reshape(T_TOTAL, NY, NX))      # (T, 50, 50)

        # windows for the GNN: x = (B, K, ny, nx)
        windows = torch.stack([frames[t : t + K] for t in starts])    # (T_eff, K, 50, 50)

        acts_r = []
        for i in range(0, T_eff, EXTRACT_BS):
            batch = windows[i : i + EXTRACT_BS].to(DEVICE)             # (B, K, 50, 50)
            model(batch)
            H     = _captured["act"]                                    # (B, 2500, 256)
            # mode-weighted pool: feat[b, j, c] = W[j,:] @ H[b, :, c]
            feat  = torch.einsum("jl,blc->bjc", W_t, H)               # (B, 8, 256)
            acts_r.append(feat.cpu().numpy())

        acts_r_full = np.concatenate(acts_r, axis=0)                   # (T_eff, 8, 256)
        activations_full[r] = acts_r_full.transpose(1, 0, 2)          # (8, T_eff, 256)
        Z_full[r]           = Z[:, [t + K for t in starts]]           # (8, T_eff)

        if (r + 1) % 10 == 0:
            print(f"  [{r+1:3d}/100]")

np.save(OUT_DIR / "activations_full.npy", activations_full)
np.save(OUT_DIR / "Z_full.npy", Z_full)
print(f"\nSaved  activations_full.npy  {activations_full.shape}")
print(f"Saved  Z_full.npy            {Z_full.shape}")


# ── PCA sanity check + ridge ceiling per mode ─────────────────────────────────
# PCA: is the mode signal present at all?  Ridge: max |r| any linear readout of the
# 256 dims achieves out-of-fold (the SAE's theoretical alignment ceiling).

print("\n── PCA sanity check + ridge ceiling ─────────────────────────────────────")
print(f"  {'Mode':<6}  {'PCA var%':>8}  {'best PCA':>9}  {'PCA|r|':>7}  {'Ceil|r|':>8}")
print(f"  {'─'*50}")

CEIL_SUBSAMPLE = 40000   # cap samples for the ridge fit (speed)
rng = np.random.default_rng(0)

mode_max_r = []
ceilings   = []
for j in range(N_MODES):
    acts_j = activations_full[:, j, :, :].reshape(-1, BASE_CH).astype(np.float64)
    Z_j    = Z_full[:, j, :].reshape(-1).astype(np.float64)

    pca   = PCA(n_components=min(8, BASE_CH))
    A_pc  = pca.fit_transform(acts_j)
    corrs = [pearsonr(A_pc[:, i], Z_j)[0] for i in range(A_pc.shape[1])]
    best_i = int(np.argmax(np.abs(corrs)))
    max_r  = max(abs(c) for c in corrs)
    mode_max_r.append(max_r)
    var_top = pca.explained_variance_ratio_[:3].sum() * 100

    # ridge ceiling (out-of-fold, on a subsample)
    n = len(Z_j)
    sel = rng.choice(n, size=min(CEIL_SUBSAMPLE, n), replace=False)
    Xs, ys = acts_j[sel], Z_j[sel]
    ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
    yhat  = cross_val_predict(ridge, Xs, ys, cv=5)
    ceil  = abs(pearsonr(yhat, ys)[0])
    ceilings.append(ceil)

    print(f"  {'X'+str(j):<6}  {var_top:>7.1f}%  {'dim '+str(best_i):>9}  "
          f"{max_r:>7.3f}  {ceil:>8.3f}")

ceilings = np.array(ceilings, dtype=np.float64)
np.save(OUT_DIR / "ceilings.npy", ceilings)
print(f"\nSaved  ceilings.npy          {ceilings.shape}  (per-mode ridge |r| ceiling)")

gate = min(mode_max_r)
if gate < 0.20:
    print(f"\n  GATE WEAK — min PCA max|r| = {gate:.3f}. Mode signal is faint in GNN")
    print("  activations; SAE alignment will be correspondingly limited (a finding).")
else:
    print(f"\n  GATE PASSED — all 8 modes align with PCA (min max|r| = {gate:.3f}).")
    print("  Proceed to SAE training:  python sae/train_sae_per_mode.py --gnn")
