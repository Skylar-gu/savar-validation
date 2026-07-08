"""
Phase 7.1-GNN MULTIVAR — Extract multivar MeshGNN activations, pooled to NC modes
through the BLOCK-DIAGONAL Ŵ (channel-aware fork of extract_activations_gnn.py).

WHAT CHANGES vs the single-channel parent:
  * The forecaster is the multivar MeshGNN (train/gnn/gnn_forecaster_multivar.py):
    per node it carries C observed channels, input (B, C, K, ny, nx), output
    (B, C, ny, nx). C is read from mv_meta.
  * Observations are (C, L, T); windows fed to the GNN are (B, C, K, ny, nx).
  * There are NC = N*C latent modes (channel-major m = channel*N + node), and the
    pooling map is the BLOCK-DIAGONAL Ŵ = I_C ⊗ W of shape (NC, C*L). The last
    message-passing layer's hidden H is (B, L, 256) — one vector per SPATIAL node.
    To pool it with Ŵ we channel-tile H to (B, C*L, 256) and apply Ŵ, giving NC
    pooled feature rows:  feat[b, m, :] = Ŵ[m, :] @ H_tiled[b, :, :].

    DOCUMENTED LIMITATION: the hidden state is per SPATIAL node (channels are
    mixed inside the 256-dim vector), so co-located modes (ch0,n) and (ch1,n)
    receive the SAME pooled vector. The activation path cannot separate channels
    of one footprint without a channel-aware readout; the pixel-side pooling of
    the raw (C,L) observation field does NOT share this limitation.

Outputs (written to sae_data/gnn_multivar/)
  activations_full.npy  (N_REAL, NC, T_eff, 256)
  Z_full.npy            (N_REAL, NC, T_eff)      Z_m(t+K) aligned to above
  ceilings.npy          (NC,)                    per-mode ridge |r| ceiling
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
from gnn_forecaster_multivar import MeshGNN, K, HIDDEN, N_MP

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default="checkpoints/multivar/best.pt",
                 help="multivar GNN checkpoint (model_state) to hook")
_ap.add_argument("--data", default="data/realisations_multivar2",
                 help="raw realisation_*.npz dir (observations (C,L,T), latent_states, W)")
_ap.add_argument("--out", default="sae_data/gnn_multivar", help="output dir")
_ap.add_argument("--layer", type=int, default=N_MP - 1,
                 help="which MP layer's output to read (default: last)")
_ap.add_argument("--stride", type=int, default=1,
                 help="temporal stride between windows (1 = every window)")
_ap.add_argument("--n-real", type=int, default=100, help="expected #realisations")
_ap.add_argument("--hook", default="auto",
                 help="mplayer (layers[LAYER] out) | decoder (final node H, pre-decoder) | auto")
_args = _ap.parse_args()

CKPT_PATH = Path(_args.ckpt)
DATA_DIR  = Path(_args.data)
OUT_DIR   = Path(_args.out)
LAYER     = _args.layer
STRIDE    = _args.stride

EXTRACT_BS = 64

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY, NX = 50, 50
BASE_CH = HIDDEN

# ── multivar layout from mv_meta [N, C, L, NC, tau_max] ──────────────────────
paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert len(paths) == _args.n_real, f"expected {_args.n_real} realisations, found {len(paths)}"
N_REAL = len(paths)
d0 = np.load(paths[0])
N_S, C, L_CH, NC, _TAU = (int(x) for x in d0["mv_meta"])
LC = C * L_CH
N_MODES = NC

# BLOCK-DIAGONAL pooling map  Ŵ = I_C ⊗ W  → (NC, C*L)
W_SINGLE = d0["W"].astype(np.float32)                     # (N, L_CH)
W_EXT = np.zeros((NC, LC), dtype=np.float32)
for ch in range(C):
    W_EXT[ch * N_S:(ch + 1) * N_S, ch * L_CH:(ch + 1) * L_CH] = W_SINGLE


# ── load multivar GNN ────────────────────────────────────────────────────────
model = MeshGNN(ny=NY, nx=NX, k=K, channels=C).to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])   # strict
model.eval()

HOOK = _args.hook
if HOOK == "auto":
    HOOK = "mplayer"
print(f"Loaded multivar MeshGNN  C={C}  NC={NC}  "
      f"val RMSE = {ckpt.get('val_rmse', float('nan')):.4f}  device = {DEVICE}  hook={HOOK}"
      + (f" layers[{LAYER}]" if HOOK == "mplayer" else " decoder-input"))

_captured = {}
if HOOK == "mplayer":
    model.layers[LAYER].register_forward_hook(
        lambda module, inp, out: _captured.update(act=out))       # (B, L, 256)
else:
    model.decoder[0].register_forward_hook(
        lambda module, inp, out: _captured.update(act=inp[0]))    # (B, L, 256)


# ── extract ──────────────────────────────────────────────────────────────────
T_TOTAL = int(d0["observations"].shape[2])                 # (C, L, T)
starts  = list(range(0, T_TOTAL - K, STRIDE))
T_eff   = len(starts)

W_t = torch.from_numpy(W_EXT).to(DEVICE)                   # (NC, C*L)

activations_full = np.empty((N_REAL, N_MODES, T_eff, BASE_CH), dtype=np.float32)
Z_full           = np.empty((N_REAL, N_MODES, T_eff),           dtype=np.float32)

print(f"\nExtracting activations  (MP layer H -> channel-tile -> block-diagonal Ŵ pool -> {BASE_CH}-dim per NC mode)")
print(f"  T_total={T_TOTAL}  stride={STRIDE}  windows/realisation={T_eff}")
print(f"  {N_REAL} realisations x {N_MODES} modes x {T_eff} windows = {N_REAL*N_MODES*T_eff:,} samples\n")

with torch.no_grad():
    for r, path in enumerate(paths):
        d   = np.load(path)
        obs = d["observations"].astype(np.float32)         # (C, L, T)
        Z   = d["latent_states"].astype(np.float32)        # (NC, T)
        # (C, L, T) -> (T, C, NY, NX)
        frames = torch.from_numpy(np.transpose(obs, (2, 0, 1)).reshape(T_TOTAL, C, NY, NX))

        # windows for the multivar GNN: (B, C, K, ny, nx)
        windows = torch.stack([frames[t:t + K] for t in starts])       # (T_eff, K, C, ny, nx)
        windows = windows.permute(0, 2, 1, 3, 4).contiguous()          # (T_eff, C, K, ny, nx)

        acts_r = []
        for i in range(0, T_eff, EXTRACT_BS):
            batch = windows[i:i + EXTRACT_BS].to(DEVICE)               # (B, C, K, ny, nx)
            model(batch)
            H = _captured["act"]                                        # (B, L, 256)
            H_tiled = torch.cat([H] * C, dim=1)                         # (B, C*L, 256)
            feat = torch.einsum("jl,blc->bjc", W_t, H_tiled)            # (B, NC, 256)
            acts_r.append(feat.cpu().numpy())

        acts_r_full = np.concatenate(acts_r, axis=0)                    # (T_eff, NC, 256)
        activations_full[r] = acts_r_full.transpose(1, 0, 2)           # (NC, T_eff, 256)
        Z_full[r]           = Z[:, [t + K for t in starts]]            # (NC, T_eff)

        if (r + 1) % 10 == 0:
            print(f"  [{r+1:3d}/{N_REAL}]")

np.save(OUT_DIR / "activations_full.npy", activations_full)
np.save(OUT_DIR / "Z_full.npy", Z_full)
print(f"\nSaved  activations_full.npy  {activations_full.shape}")
print(f"Saved  Z_full.npy            {Z_full.shape}")


# ── PCA sanity check + ridge ceiling per NC mode ─────────────────────────────
print("\n── PCA sanity check + ridge ceiling ─────────────────────────────────────")
print(f"  {'Mode':<8}  {'PCA var%':>8}  {'best PCA':>9}  {'PCA|r|':>7}  {'Ceil|r|':>8}")
print(f"  {'─'*52}")

CEIL_SUBSAMPLE = 40000
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

    n = len(Z_j)
    sel = rng.choice(n, size=min(CEIL_SUBSAMPLE, n), replace=False)
    Xs, ys = acts_j[sel], Z_j[sel]
    ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
    yhat  = cross_val_predict(ridge, Xs, ys, cv=5)
    ceil  = abs(pearsonr(yhat, ys)[0])
    ceilings.append(ceil)

    ch, node = j // N_S, j % N_S
    print(f"  {'ch'+str(ch)+'X'+str(node):<8}  {var_top:>7.1f}%  {'dim '+str(best_i):>9}  "
          f"{max_r:>7.3f}  {ceil:>8.3f}")

ceilings = np.array(ceilings, dtype=np.float64)
np.save(OUT_DIR / "ceilings.npy", ceilings)
print(f"\nSaved  ceilings.npy          {ceilings.shape}  (per-mode ridge |r| ceiling)")

gate = min(mode_max_r)
if gate < 0.20:
    print(f"\n  GATE WEAK — min PCA max|r| = {gate:.3f}. Mode signal is faint in GNN")
    print("  activations (compounded by co-located modes sharing activation geometry).")
else:
    print(f"\n  GATE PASSED — all {N_MODES} modes align with PCA (min max|r| = {gate:.3f}).")
