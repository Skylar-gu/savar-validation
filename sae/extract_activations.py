"""
Phase 7.1 — Extract CNN res3 activations (mode-weighted pooling).
Phase 7.1b — PCA sanity check: confirm mode signal is present before SAE training.

Why mode-weighted pooling instead of global average pool
---------------------------------------------------------
Each of the 8 modes occupies ~1/9 of the 50×50 grid.  Global average pooling
attenuates each mode's signal by this factor (~9×), which reduces all feature-mode
correlations to ~0.01 — too weak for either PCA or SAE to recover.

Instead, for each mode j, we compute a mode-weighted feature vector:
    feat_j(t, c) = W[j, :] @ act(t, c, :)        (256-dim, one per mode)
where W[j, :] is the mode weight map stored in each realisation's .npz file.
This mirrors how Z_j(t) = W[j, :] @ y_t is computed, so the features are
directly comparable to the latent states.

Output shape: (100, 8, 497, 256) — one 256-dim view per realisation, mode, and
time step.  The SAE is trained on all 100×8×497 = 397,600 samples but at the
same 256-dim input dimensionality as the original design.

Window alignment: window t predicts frame t+K, so feature aligned to Z_j(t+K).

Outputs (written to sae_data/)
-------------------------------
activations_full.npy  shape (100, 8, 497, 256)
    activations_full[r, j, t] = W[j,:] @ res3(t) for realisation r, mode j

Z_full.npy            shape (100, 8, 497)
    Z_full[r, j, t] = Z_j(t+K) — latent state aligned to above
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH

_ap = argparse.ArgumentParser()
_ap.add_argument("--dy005", action="store_true", help="Use D_y=0.05·I_L dataset and CNN")
_ap.add_argument("--diurnal", action="store_true", help="Use diurnal/annual dataset and CNN")
_args = _ap.parse_args()

if _args.diurnal:
    CKPT_PATH = Path("checkpoints_diurnal/best.pt")
    DATA_DIR  = Path("data/realisations_diurnal")
    OUT_DIR   = Path("sae_data_diurnal")
elif _args.dy005:
    CKPT_PATH = Path("checkpoints_dy005/best.pt")
    DATA_DIR  = Path("data/realisations_dy005")
    OUT_DIR   = Path("sae_data_dy005")
else:
    CKPT_PATH = Path("checkpoints/best.pt")
    DATA_DIR  = Path("data/realisations")
    OUT_DIR   = Path("sae_data")

EXTRACT_BS = 128   # windows per forward pass during extraction

OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY, NX = 50, 50


# ── load CNN ──────────────────────────────────────────────────────────────────

model = SpatioTemporalCNN(ny=NY, nx=NX, k=K, base_ch=BASE_CH).to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded CNN  val RMSE = {ckpt['val_rmse']:.4f}  device = {DEVICE}")

# ── register hook on res3 ─────────────────────────────────────────────────────

_captured = {}

def _res3_hook(module, inp, out):
    _captured["act"] = out   # (B, 256, 50, 50)

model.res3.register_forward_hook(_res3_hook)


# ── extract ──────────────────────────────────────────────────────────────────

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert len(paths) == 100

# Infer series length from the data (T=500 baseline, T=2920 diurnal)
T_TOTAL = int(np.load(paths[0])["observations"].shape[1])
T_eff   = T_TOTAL - K   # valid windows per realisation
N_MODES = 8

activations_full = np.empty((100, N_MODES, T_eff, BASE_CH), dtype=np.float32)
Z_full           = np.empty((100, N_MODES, T_eff),           dtype=np.float32)

print(f"\nExtracting activations  (res3 → mode-weighted pool → {BASE_CH}-dim per mode)")
print(f"  100 realisations × {N_MODES} modes × {T_eff} windows = {100*N_MODES*T_eff:,} samples\n")

with torch.no_grad():
    for r, path in enumerate(paths):
        d      = np.load(path)
        obs    = d["observations"].astype(np.float32)   # (2500, 500) = (L, T)
        Z      = d["latent_states"].astype(np.float32)  # (8, 500)
        W      = d["W"].astype(np.float32)              # (8, 2500)

        # W rows are L1-normalised; use as-is for weighted averaging
        W_t    = torch.from_numpy(W).to(DEVICE)         # (8, 2500)

        # reshape to (T, 1, ny, nx) for windowing
        frames = torch.from_numpy(obs.T.reshape(T_TOTAL, 1, NY, NX))  # (T, 1, 50, 50)

        # build all windows: (T_eff, 1, K, 50, 50)
        windows = torch.stack([
            frames[t : t + K].permute(1, 0, 2, 3)   # (1, K, 50, 50)
            for t in range(T_eff)
        ])  # (497, 1, K, 50, 50)

        # batched forward passes
        acts_r = []   # will be list of (B, 256, L) arrays
        for i in range(0, T_eff, EXTRACT_BS):
            batch = windows[i : i + EXTRACT_BS].to(DEVICE)   # (B, 1, K, 50, 50)
            model(batch)
            act   = _captured["act"]                          # (B, 256, 50, 50)
            B     = act.shape[0]
            act_flat = act.view(B, BASE_CH, -1)               # (B, 256, 2500)
            # mode-weighted pool: feat[b, j, c] = W[j,:] @ act[b, c, :]
            # = einsum('jl, bcl -> bjc', W_t, act_flat)
            feat  = torch.einsum("jl,bcl->bjc", W_t, act_flat)  # (B, 8, 256)
            acts_r.append(feat.cpu().numpy())

        acts_r_full = np.concatenate(acts_r, axis=0)  # (497, 8, 256)
        activations_full[r] = acts_r_full.transpose(1, 0, 2)  # (8, 497, 256)
        Z_full[r]           = Z[:, K:]                         # (8, 497)

        if (r + 1) % 10 == 0:
            print(f"  [{r+1:3d}/100]")

np.save(OUT_DIR / "activations_full.npy", activations_full)
np.save(OUT_DIR / "Z_full.npy", Z_full)
print(f"\nSaved  activations_full.npy  {activations_full.shape}")
print(f"Saved  Z_full.npy            {Z_full.shape}")


# ── PCA sanity check (Phase 7.1b) ─────────────────────────────────────────────
# For each mode j, run PCA on that mode's projected activations (shape N_real*T_eff, 256)
# and check whether any PCA dim correlates with Z_j.

print("\n── PCA sanity check ─────────────────────────────────────────────────────")
print("  For each mode j: PCA on mode-j activations vs Z_j time series\n")

print(f"  {'Mode':<6}  {'PCA var%':>8}  {'best PCA':>9}  {'max |r|':>8}")
print(f"  {'─'*40}")

mode_max_r = []
for j in range(N_MODES):
    # activations_full[:, j, :, :] shape: (100, 497, 256)
    acts_j = activations_full[:, j, :, :].reshape(-1, BASE_CH).astype(np.float64)
    Z_j    = Z_full[:, j, :].reshape(-1).astype(np.float64)   # (49700,)

    pca   = PCA(n_components=min(8, BASE_CH))
    A_pc  = pca.fit_transform(acts_j)   # (49700, 8)

    corrs = [pearsonr(A_pc[:, i], Z_j)[0] for i in range(A_pc.shape[1])]
    best_i = int(np.argmax(np.abs(corrs)))
    max_r  = max(abs(c) for c in corrs)
    mode_max_r.append(max_r)
    var_top = pca.explained_variance_ratio_[:3].sum() * 100
    flag   = "  *** WEAK" if max_r < 0.3 else ""
    print(f"  {'X'+str(j):<6}  {var_top:>7.1f}%  {'dim '+str(best_i):>9}  {max_r:>8.3f}{flag}")

print()
gate = min(mode_max_r)
if gate < 0.25:
    print("  GATE FAILED — at least one mode has no PCA alignment > 0.25")
    print("  SAE will not find the mode structure. Investigate the CNN activations.")
    sys.exit(1)
else:
    print(f"  GATE PASSED — all 8 modes align with PCA (min max|r| = {gate:.3f})")
    weak = [f"X{j}" for j, r in enumerate(mode_max_r) if r < 0.3]
    if weak:
        print(f"  NOTE: {weak} below 0.3 — expect weaker SAE alignment for these modes.")
    print("  Proceed to SAE training.")
