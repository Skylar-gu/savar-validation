"""
Generate 100 realisations with DISJOINT BINARY mode masks.

Same ground-truth graph G, noise (D_y = I_L), and time length as the baseline.
The ONLY change vs the baseline is the within-block weight *shape*:

  baseline : each mode = an L1-normalised Gaussian bump over its 16×16 block
  binary   : each mode = a FLAT 0/1 indicator over its 16×16 block, then
             L1-normalised (every in-block cell = 1/256)

Both conventions L1-normalise (rows sum to 1), exactly like the baseline's
`blob /= blob.sum()`. So Z_j = W[j,:] @ y stays on the same scale and the
mode-weighted pooling in extract_activations is directly comparable. The
*only* degree of freedom that changes is the spatial profile: concentrated
Gaussian vs uniform flat. The modes are already spatially disjoint in the
baseline (8 of a 3×3 grid of 16px blocks), so this isolates "does the mode
shape drive the PC0 global-activity collapse?" — nothing else moves.

Output: data/realisations_binary/
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

# ── Override W: flat 0/1 disjoint masks (L1-normalised) ──────────────────────
# `positions` and `size` come from instantiate_model.py (find_mode_positions).
W = np.zeros((N, ny, nx))
for i in range(N):
    y1, y2, x1, x2 = positions[i]
    mask = np.ones((y2 - y1, x2 - x1))   # flat binary indicator over the block
    mask /= mask.sum()                   # L1-normalise (matches baseline convention)
    W[i, y1:y2, x1:x2] = mask

W_flat = W.reshape(N, L)                  # (8, 2500)
W_plus = np.linalg.pinv(W_flat)           # (2500, 8)

# sanity: disjoint support → W_flat W_flatᵀ is diagonal
_gram = W_flat @ W_flat.T
assert np.allclose(_gram - np.diag(np.diag(_gram)), 0.0, atol=1e-12), \
    "binary masks are not disjoint!"

N_REALISATIONS = 100
OUT_DIR = os.path.join("data", "realisations_binary")
os.makedirs(OUT_DIR, exist_ok=True)

# ── spectral radius for metadata (G unchanged, but recompute for record) ─────
from savar.functions import create_graph
_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

# ── precompute Phi(tau) @ W_flat for O(N·L) VAR loop ─────────────────────────
phi_mat = G                                              # (N, N, tau_max)
tau_max = phi_mat.shape[2]
phi_W   = [phi_mat[:, :, i] @ W_flat for i in range(tau_max)]

def generate_obs(noise_field: np.ndarray) -> np.ndarray:
    total_T = T + burn
    data    = noise_field.copy()
    for t in range(tau_max, total_T):
        for i in range(tau_max):
            data[:, t] += W_plus @ (phi_W[i] @ data[:, t - 1 - i])
    return data[:, burn:]

print(f"\nGenerating {N_REALISATIONS} realisations  [DISJOINT BINARY masks]")
print(f"  Grid: {ny}×{nx}  L={L}  N={N}  T={T}  burn={burn}")
print(f"  block size: {size}×{size} px, flat 1/{size*size:.0f} weight per in-block cell")
print(f"  Spectral radius: {spectral_radius:.4f}  (G unchanged from baseline)")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)

    eps_x       = rng.standard_normal((N, T + burn))
    eps_y       = rng.standard_normal((L, T + burn))   # D_y = I_L (baseline noise)
    noise_field = W_plus @ eps_x + eps_y

    obs = generate_obs(noise_field)
    Z   = W_flat @ obs

    np.savez_compressed(
        os.path.join(OUT_DIR, f"realisation_{seed:03d}.npz"),
        observations       = obs.astype(np.float32),
        latent_states      = Z.astype(np.float32),
        ground_truth_graph = G.astype(np.float32),
        W                  = W_flat.astype(np.float32),
        W_plus             = W_plus.astype(np.float32),
        metadata           = np.array([N, L, T, lam, seed, spectral_radius]),
    )

    if (seed + 1) % 10 == 0:
        elapsed = time.time() - t_start
        rate    = (seed + 1) / elapsed
        eta     = (N_REALISATIONS - seed - 1) / rate
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s")

# verification
d   = np.load(os.path.join(OUT_DIR, f"realisation_{N_REALISATIONS-1:03d}.npz"))
obs = d["observations"]
Z   = d["latent_states"]
print(f"\nVerification (realisation {N_REALISATIONS-1}):")
print(f"  obs  shape={obs.shape}  mean={obs.mean():.4f}  std={obs.std():.4f}")
print(f"  Z    shape={Z.shape}    mean={Z.mean():.4f}  std={Z.std():.4f}")
print(f"  W rows L1 sums: {W_flat.sum(axis=1)}  (all ≈ 1)")
