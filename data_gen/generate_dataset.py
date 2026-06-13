"""
Generate 100 independent SAVAR realisations.

Each realisation shares the same W and G but uses a different random seed
for the noise process, producing different trajectories.

Noise is generated using the factored form to avoid sampling from the full
2500×2500 covariance matrix:

    Sigma_y = lambda * W+ D_x (W+)^T + D_y
            = W+ W+^T + I_L          (with lambda=1, D_x=I, D_y=I)

    noise(t) = W+ @ eps_x(t) + eps_y(t)
               eps_x ~ N(0, I_N),  eps_y ~ N(0, I_L)

This is O(N*L*T) per realisation vs O(L^3) for full Cholesky.

Output: data/realisations/realisation_NNN.npz  (100 files)
"""

import numpy as np
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))
from savar.savar import SAVAR

exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

N_REALISATIONS = 100
OUT_DIR = os.path.join("data", "realisations")
os.makedirs(OUT_DIR, exist_ok=True)

# ── spectral radius for metadata ─────────────────────────────────────────────
from savar.functions import create_graph
_g   = create_graph(links_coeffs, return_lag=False)   # (N, N, tau_max)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

# ── efficient VAR propagation ─────────────────────────────────────────────────
# SAVAR._create_linear evaluates  W⁺ @ Φ @ W @ x  left-to-right, which
# materialises a (L×L) intermediate matrix at every timestep — O(N·L²) per step.
# Correct parenthesisation gives O(N·L) per step: W⁺ @ (Φ @ (W @ x)).
# Precompute Φ(τ) @ W_flat once per lag → (N, N) @ (N, L) = (N, L).
phi_mat   = G                                              # (N, N, tau_max)
tau_max   = phi_mat.shape[2]
phi_W     = [phi_mat[:, :, i] @ W_flat for i in range(tau_max)]   # list of (N, L)

def generate_obs(noise_field: np.ndarray) -> np.ndarray:
    """
    Run the SAVAR linear recurrence on a pre-generated noise field.
    noise_field: (L, T+burn)
    Returns obs: (L, T)
    """
    total_T = T + burn
    data    = noise_field.copy()            # (L, total_T)

    for t in range(tau_max, total_T):
        for i in range(tau_max):
            # O(N·L) per step: project → latent → spatial
            data[:, t] += W_plus @ (phi_W[i] @ data[:, t - 1 - i])

    return data[:, burn:]                   # discard burn-in, return (L, T)

print(f"\nGenerating {N_REALISATIONS} realisations")
print(f"  Grid: {ny}×{nx}  L={L}  N={N}  T={T}  burn={burn}")
print(f"  Spectral radius: {spectral_radius:.4f}")
print(f"  Noise: factored  W⁺ ε_x + ε_y  (no Cholesky)")
print(f"  VAR loop: O(N·L) per step  (parenthesised correctly)")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)

    # Factored noise: Σ_y = W⁺(W⁺)ᵀ + I_L
    eps_x       = rng.standard_normal((N, T + burn))
    eps_y       = rng.standard_normal((L, T + burn))
    noise_field = W_plus @ eps_x + eps_y              # (L, T+burn)

    obs = generate_obs(noise_field)                   # (L, T)
    Z   = W_flat @ obs                                # (N, T)

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
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s elapsed  "
              f"ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s  "
      f"({total/N_REALISATIONS:.2f}s per realisation)")

# ── Quick verification on last realisation ───────────────────────────────────
d  = np.load(os.path.join(OUT_DIR, f"realisation_{N_REALISATIONS-1:03d}.npz"))
obs = d["observations"]
Z   = d["latent_states"]
print(f"\nVerification (realisation {N_REALISATIONS-1}):")
print(f"  observations shape : {obs.shape}  dtype={obs.dtype}")
print(f"  latent_states shape: {Z.shape}")
print(f"  obs  mean={obs.mean():.4f}  std={obs.std():.4f}")
print(f"  Z    mean={Z.mean():.4f}  std={Z.std():.4f}")
print(f"  G shape            : {d['ground_truth_graph'].shape}")
print(f"  W  shape           : {d['W'].shape}")
print(f"  W+ shape           : {d['W_plus'].shape}")
print(f"  metadata           : {d['metadata']}")
