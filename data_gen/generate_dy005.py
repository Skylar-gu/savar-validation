"""
Generate 100 realisations with D_y = 0.05 * I_L (low observation noise).
Same ground-truth graph G and weight matrix W as the baseline experiment.
Output: data/realisations_dy005/
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

# Override noise parameter
DY_SCALE   = 0.05          # D_y = 0.05 * I_L  (vs 1.0 baseline)
EPS_Y_STD  = DY_SCALE ** 0.5   # sqrt(0.05) ≈ 0.2236

N_REALISATIONS = 100
OUT_DIR = os.path.join("data", "realisations_dy005")
os.makedirs(OUT_DIR, exist_ok=True)

# ── spectral radius for metadata ─────────────────────────────────────────────
from savar.functions import create_graph
_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

# ── precompute Phi(tau) @ W_flat for O(N·L) VAR loop ────────────────────────
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

print(f"\nGenerating {N_REALISATIONS} realisations  [D_y = {DY_SCALE} × I_L]")
print(f"  Grid: {ny}×{nx}  L={L}  N={N}  T={T}  burn={burn}")
print(f"  Spectral radius: {spectral_radius:.4f}")
print(f"  eps_y std: {EPS_Y_STD:.4f}  (eps_x std: 1.0000)")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)

    eps_x       = rng.standard_normal((N, T + burn))
    eps_y       = EPS_Y_STD * rng.standard_normal((L, T + burn))
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
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
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
