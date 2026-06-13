"""
SAVAR model definition — ground-truth parameters.
Follows savar_dataset_requirements.md exactly.

All downstream scripts exec() this file to get:
  N, ny, nx, L, W, W_flat, W_plus, links_coeffs, lam, D_x, D_y, T, burn, G
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

from savar.savar import SAVAR, dict_to_matrix
from savar.functions import check_stability, create_random_mode
from savar.model_generator import SavarGenerator

# ── 1. Spatial domain ───────────────────────────────────────────────────────
N  = 8
ny = 50
nx = 50
L  = ny * nx   # 2500

# ── 2. Weight matrix W — deterministic nonoverlapping Gaussian blobs ─────────
# SavarGenerator.find_mode_positions gives a 3×3 grid of size-16 blocks;
# we use the first N=8 of the 9 slots. random_mode=False → symmetric blob.
size, positions = SavarGenerator.find_mode_positions(res=(ny, nx), n_var=N)

W = np.zeros((N, ny, nx))
for i in range(N):
    y1, y2, x1, x2 = positions[i]
    blob_h, blob_w = y2 - y1, x2 - x1
    # create_random_mode((size_x, size_y)) returns shape (size_y, size_x)
    blob = create_random_mode((blob_w, blob_h), random=False)
    blob /= blob.sum()          # L1-normalise
    W[i, y1:y2, x1:x2] = blob

W_flat = W.reshape(N, L)       # (8, 2500)
W_plus = np.linalg.pinv(W_flat)  # (2500, 8)

# ── 3. Ground-truth causal graph G = Phi(tau) ────────────────────────────────
#   links_coeffs format: { j : [((i, -tau), coeff), ...] }
#   meaning X_i(t-tau) → X_j(t) with given coefficient.
#
#   Graph properties (per requirements):
#     - X0: hub (high out-degree)
#     - X3: converging node (two parents at different lags)
#     - X7: pure sink (zero out-degree)
#     - mixed lags: lag-1 and lag-2 edges present
#     - negative edges: X0→X5, X2→X3, X3→X7
#     - diverging paths: X0→X1, X0→X3
#     - converging paths: X0→X3 and X2→X3; X3→X6 and X5→X6
#     - weak downstream feedback: X2→X0 (lag-2)
# ─────────────────────────────────────────────────────────────────────────────
links_coeffs = {
    0: [((0, -1),  0.45), ((2, -2),  0.22)],               # auto + feedback from X2
    1: [((1, -1),  0.50), ((0, -1),  0.35)],               # auto + hub drive
    2: [((2, -1),  0.35), ((1, -1),  0.40)],               # auto + chain
    3: [((3, -1),  0.55), ((0, -1),  0.30), ((2, -1), -0.30)],  # auto + converging
    4: [((4, -1),  0.40), ((1, -2),  0.25)],               # auto + lag-2 only
    5: [((5, -1),  0.30), ((4, -1),  0.35), ((0, -2), -0.20)],  # auto + chain + neg
    6: [((6, -1),  0.50), ((3, -1),  0.30), ((5, -2),  0.25)],  # auto + two paths
    7: [((7, -1),  0.45), ((6, -1),  0.20), ((3, -2), -0.15)],  # auto + sink inputs
}

# ── 4. Stationarity check ────────────────────────────────────────────────────
check_stability(links_coeffs)

G = dict_to_matrix(links_coeffs)   # (N, N, tau_max) = (8, 8, 2)

# ── 5. Noise parameters ──────────────────────────────────────────────────────
lam = 1.0            # noise_strength  λ
D_x = np.eye(N)      # I_N — independent latent innovations
D_y = np.eye(L)      # I_L — per-grid-point independent noise

# ── 6. Time series length ────────────────────────────────────────────────────
T    = 500   # usable timesteps
burn = 200   # burn-in (discarded)

# ── 7. Sigma_y verification ──────────────────────────────────────────────────
# Sigma_y = lambda * W+ D_x (W+)^T + D_y = W+ W+^T + I_L
# Guaranteed positive definite because D_y = I_L.
Sigma_y_diag_approx = W_plus @ W_plus.T   # low-rank part; I_L adds 1 to all eigenvalues

# ── Print summary ────────────────────────────────────────────────────────────
_tau_max = G.shape[2]
print(f"Model config loaded")
print(f"  N={N}, grid={ny}×{nx}, L={L}, T={T}, burn={burn}")
print(f"  blob size: {size}×{size} px, {N} of {(ny//size)*(nx//size)} slots used")
print(f"  tau_max={_tau_max}, lambda={lam}")
print(f"  D_x=I_{N}, D_y=I_{L}")
print(f"  Stationarity: OK")
