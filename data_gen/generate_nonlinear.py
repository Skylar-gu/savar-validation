"""
Generate SAVAR realisations with NONLINEAR latent dynamics on top of the
GraphCast-like 6-hourly diurnal/annual variant.

Motivation
----------
The baseline / dy005 / diurnal datasets are all driven by a *linear* VAR. A
linear-Gaussian generating process makes the optimal forecaster essentially
linear, so a nonlinear CNN's internal representation collapses onto a single
global-activity direction (PC0 ~ 86-88% of activation variance) and SAE
features come out polysemantic. That collapse is a property of the LINEAR
GAUSSIAN data-gen process, not of GraphCast — a real ML weather model is
trained on genuinely nonlinear, advective, multivariate dynamics.

This generator adds nonlinearity so that the forecaster MUST learn nonlinear
features, letting us test whether the activation-collapse / polysemanticity
finding survives once the generating process is no longer linear.

What is kept identical to generate_diurnal.py (controlled comparison):
  - grid, modes, weights W / W_plus
  - ground-truth causal graph G (same parents → same EDGES; only the functional
    form of each edge changes), so the edge set used as PCMCI ground truth is
    unchanged
  - 6 h cadence, T = 2920 (2 yr), D_y = 0.05 * I_L
  - diurnal + annual forcing, afternoon heteroskedasticity

What changes — two bounded, graph-respecting nonlinear terms in MODE space:
  1. Saturating autoregression. Each lagged mode state m is passed through
        g(m) = (1 - NL_ALPHA) * m + NL_ALPHA * tanh(m)
     before the linear map G(tau). At small amplitude g(m) ≈ m (Jacobian at 0
     equals the linear model, so stability / spectral radius is preserved); at
     large amplitude it saturates (bounded growth, like real climate).
  2. Bilinear (advective) coupling along the lag-1 cross edges. For every
     cross-mode edge i -> j present in G(tau=1) with coefficient c, a product
     term c * m_i * m_j feeds mode j. This is the Lorenz-style multiplicative
     interaction that a linear forecaster cannot represent. It is wrapped in
     tanh and scaled by NL_BETA so it can never exceed |NL_BETA| per mode,
     guaranteeing the process stays bounded.

NL_ALPHA = 0 and NL_BETA = 0 reproduces generate_diurnal.py exactly.

Knobs (env-overridable for quick smoke tests):
  NL_ALPHA, NL_BETA, N_REALISATIONS, NL_T

Output: data/realisations_nonlinear/
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

# ── noise level (GraphCast-like low noise) ───────────────────────────────────
DY_SCALE  = 0.05
EPS_Y_STD = DY_SCALE ** 0.5

# ── temporal cadence ─────────────────────────────────────────────────────────
DT_HOURS = 6
T        = int(os.environ.get("NL_T", 2920))     # 2 years at 6 h
burn     = 200

P_D = 24 // DT_HOURS                              # diurnal period = 4 steps
P_A = int(round(365.25 * 24 / DT_HOURS))         # annual period  = 1461 steps
HET_AMP = 0.6

# ── nonlinearity knobs ───────────────────────────────────────────────────────
NL_ALPHA = float(os.environ.get("NL_ALPHA", 0.5))   # saturating-AR blend  [0,1]
NL_BETA  = float(os.environ.get("NL_BETA",  0.15))  # bilinear coupling cap per mode

N_REALISATIONS = int(os.environ.get("N_REALISATIONS", 100))
OUT_DIR = os.path.join("data", "realisations_nonlinear")
os.makedirs(OUT_DIR, exist_ok=True)

# ── fixed forcing climatology (shared across realisations) ────────────────────
x_center = np.array([(p[2] + p[3]) / 2 for p in positions[:N]])   # "longitude"
y_center = np.array([(p[0] + p[1]) / 2 for p in positions[:N]])   # "latitude"
phi_d = -2 * np.pi * x_center / nx
phi_a = np.where(y_center > ny / 2, np.pi, 0.0)

_clim_rng = np.random.default_rng(12345)
A_d = _clim_rng.uniform(0.75, 1.40, size=N)
A_a = _clim_rng.uniform(0.35, 0.70, size=N)

# ── spectral radius (of the LINEAR skeleton) for metadata ─────────────────────
from savar.functions import create_graph
_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

# ── causal-graph tensors ──────────────────────────────────────────────────────
tau_max = G.shape[2]

# Lag-1 cross-edge list (i -> j, i != j) for the bilinear coupling term.
G1 = G[:, :, 0]                                   # G1[j, i] = coeff of X_i(t-1) in X_j(t)
cross_edges = [(j, i, G1[j, i]) for j in range(N) for i in range(N)
               if i != j and abs(G1[j, i]) > 0]

# ── deterministic latent forcing (same every realisation) ─────────────────────
total_T = T + burn
t_idx   = np.arange(total_T)
s_diurnal = A_d[:, None] * np.sin(2 * np.pi * t_idx / P_D + phi_d[:, None])
s_annual  = A_a[:, None] * np.sin(2 * np.pi * t_idx / P_A + phi_a[:, None])
s_latent  = s_diurnal + s_annual
forcing_field = W_plus @ s_latent

conv_env = 1.0 + HET_AMP * np.clip(
    np.sin(2 * np.pi * t_idx / P_D + phi_d[:, None] - np.pi / 2), 0, None)


def _g_sat(m):
    """Saturating nonlinearity; reduces to identity when NL_ALPHA == 0."""
    return (1.0 - NL_ALPHA) * m + NL_ALPHA * np.tanh(m)


def _bilinear(m1):
    """Bounded advective coupling along lag-1 cross edges. Returns (N,)."""
    if NL_BETA == 0.0 or not cross_edges:
        return np.zeros(N)
    q = np.zeros(N)
    for j, i, c in cross_edges:
        q[j] += c * m1[i] * m1[j]
    return NL_BETA * np.tanh(q)


def generate_obs(noise_field: np.ndarray) -> np.ndarray:
    data = (noise_field + forcing_field).copy()    # forcing enters before recurrence
    for t in range(tau_max, total_T):
        contrib = np.zeros(N)
        for i in range(tau_max):
            m_lag = W_flat @ data[:, t - 1 - i]    # (N,) lagged mode state
            contrib += G[:, :, i] @ _g_sat(m_lag)  # saturating linear map
        m1 = W_flat @ data[:, t - 1]
        contrib += _bilinear(m1)                   # bounded bilinear coupling
        data[:, t] += W_plus @ contrib
    return data[:, burn:]


print(f"\nGenerating {N_REALISATIONS} realisations  [NONLINEAR diurnal+annual, D_y={DY_SCALE}*I_L]")
print(f"  Grid: {ny}x{nx}  L={L}  N={N}  T={T}  burn={burn}")
print(f"  Cadence: dt={DT_HOURS}h  P_diurnal={P_D}  P_annual={P_A}")
print(f"  Nonlinearity: NL_ALPHA(sat)={NL_ALPHA}  NL_BETA(bilinear)={NL_BETA}  cross-edges={len(cross_edges)}")
print(f"  Linear-skeleton spectral radius: {spectral_radius:.4f}")
print(f"  eps_y std: {EPS_Y_STD:.4f}   het_amp: {HET_AMP}")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()
max_abs_global = 0.0

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)
    eps_x = rng.standard_normal((N, total_T)) * conv_env
    eps_y = EPS_Y_STD * rng.standard_normal((L, total_T))
    noise_field = W_plus @ eps_x + eps_y

    obs = generate_obs(noise_field)
    Z   = W_flat @ obs
    max_abs_global = max(max_abs_global, float(np.abs(Z).max()))

    np.savez_compressed(
        os.path.join(OUT_DIR, f"realisation_{seed:03d}.npz"),
        observations       = obs.astype(np.float32),
        latent_states      = Z.astype(np.float32),
        ground_truth_graph = G.astype(np.float32),
        W                  = W_flat.astype(np.float32),
        W_plus             = W_plus.astype(np.float32),
        forcing_latent     = s_latent[:, burn:].astype(np.float32),
        diurnal_amp        = A_d.astype(np.float32),
        diurnal_phase      = phi_d.astype(np.float32),
        annual_amp         = A_a.astype(np.float32),
        annual_phase       = phi_a.astype(np.float32),
        cycle_meta         = np.array([DT_HOURS, P_D, P_A, HET_AMP], dtype=np.float32),
        nl_meta            = np.array([NL_ALPHA, NL_BETA], dtype=np.float32),
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
    )

    if (seed + 1) % 10 == 0:
        elapsed = time.time() - t_start
        rate    = (seed + 1) / elapsed
        eta     = (N_REALISATIONS - seed - 1) / rate
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s")
print(f"  Global max |Z| across realisations: {max_abs_global:.3f}  "
      f"({'STABLE' if np.isfinite(max_abs_global) and max_abs_global < 1e3 else 'UNSTABLE — reduce NL_BETA'})")

# ── verification ──────────────────────────────────────────────────────────────
d   = np.load(os.path.join(OUT_DIR, f"realisation_{N_REALISATIONS-1:03d}.npz"))
obs = d["observations"]
Z   = d["latent_states"]
print(f"\nVerification (realisation {N_REALISATIONS-1}):")
print(f"  obs  shape={obs.shape}  mean={obs.mean():.4f}  std={obs.std():.4f}")
print(f"  Z    shape={Z.shape}    mean={Z.mean():.4f}  std={Z.std():.4f}")
