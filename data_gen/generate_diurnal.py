"""
Generate 100 SAVAR realisations with GraphCast-like 6-hourly cadence and
deterministic atmospheric cycles layered on top of the linear VAR dynamics.

Same ground-truth graph G and weight matrix W as the baseline / dy005 runs —
the causal structure is UNCHANGED, so Phase 6 (PCMCI) and Phase 7 (SAE) stay
directly comparable. What changes is the addition of deterministic, physically
motivated forcing that the dynamics propagate through Phi:

  dt = 6 h  →  1 step = 6 hours,  T=2920 steps = 2 years.

Added phenomena (Tier 1, adapted for 6-h sampling):
  #1 Diurnal cycle (24 h = 4 steps), per-mode phase tied to LONGITUDE
     (blob x-center). Eastern modes peak first → westward heating wave.
     Phasing by longitude is deliberate: it spreads the diurnal signal across
     several activation PCs instead of collapsing it onto the global-activity
     direction (PC0), which would wreck mode-level SAE alignment.
  #2 Annual cycle (1 yr = 1461 steps), per-mode phase tied to LATITUDE
     (blob y-center): northern vs southern modes get opposite seasonal phase.
     (The 12-h semidiurnal tide is DROPPED — it sits exactly at the 6-h Nyquist
      frequency and aliases away, just as it does in ERA5/GraphCast.)
  #4 Afternoon-peaked heteroskedasticity: per-mode innovation variance peaks a
     few hours past local noon (convective turbulence), phased by longitude.

The diurnal/annual CLIMATOLOGY (amplitudes, phases) is fixed across all
realisations — same Earth, different weather — exactly like W and G. Only the
stochastic noise differs per seed.

Forcing is injected in latent (mode) space and added to the field BEFORE the
VAR recurrence, so it is propagated and amplified through the teleconnections
(matching SAVAR.generate_data semantics), then projected with W_plus.

Output: data/realisations_diurnal/   (100 files)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

# ── noise level ───────────────────────────────────────────────────────────────
# D_y = 0.05 * I_L  → the low-noise, reanalysis-like (GraphCast-like) regime.
# Flip to 1.0 to reproduce the high-noise baseline regime for comparison.
DY_SCALE  = 0.05
EPS_Y_STD = DY_SCALE ** 0.5

# ── temporal cadence ──────────────────────────────────────────────────────────
DT_HOURS = 6                       # GraphCast cadence
T        = 2920                    # 2 years  (2920 * 6 h = 730 days)
# burn comes from instantiate_model.py (200); override for a longer transient
burn     = 200

P_D = 24 // DT_HOURS               # diurnal period in steps  = 4
P_A = int(round(365.25 * 24 / DT_HOURS))   # annual period in steps = 1461

HET_AMP = 0.6                      # afternoon variance enhancement (fraction)

N_REALISATIONS = 100
OUT_DIR = os.path.join("data", "realisations_diurnal")
os.makedirs(OUT_DIR, exist_ok=True)

# ── fixed forcing climatology (shared across realisations) ────────────────────
# Mode centres on the 3x3 blob grid (from instantiate_model.positions[:N]).
x_center = np.array([(p[2] + p[3]) / 2 for p in positions[:N]])   # "longitude"
y_center = np.array([(p[0] + p[1]) / 2 for p in positions[:N]])   # "latitude"

# Diurnal phase from longitude: full grid width spans one 24-h solar sweep.
# Eastern modes (larger x) lead.
phi_d = -2 * np.pi * x_center / nx                    # (N,)

# Annual phase from latitude: hemispheric flip about the grid midline.
phi_a = np.where(y_center > ny / 2, np.pi, 0.0)       # (N,)

# Amplitudes fixed once (not per-seed) so the climatology is identical everywhere.
_clim_rng = np.random.default_rng(12345)
# Amplitudes chosen so post-VAR variance is balanced annual≈dynamics, diurnal≈10%.
# Annual sits at ω≈0 and is amplified ~3.5x by the low-frequency AR gain (1/(1-a)),
# so its input amplitude is kept small; diurnal (ω=π/2) is slightly attenuated, so
# its input amplitude is raised. Realistic ordering annual > dynamics > diurnal kept.
A_d = _clim_rng.uniform(0.75, 1.40, size=N)           # diurnal amplitude per mode
A_a = _clim_rng.uniform(0.35, 0.70, size=N)           # annual amplitude per mode

# ── spectral radius for metadata ──────────────────────────────────────────────
from savar.functions import create_graph
_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

# ── precompute Phi(tau) @ W_flat for O(N·L) VAR loop ──────────────────────────
phi_mat = G
tau_max = phi_mat.shape[2]
phi_W   = [phi_mat[:, :, i] @ W_flat for i in range(tau_max)]

# ── deterministic latent forcing (same every realisation) ─────────────────────
total_T = T + burn
t_idx   = np.arange(total_T)
s_diurnal = A_d[:, None] * np.sin(2 * np.pi * t_idx / P_D + phi_d[:, None])   # (N, total_T)
s_annual  = A_a[:, None] * np.sin(2 * np.pi * t_idx / P_A + phi_a[:, None])   # (N, total_T)
s_latent  = s_diurnal + s_annual                                             # (N, total_T)
forcing_field = W_plus @ s_latent                                           # (L, total_T)

# Afternoon-peaked variance envelope, per mode (phased by longitude, +3 h lag).
conv_env = 1.0 + HET_AMP * np.clip(
    np.sin(2 * np.pi * t_idx / P_D + phi_d[:, None] - np.pi / 2), 0, None)   # (N, total_T)


def generate_obs(noise_field: np.ndarray) -> np.ndarray:
    data = (noise_field + forcing_field).copy()        # forcing enters before recurrence
    for t in range(tau_max, total_T):
        for i in range(tau_max):
            data[:, t] += W_plus @ (phi_W[i] @ data[:, t - 1 - i])
    return data[:, burn:]


print(f"\nGenerating {N_REALISATIONS} realisations  [diurnal+annual, D_y={DY_SCALE}*I_L]")
print(f"  Grid: {ny}x{nx}  L={L}  N={N}  T={T}  burn={burn}")
print(f"  Cadence: dt={DT_HOURS}h  P_diurnal={P_D} steps  P_annual={P_A} steps")
print(f"  Spectral radius: {spectral_radius:.4f}")
print(f"  Diurnal phase (longitude): {np.round(phi_d, 2)}")
print(f"  Annual  phase (latitude):  {np.round(phi_a, 2)}")
print(f"  eps_y std: {EPS_Y_STD:.4f}   het_amp: {HET_AMP}")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)

    # Afternoon-heteroskedastic per-mode latent innovations.
    eps_x = rng.standard_normal((N, total_T)) * conv_env
    eps_y = EPS_Y_STD * rng.standard_normal((L, total_T))
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
        forcing_latent     = s_latent[:, burn:].astype(np.float32),   # (N, T) recoverable forcing
        diurnal_amp        = A_d.astype(np.float32),
        diurnal_phase      = phi_d.astype(np.float32),
        annual_amp         = A_a.astype(np.float32),
        annual_phase       = phi_a.astype(np.float32),
        cycle_meta         = np.array([DT_HOURS, P_D, P_A, HET_AMP], dtype=np.float32),
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
    )

    if (seed + 1) % 10 == 0:
        elapsed = time.time() - t_start
        rate    = (seed + 1) / elapsed
        eta     = (N_REALISATIONS - seed - 1) / rate
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s")

# ── verification ──────────────────────────────────────────────────────────────
d   = np.load(os.path.join(OUT_DIR, f"realisation_{N_REALISATIONS-1:03d}.npz"))
obs = d["observations"]
Z   = d["latent_states"]
print(f"\nVerification (realisation {N_REALISATIONS-1}):")
print(f"  obs  shape={obs.shape}  mean={obs.mean():.4f}  std={obs.std():.4f}")
print(f"  Z    shape={Z.shape}    mean={Z.mean():.4f}  std={Z.std():.4f}")
print(f"  forcing_latent shape={d['forcing_latent'].shape}")
# Diurnal power check: variance explained by the 4-step cycle in mode 0.
z0 = Z[0]
ph = 2 * np.pi * np.arange(Z.shape[1]) / P_D
amp = 2 * np.abs(np.mean(z0 * np.exp(-1j * ph)))
print(f"  mode0 recovered diurnal amplitude ~ {amp:.3f}  (input A_d[0]={d['diurnal_amp'][0]:.3f})")
