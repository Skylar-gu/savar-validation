"""
Generate SAVAR realisations with ALL stress knobs on simultaneously:

  1. DIURNAL + ANNUAL FORCING (from generate_diurnal.py / generate_nonlinear.py)
     6 h cadence, T = 2920 (2 yr), deterministic per-mode sin forcing at the
     diurnal (4-step) and annual (1461-step) periods, plus afternoon
     heteroskedasticity on the latent innovations (conv_env).
  2. NONLINEAR LATENT DYNAMICS (from generate_nonlinear.py)
     Saturating autoregression g(m) = (1-NL_ALPHA) m + NL_ALPHA tanh(m) and
     bounded bilinear (advective) coupling along the lag-1 cross edges,
     NL_BETA * tanh(sum c * m_i * m_j). Same edge set as every other variant.
  3. NON-GAUSSIAN INNOVATIONS (from savar-project generate_hetdynamics.py)
     Latent innovations eps_x are skew-normal (delta construction, shape
     NG_SKEW, standardized to zero mean / unit variance so VAR stability is
     untouched). Observation noise eps_y stays Gaussian, D_y = 0.05 * I_L.
  4. MODE MOVEMENT (the movmech MOVE=place knob, coarse-grid version)
     Mechanism identity (self-coeff, parents, forcing amplitudes A_d/A_a) is
     fixed, but each mechanism's Gaussian-blob CENTRE is drawn freshly per
     realisation: uniform with margin, rejection-sampled to pairwise centre
     distance >= MIN_SEP, so truncated footprints stay pairwise DISJOINT
     (Jaccard 0). W and W_plus are therefore PER-REALISATION. Forcing phases
     follow the realised centres (phi_d from x — "local solar time", phi_a
     from hemisphere y), as in the fixed-position variants where phases were
     computed from the slot centres.

Blob geometry mirrors generate_movmech.py's parent blobs scaled to the coarse
grid: sigma 2.5, truncation radius 5 (11x11 support), MIN_SEP = 2*R+2 = 12,
MARGIN = R+1 = 6. Footprints are smaller than the fixed 16x16 slot blobs, a
necessary consequence of random disjoint placement on a 50x50 grid.

Everything else matches generate_nonlinear.py: same links_coeffs / G (edge
set), NL_ALPHA = 0.5, NL_BETA = 0.15, D_y = 0.05, burn = 200.

Knobs (env-overridable): NL_ALPHA, NL_BETA, NG_SKEW, N_REALISATIONS, NL_T
Output: data/realisations_allknobs/
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
P_A = int(round(365.25 * 24 / DT_HOURS))          # annual period  = 1461 steps
HET_AMP = 0.6

# ── nonlinearity knobs ───────────────────────────────────────────────────────
NL_ALPHA = float(os.environ.get("NL_ALPHA", 0.5))
NL_BETA  = float(os.environ.get("NL_BETA",  0.15))

# ── non-Gaussianity knob ─────────────────────────────────────────────────────
NG_SKEW = float(os.environ.get("NG_SKEW", 4.0))   # skew-normal shape alpha

# ── mode-movement geometry (coarse-grid movmech MOVE=place) ──────────────────
BLOB_SIGMA = 2.5
BLOB_R     = 5                                    # truncation radius (11x11 support)
MIN_SEP    = 2 * BLOB_R + 2                       # disjoint + 2 px gap
MARGIN     = BLOB_R + 1

N_REALISATIONS = int(os.environ.get("N_REALISATIONS", 100))
OUT_DIR = os.path.join("data", "realisations_allknobs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── fixed mechanism-identity forcing amplitudes (shared across realisations) ──
_clim_rng = np.random.default_rng(12345)
A_d = _clim_rng.uniform(0.75, 1.40, size=N)
A_a = _clim_rng.uniform(0.35, 0.70, size=N)

# ── spectral radius of the LINEAR skeleton (metadata) ────────────────────────
from savar.functions import create_graph
_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

# ── causal-graph tensors ─────────────────────────────────────────────────────
tau_max = G.shape[2]
G1 = G[:, :, 0]
cross_edges = [(j, i, G1[j, i]) for j in range(N) for i in range(N)
               if i != j and abs(G1[j, i]) > 0]

total_T = T + burn
t_idx   = np.arange(total_T)

YY, XX = np.mgrid[0:ny, 0:nx]


def draw_innovations(rng, shape, a=NG_SKEW):
    """Zero-mean, unit-variance skew-normal innovations (delta construction)."""
    delta = a / np.sqrt(1.0 + a * a)
    z0 = np.abs(rng.standard_normal(shape))
    z1 = rng.standard_normal(shape)
    x = delta * z0 + np.sqrt(1.0 - delta * delta) * z1
    mean = delta * np.sqrt(2.0 / np.pi)
    var  = 1.0 - 2.0 * delta * delta / np.pi
    return (x - mean) / np.sqrt(var)


def draw_centres(rng):
    """8 blob centres, uniform with margin, pairwise distance >= MIN_SEP."""
    for _ in range(2000):
        pts, tries = [], 0
        while len(pts) < N and tries < 500:
            p = rng.uniform(MARGIN, [ny - MARGIN, nx - MARGIN], size=2)
            if all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= MIN_SEP**2 for q in pts):
                pts.append(p)
            tries += 1
        if len(pts) == N:
            return np.array(pts)
    raise RuntimeError("centre rejection sampling failed")


def build_W(centres):
    """Per-realisation (N, L) weight matrix: truncated, L1-normalised Gaussians."""
    Wr = np.zeros((N, ny, nx))
    for j in range(N):
        cy, cx = centres[j]
        env = np.exp(-(((YY - cy) ** 2 + (XX - cx) ** 2) / (2 * BLOB_SIGMA ** 2)))
        env[(YY - cy) ** 2 + (XX - cx) ** 2 > BLOB_R ** 2] = 0.0
        Wr[j] = env / env.sum()
    return Wr.reshape(N, L)


def _g_sat(m):
    return (1.0 - NL_ALPHA) * m + NL_ALPHA * np.tanh(m)


def _bilinear(m1):
    if NL_BETA == 0.0 or not cross_edges:
        return np.zeros(N)
    q = np.zeros(N)
    for j, i, c in cross_edges:
        q[j] += c * m1[i] * m1[j]
    return NL_BETA * np.tanh(q)


def generate_obs(noise_field, W_flat_r, W_plus_r, forcing_field):
    data = (noise_field + forcing_field).copy()
    for t in range(tau_max, total_T):
        contrib = np.zeros(N)
        for i in range(tau_max):
            m_lag = W_flat_r @ data[:, t - 1 - i]
            contrib += G[:, :, i] @ _g_sat(m_lag)
        m1 = W_flat_r @ data[:, t - 1]
        contrib += _bilinear(m1)
        data[:, t] += W_plus_r @ contrib
    return data[:, burn:]


print(f"\nGenerating {N_REALISATIONS} realisations  "
      f"[ALL KNOBS: diurnal+annual, nonlinear, skew-normal, moving modes]")
print(f"  Grid: {ny}x{nx}  L={L}  N={N}  T={T}  burn={burn}")
print(f"  Cadence: dt={DT_HOURS}h  P_diurnal={P_D}  P_annual={P_A}  het_amp={HET_AMP}")
print(f"  Nonlinearity: NL_ALPHA={NL_ALPHA}  NL_BETA={NL_BETA}  cross-edges={len(cross_edges)}")
print(f"  Non-Gaussian: skew-normal alpha={NG_SKEW}")
print(f"  Mode movement: sigma={BLOB_SIGMA}  R={BLOB_R}  min_sep={MIN_SEP}  margin={MARGIN}")
print(f"  Linear-skeleton spectral radius: {spectral_radius:.4f}")
print(f"  eps_y std: {EPS_Y_STD:.4f}  (Gaussian)")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()
max_abs_global = 0.0
skew_check = None

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)

    # per-realisation geometry and forcing
    centres  = draw_centres(rng)
    W_flat_r = build_W(centres)
    W_plus_r = np.linalg.pinv(W_flat_r)

    # disjointness gate (Jaccard 0 by construction; verify anyway)
    supp = (W_flat_r > 0)
    overlap = int((supp.sum(0) > 1).sum())
    assert overlap == 0, f"seed {seed}: {overlap} cells shared between footprints"

    y_center = centres[:, 0]
    x_center = centres[:, 1]
    phi_d = -2 * np.pi * x_center / nx
    phi_a = np.where(y_center > ny / 2, np.pi, 0.0)

    s_diurnal = A_d[:, None] * np.sin(2 * np.pi * t_idx / P_D + phi_d[:, None])
    s_annual  = A_a[:, None] * np.sin(2 * np.pi * t_idx / P_A + phi_a[:, None])
    s_latent  = s_diurnal + s_annual
    forcing_field = W_plus_r @ s_latent

    conv_env = 1.0 + HET_AMP * np.clip(
        np.sin(2 * np.pi * t_idx / P_D + phi_d[:, None] - np.pi / 2), 0, None)

    eps_x = draw_innovations(rng, (N, total_T)) * conv_env
    eps_y = EPS_Y_STD * rng.standard_normal((L, total_T))
    noise_field = W_plus_r @ eps_x + eps_y

    if skew_check is None:
        m, s = eps_x.mean(), eps_x.std()
        skew_check = float((((eps_x - m) / s) ** 3).mean())

    obs = generate_obs(noise_field, W_flat_r, W_plus_r, forcing_field)
    assert np.isfinite(obs).all(), f"seed {seed}: non-finite observations"
    Z   = W_flat_r @ obs
    max_abs_global = max(max_abs_global, float(np.abs(Z).max()))

    np.savez_compressed(
        os.path.join(OUT_DIR, f"realisation_{seed:03d}.npz"),
        observations       = obs.astype(np.float32),
        latent_states      = Z.astype(np.float32),
        ground_truth_graph = G.astype(np.float32),
        W                  = W_flat_r.astype(np.float32),
        W_plus             = W_plus_r.astype(np.float32),
        centres            = centres.astype(np.float32),
        forcing_latent     = s_latent[:, burn:].astype(np.float32),
        diurnal_amp        = A_d.astype(np.float32),
        diurnal_phase      = phi_d.astype(np.float32),
        annual_amp         = A_a.astype(np.float32),
        annual_phase       = phi_a.astype(np.float32),
        cycle_meta         = np.array([DT_HOURS, P_D, P_A, HET_AMP], dtype=np.float32),
        nl_meta            = np.array([NL_ALPHA, NL_BETA], dtype=np.float32),
        ng_meta            = np.array([1, NG_SKEW], dtype=np.float32),  # 1 = skewnorm
        move_meta          = np.array([BLOB_SIGMA, BLOB_R, MIN_SEP, MARGIN], dtype=np.float32),
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
    )

    if (seed + 1) % 10 == 0 or seed == 0:
        elapsed = time.time() - t_start
        rate    = (seed + 1) / elapsed
        eta     = (N_REALISATIONS - seed - 1) / rate
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s")
print(f"  Innovation sample skewness (realisation 0): {skew_check:+.3f}  "
      f"(skew-normal alpha=4 target ~ +0.78)")
print(f"  Global max |Z| across realisations: {max_abs_global:.3f}  "
      f"({'STABLE' if np.isfinite(max_abs_global) and max_abs_global < 1e3 else 'UNSTABLE — reduce NL_BETA'})")

# ── verification ─────────────────────────────────────────────────────────────
d   = np.load(os.path.join(OUT_DIR, f"realisation_{N_REALISATIONS-1:03d}.npz"))
obs = d["observations"]
Z   = d["latent_states"]
print(f"\nVerification (realisation {N_REALISATIONS-1}):")
print(f"  obs  shape={obs.shape}  mean={obs.mean():.4f}  std={obs.std():.4f}")
print(f"  Z    shape={Z.shape}    mean={Z.mean():.4f}  std={Z.std():.4f}")
print(f"  centres:\n{np.round(d['centres'], 1)}")
