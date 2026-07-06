"""
Generate SAVAR realisations with HETEROGENEOUS PER-MODE SELF-DYNAMICS (a variant
of generate_finecadence.py for the mode-specialization experiment).

Why this variant
----------------
In generate_finecadence.py the diagonal AR self-loops are inherited verbatim from
instantiate_model.py and sit in a narrow band φ ∈ [0.30, 0.55] — a <2× spread of
decorrelation timescales.  A 1-step forecaster's optimal move is then the SAME
scalar shrink (ŷ ≈ φ·current) for every mode, so no mode-specific computation pays
for itself: the GNN learns one global "shrink-to-mean" operator, the SAE finds a
single dominant global-activity direction, and VPD components fire on all modes
equally (see notes/vpd_results.md, project_savar_sae_findings).

This generator WIDENS the self-loop band to a real timescale spread (default
φ ∈ [0.15, 0.92], ~18× in τ = −1/ln φ).  At a fixed cadence, fast modes have
decorrelated (optimal: shrink hard toward the mean) while slow modes still persist
(optimal: near-copy).  The forecaster must apply a DIFFERENT gain per mode → the
substrate for mode-specialized features.  Everything else (edge set, cross-edge
fine lags, nonlinearity, non-Gaussian innovations, noise level) is kept identical
to generate_finecadence.py so this is a controlled one-factor change.

Knob: HD_PHI = comma-separated 8 self-loop coefficients (env-overridable).

Inherited design (unchanged from generate_finecadence.py)
---------------------------------------------------------
1. NONLINEARITY (note §2)   — saturating AR + bilinear advective coupling.
2. NON-GAUSSIANITY (note §5) — skewed/heavy-tailed innovations, unit variance.
3. FINE CADENCE + HETEROGENEOUS CROSS-LAGS — each cross-edge keeps its fine lag ℓ;
   subsampling at stride s aliases couplings with ℓ < s into τ=0 (PCMCI+ regime).

The edge SET and cross-edge lags are identical to generate_finecadence.py; ONLY
the diagonal self-loop coefficients are widened.  Innovations are standardized to
unit variance so the VAR stationarity condition is preserved.

Knobs (env-overridable):
  NL_ALPHA, NL_BETA          nonlinearity strengths   (default 0.5, 0.15)
  NG_DIST                    {skewnorm, t, gaussian}  (default skewnorm)
  NG_SKEW                    skew-normal shape alpha   (default 4.0)
  NG_DF                      Student-t dof             (default 5.0)
  N_REALISATIONS, NL_T       (default 100, 2400 fine steps)

Output: data/realisations_finecadence/
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

# Pull in grid / modes / weights (N, ny, nx, L, W_flat, W_plus, positions, ...).
# We deliberately OVERRIDE links_coeffs and G afterwards with the fine-lag graph.
exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

from savar.savar import dict_to_matrix
from savar.functions import check_stability, create_graph

# ── FINE-LAG ground-truth graph ───────────────────────────────────────────────
# Same edge set + cross-edge fine lags as generate_finecadence.py; ONLY the
# diagonal self-loop coefficients φ_j are widened to a real timescale spread.
# Self-loops stay at fine lag 1 (autoregression at the finest scale).
#   fast   (ℓ=1): X0→X1, X1→X2, X0→X3
#   med    (ℓ=2): X2→X3, X4→X5, X3→X6
#   med-sl (ℓ=3): X2→X0, X1→X4
#   slow   (ℓ=4): X0→X5, X6→X7
#   slow   (ℓ=6): X5→X6, X3→X7
#
# HD_PHI: 8 per-mode self-loop coefficients (decorrelation timescale τ=−1/ln φ).
# Default spans φ∈[0.15,0.92] → τ∈[0.53, 8.9] fine steps (~17× spread) vs the
# finecadence band [0.30,0.55] (τ∈[0.83,1.67], <2×).
_PHI_DEFAULT = "0.15,0.30,0.42,0.55,0.68,0.78,0.86,0.92"
PHI = [float(x) for x in os.environ.get("HD_PHI", _PHI_DEFAULT).split(",")]
assert len(PHI) == 8, f"HD_PHI needs 8 values, got {len(PHI)}"

# Per-mode innovation scaling to control the amplitude confound (slow modes
# otherwise louder because AR variance grows with φ).  Two ways to set it:
#   HD_INNOV_SCALE="s0,...,s7"  explicit per-mode scales (EMPIRICAL calibration:
#                               set s_j ∝ 1/std_j measured from an unscaled run,
#                               since mode std ≈ linear in innovation scale).
#   HD_EQUAL_VAR=1              analytic √(1-φ²) guess — NOTE this OVERSHOOTS here
#                               because the tanh saturation compresses slow/large
#                               modes, so prefer the empirical HD_INNOV_SCALE.
# Explicit HD_INNOV_SCALE wins.  Scales renormalised to mean 1 (amplitude preserved).
EQUAL_VAR = os.environ.get("HD_EQUAL_VAR", "0") == "1"
if os.environ.get("HD_INNOV_SCALE"):
    _raw_scale = np.array([float(x) for x in os.environ["HD_INNOV_SCALE"].split(",")])
    assert len(_raw_scale) == 8, "HD_INNOV_SCALE needs 8 values"
    EQUAL_VAR = True   # treat as a variance-controlled run for the output suffix
elif EQUAL_VAR:
    _raw_scale = np.sqrt(np.clip(1.0 - np.array(PHI) ** 2, 1e-6, None))
else:
    _raw_scale = np.ones(8)
INNOV_SCALE = _raw_scale / _raw_scale.mean()

links_coeffs = {
    0: [((0, -1),  PHI[0]), ((2, -3),  0.22)],                     # auto + X2→X0 (ℓ3)
    1: [((1, -1),  PHI[1]), ((0, -1),  0.35)],                     # auto + X0→X1 (ℓ1)
    2: [((2, -1),  PHI[2]), ((1, -1),  0.40)],                     # auto + X1→X2 (ℓ1)
    3: [((3, -1),  PHI[3]), ((0, -1),  0.30), ((2, -2), -0.30)],   # X0→X3 (ℓ1), X2→X3 (ℓ2)
    4: [((4, -1),  PHI[4]), ((1, -3),  0.25)],                     # auto + X1→X4 (ℓ3)
    5: [((5, -1),  PHI[5]), ((4, -2),  0.35), ((0, -4), -0.20)],   # X4→X5 (ℓ2), X0→X5 (ℓ4)
    6: [((6, -1),  PHI[6]), ((3, -2),  0.30), ((5, -6),  0.25)],   # X3→X6 (ℓ2), X5→X6 (ℓ6)
    7: [((7, -1),  PHI[7]), ((6, -4),  0.20), ((3, -6), -0.15)],   # X6→X7 (ℓ4), X3→X7 (ℓ6)
}

check_stability(links_coeffs)                 # raises if the fine VAR is non-stationary
G = dict_to_matrix(links_coeffs)              # (N, N, tau_max_fine);  G[j, i, ℓ-1] = coeff i→j at lag ℓ
tau_max = G.shape[2]                           # = 6 (max fine lag)

# Per-edge fine-lag table (cause, eff, fine_lag, coeff) for the cross edges only —
# saved to the npz so the subsample script knows each edge's intrinsic timescale.
fine_edges = []
for eff in range(N):
    for cause in range(N):
        if cause == eff:
            continue
        for lag in range(1, tau_max + 1):
            c = G[eff, cause, lag - 1]
            if c != 0:
                fine_edges.append((cause, eff, lag, float(c)))
fine_edges_arr = np.array([[c, e, l, v] for (c, e, l, v) in fine_edges], dtype=np.float32)

# ── noise level (GraphCast-like low noise) ────────────────────────────────────
# D_y = DY_SCALE · I_L observation noise. Env-overridable so the same generator
# produces a noise sweep; each level writes to its own suffixed output dir.
DY_SCALE  = float(os.environ.get("DY_SCALE", 0.05))
EPS_Y_STD = DY_SCALE ** 0.5

# ── lengths ───────────────────────────────────────────────────────────────────
T    = int(os.environ.get("NL_T", 2400))      # usable FINE steps
burn = 300

# ── nonlinearity knobs ────────────────────────────────────────────────────────
NL_ALPHA = float(os.environ.get("NL_ALPHA", 0.5))
NL_BETA  = float(os.environ.get("NL_BETA",  0.15))

# ── non-Gaussian innovation knobs ─────────────────────────────────────────────
NG_DIST = os.environ.get("NG_DIST", "skewnorm").lower()
NG_SKEW = float(os.environ.get("NG_SKEW", 4.0))    # skew-normal shape alpha
NG_DF   = float(os.environ.get("NG_DF",   5.0))    # Student-t dof

N_REALISATIONS = int(os.environ.get("N_REALISATIONS", 100))
# default (DY_SCALE=0.05) keeps the canonical dir; other levels get a suffix so a
# sweep never clobbers the existing dataset.
_SUFFIX = "" if abs(DY_SCALE - 0.05) < 1e-9 else f"_dy{DY_SCALE:g}".replace(".", "p")
if EQUAL_VAR:
    _SUFFIX += "_eqvar"
OUT_DIR = os.environ.get("FC_OUT_DIR", os.path.join("data", f"realisations_hetdynamics{_SUFFIX}"))
os.makedirs(OUT_DIR, exist_ok=True)

# ── spectral radius of the (linear) fine skeleton — for metadata / stability ──
_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

# ── lag-ℓ cross-edge list for the bilinear coupling (cause i, eff j, lag, coeff)
cross_edges = [(j, i, lag, G[j, i, lag - 1])
               for j in range(N) for i in range(N) for lag in range(1, tau_max + 1)
               if i != j and G[j, i, lag - 1] != 0]


def _g_sat(m):
    """Saturating nonlinearity; identity when NL_ALPHA == 0."""
    return (1.0 - NL_ALPHA) * m + NL_ALPHA * np.tanh(m)


def _bilinear(data, t):
    """Bounded advective coupling: product of the lagged cause and lag-1 effect."""
    if NL_BETA == 0.0 or not cross_edges:
        return np.zeros(N)
    q = np.zeros(N)
    m1 = W_flat @ data[:, t - 1]                       # lag-1 mode state (effect side)
    for j, i, lag, c in cross_edges:
        m_cause = W_flat @ data[:, t - lag]            # cause at its own fine lag
        q[j] += c * m_cause[i] * m1[j]
    return NL_BETA * np.tanh(q)


def draw_innovations(rng, shape):
    """Zero-mean, unit-variance non-Gaussian innovations (variance preserved so
    the VAR stationarity condition is untouched)."""
    if NG_DIST == "gaussian":
        return rng.standard_normal(shape)
    if NG_DIST == "t":
        # Student-t standardized to unit variance (heavy-tailed, symmetric).
        x = rng.standard_t(NG_DF, size=shape)
        return x / np.sqrt(NG_DF / (NG_DF - 2.0))
    if NG_DIST == "skewnorm":
        # Skew-normal via the delta construction, then standardize.
        a = NG_SKEW
        delta = a / np.sqrt(1.0 + a * a)
        z0 = np.abs(rng.standard_normal(shape))
        z1 = rng.standard_normal(shape)
        x = delta * z0 + np.sqrt(1.0 - delta * delta) * z1
        mean = delta * np.sqrt(2.0 / np.pi)
        var  = 1.0 - 2.0 * delta * delta / np.pi
        return (x - mean) / np.sqrt(var)
    raise ValueError(f"unknown NG_DIST={NG_DIST!r}")


def generate_obs(noise_field):
    total_T = noise_field.shape[1]
    data = noise_field.copy()
    for t in range(tau_max, total_T):
        contrib = np.zeros(N)
        for lag in range(1, tau_max + 1):
            m_lag = W_flat @ data[:, t - lag]
            contrib += G[:, :, lag - 1] @ _g_sat(m_lag)
        contrib += _bilinear(data, t)
        data[:, t] += W_plus @ contrib
    return data[:, burn:]


total_T = T + burn
print(f"\nGenerating {N_REALISATIONS} realisations  [FINE cadence, het. lags, nonlinear, {NG_DIST}]")
print(f"  Grid: {ny}x{nx}  L={L}  N={N}  T_fine={T}  burn={burn}")
print(f"  Fine lags present: {sorted(set(int(l) for _,_,l,_ in fine_edges))}  (tau_max_fine={tau_max})")
_tau = [(-1.0/np.log(p) if 0 < p < 1 else float('inf')) for p in PHI]
print(f"  Self-loops φ: {[round(p,2) for p in PHI]}")
print(f"  Timescales τ=−1/lnφ: {[round(t,2) for t in _tau]}  (spread {max(_tau)/min(_tau):.1f}×)")
print(f"  Equal-variance mode: {EQUAL_VAR}"
      + (f"  innov scales={[round(s,2) for s in INNOV_SCALE]}" if EQUAL_VAR else ""))
print(f"  Nonlinearity: NL_ALPHA(sat)={NL_ALPHA}  NL_BETA(bilinear)={NL_BETA}  cross-edges={len(cross_edges)}")
print(f"  Non-Gaussian: dist={NG_DIST}  skew_alpha={NG_SKEW}  t_dof={NG_DF}")
print(f"  Linear-skeleton spectral radius: {spectral_radius:.4f}  ({'STABLE' if spectral_radius<1 else 'UNSTABLE'})")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()
max_abs_global = 0.0
# sample skewness of the realised innovations (sanity check of non-Gaussianity)
skew_check = None

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)
    eps_x = draw_innovations(rng, (N, total_T))
    eps_x *= INNOV_SCALE[:, None]                       # per-mode variance control (no-op unless HD_EQUAL_VAR=1)
    eps_y = EPS_Y_STD * rng.standard_normal((L, total_T))
    if seed == 0:
        m = eps_x.mean(); s = eps_x.std()
        skew_check = float((((eps_x - m) / s) ** 3).mean())
    noise_field = W_plus @ eps_x + eps_y

    obs = generate_obs(noise_field)
    Z   = W_flat @ obs
    max_abs_global = max(max_abs_global, float(np.abs(Z).max()))

    np.savez_compressed(
        os.path.join(OUT_DIR, f"realisation_{seed:03d}.npz"),
        observations       = obs.astype(np.float32),
        latent_states      = Z.astype(np.float32),
        ground_truth_graph = G.astype(np.float32),
        fine_edges         = fine_edges_arr,            # (n_edges, 4): cause, eff, fine_lag, coeff
        W                  = W_flat.astype(np.float32),
        W_plus             = W_plus.astype(np.float32),
        nl_meta            = np.array([NL_ALPHA, NL_BETA], dtype=np.float32),
        ng_meta            = np.array([{"gaussian":0,"skewnorm":1,"t":2}[NG_DIST],
                                       NG_SKEW, NG_DF], dtype=np.float32),
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
    )

    if (seed + 1) % 10 == 0:
        elapsed = time.time() - t_start
        rate    = (seed + 1) / elapsed
        eta     = (N_REALISATIONS - seed - 1) / rate
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s")
print(f"  Innovation sample skewness (realisation 0): {skew_check:+.3f}  "
      f"(0 ⇒ symmetric/Gaussian)")
print(f"  Global max |Z|: {max_abs_global:.3f}  "
      f"({'STABLE' if np.isfinite(max_abs_global) and max_abs_global < 1e3 else 'UNSTABLE — reduce NL_BETA'})")

# ── verification ──────────────────────────────────────────────────────────────
d   = np.load(os.path.join(OUT_DIR, f"realisation_{N_REALISATIONS-1:03d}.npz"))
obs = d["observations"]; Z = d["latent_states"]
print(f"\nVerification (realisation {N_REALISATIONS-1}):")
print(f"  obs shape={obs.shape}  mean={obs.mean():.4f}  std={obs.std():.4f}")
print(f"  Z   shape={Z.shape}    mean={Z.mean():.4f}  std={Z.std():.4f}")
print(f"  fine_edges: {len(d['fine_edges'])} cross edges saved")
