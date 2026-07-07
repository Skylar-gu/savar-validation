"""
R6 "atmosphere regime" rung (litext plan §3 E5/R6) — fork of
generate_hetdynamics.py pushing the self-dynamics into GraphCast's one-step
statistical regime: near-unit memory, one-step R^2 ~ 0.9+.

Changes vs generate_hetdynamics.py (everything else inherited verbatim):
  1. phi band 0.90–0.99: 8 modes log-spaced in tau = -1/ln(phi) from ~9.5 to
     ~100 steps (ratio ~10.5x, compressed from het's 17x per the plan's
     under-sampling argument).
  2. Cross-edge SET AND LAGS identical to hetdynamics; coefficients globally
     rescaled by ATMO_CROSS_SCALE (default 0.20) — required for stationarity:
     the 0->1->2->0 cycle with near-integrator diagonals blows up at full
     strength (companion spectral radius 1.203 at scale 1.0; 0.9911 at 0.20;
     the critical scale for radius<=0.995 is 0.2135). Radius is verified and
     printed; assert < 1.
  3. T = 9600 usable steps (slow mode keeps ~96 e-folds per realisation),
     burn = 900 (9 e-folds of the slowest mode), 40 realisations.
  4. DY_SCALE default 0.0125 (4x below het's 0.05) — high-SNR regime.
  5. Equal stationary variance per mode via ABSOLUTE per-mode innovation
     scales (HD_INNOV_SCALE, no mean-1 renormalisation unlike the parent):
     calibrate empirically with a pilot so every mode sits at std ~1.23 —
     the parent eqvar amplitude — keeping the tanh-saturation regime matched.

Innovations stay exogenous / per-mode / skew-normal (NG_DIST inherited).

Env knobs: ATMO_CROSS_SCALE (0.20), HD_PHI (atmo band), HD_INNOV_SCALE
(absolute scales), NL_T (9600), DY_SCALE (0.0125), N_REALISATIONS (40),
FC_OUT_DIR (data/realisations_atmoregime), NL_ALPHA/NL_BETA/NG_* as parent.
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

# Grid / modes / weights (N, ny, nx, L, W_flat, W_plus, positions, ...) —
# same blob layout as every other rung.
exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

from savar.savar import dict_to_matrix
from savar.functions import check_stability, create_graph

# ── atmosphere-regime phi band: tau log-spaced 9.5 -> 100 steps ───────────────
_TAU = 9.5 * (100.0 / 9.5) ** (np.arange(8) / 7.0)
_PHI_DEFAULT = ",".join(f"{np.exp(-1.0/t):.6f}" for t in _TAU)
PHI = [float(x) for x in os.environ.get("HD_PHI", _PHI_DEFAULT).split(",")]
assert len(PHI) == 8

# v2: DC-GAIN-MATCHED cross coefficients. A global scale is the wrong knob in
# the near-integrator regime: each edge's integrated (quasi-static) gain is
# c/(1-phi_effect), so with phi -> 0.99 even c=0.05 gives DC gain 5-20 and
# chained modes random-walk (v2 pilot: X7 std 8, ACF 0.9996). Instead match
# each edge's DC gain to the PARENT rung's: c_new = c_parent *
# (1-phi_new[eff])/(1-phi_parent[eff]). Keeps integrated teleconnection
# strength, sign, lag and the cycle's DC loop gain (stability) at parent
# levels — the "weak instantaneous, strong integrated coupling" structure of
# real slow climate modes. ATMO_CROSS_SCALE remains as an extra multiplier.
CROSS_SCALE = float(os.environ.get("ATMO_CROSS_SCALE", 1.0))
PHI_PARENT = [0.15, 0.30, 0.42, 0.55, 0.68, 0.78, 0.86, 0.92]
def _dc(eff):
    return CROSS_SCALE * (1.0 - PHI[eff]) / (1.0 - PHI_PARENT[eff])

# ABSOLUTE per-mode innovation scales (parent renormalises to mean 1; here the
# scales set the amplitude directly so eqvar calibration can target the parent
# rung's mode std ~1.23 and keep the tanh regime matched).
if os.environ.get("HD_INNOV_SCALE"):
    INNOV_SCALE = np.array([float(x) for x in os.environ["HD_INNOV_SCALE"].split(",")])
    assert len(INNOV_SCALE) == 8
else:
    INNOV_SCALE = np.ones(8)

# ── ground-truth graph: SAME edge set + lags as hetdynamics, DC-matched coeffs ──
links_coeffs = {
    0: [((0, -1),  PHI[0]), ((2, -3),  0.22*_dc(0))],
    1: [((1, -1),  PHI[1]), ((0, -1),  0.35*_dc(1))],
    2: [((2, -1),  PHI[2]), ((1, -1),  0.40*_dc(2))],
    3: [((3, -1),  PHI[3]), ((0, -1),  0.30*_dc(3)), ((2, -2), -0.30*_dc(3))],
    4: [((4, -1),  PHI[4]), ((1, -3),  0.25*_dc(4))],
    5: [((5, -1),  PHI[5]), ((4, -2),  0.35*_dc(5)), ((0, -4), -0.20*_dc(5))],
    6: [((6, -1),  PHI[6]), ((3, -2),  0.30*_dc(6)), ((5, -6),  0.25*_dc(6))],
    7: [((7, -1),  PHI[7]), ((6, -4),  0.20*_dc(7)), ((3, -6), -0.15*_dc(7))],
}

check_stability(links_coeffs)
G = dict_to_matrix(links_coeffs)
tau_max = G.shape[2]

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

# ── noise / lengths ───────────────────────────────────────────────────────────
DY_SCALE  = float(os.environ.get("DY_SCALE", 0.0125))
EPS_Y_STD = DY_SCALE ** 0.5
T    = int(os.environ.get("NL_T", 9600))
burn = 900

NL_ALPHA = float(os.environ.get("NL_ALPHA", 0.5))
NL_BETA  = float(os.environ.get("NL_BETA",  0.15))
NG_DIST = os.environ.get("NG_DIST", "skewnorm").lower()
NG_SKEW = float(os.environ.get("NG_SKEW", 4.0))
NG_DF   = float(os.environ.get("NG_DF",   5.0))

N_REALISATIONS = int(os.environ.get("N_REALISATIONS", 40))
OUT_DIR = os.environ.get("FC_OUT_DIR", os.path.join("data", "realisations_atmoregime"))
os.makedirs(OUT_DIR, exist_ok=True)

# ── spectral radius of the linear fine skeleton (MUST be < 1) ─────────────────
_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))
assert spectral_radius < 1.0, f"unstable linear skeleton: radius={spectral_radius:.4f}"

cross_edges = [(j, i, lag, G[j, i, lag - 1])
               for j in range(N) for i in range(N) for lag in range(1, tau_max + 1)
               if i != j and G[j, i, lag - 1] != 0]


def _g_sat(m):
    return (1.0 - NL_ALPHA) * m + NL_ALPHA * np.tanh(m)


# v2 (2026-07-07): SELF-LOOPS LINEAR, cross-terms saturated. v1 applied the
# parent's _g_sat to the full lagged state INCLUDING the phi self-loop; at
# operating amplitude ~1.23 the saturation derivative (0.5 + 0.5*sech^2(1.2)
# ~ 0.66) caps the SMALL-SIGNAL memory at tau_eff = -1/ln(0.66*phi) ~ 3 steps
# for EVERY phi — realized ACF(1) was 0.69-0.77 (spread 1.36x) instead of the
# designed 0.90-0.99 (10.5x): the rung's defining regime was destroyed (the
# same mechanism compresses the parent rung's phi=0.92 to tau_eff~3.3).
# Diagonal linear => realized memory = phi exactly; the nonlinearity the
# ladder cares about (saturating CROSS-mode transfer + bilinear advection)
# is preserved.
G_DIAG = np.zeros_like(G)
G_CROSS = G.copy()
for _j in range(N):
    G_DIAG[_j, _j, 0] = G[_j, _j, 0]
    G_CROSS[_j, _j, 0] = 0.0


def _bilinear(data, t):
    if NL_BETA == 0.0 or not cross_edges:
        return np.zeros(N)
    q = np.zeros(N)
    m1 = W_flat @ data[:, t - 1]
    for j, i, lag, c in cross_edges:
        m_cause = W_flat @ data[:, t - lag]
        q[j] += c * m_cause[i] * m1[j]
    return NL_BETA * np.tanh(q)


def draw_innovations(rng, shape):
    if NG_DIST == "gaussian":
        return rng.standard_normal(shape)
    if NG_DIST == "t":
        x = rng.standard_t(NG_DF, size=shape)
        return x / np.sqrt(NG_DF / (NG_DF - 2.0))
    if NG_DIST == "skewnorm":
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
            contrib += G_DIAG[:, :, lag - 1] @ m_lag            # linear memory
            contrib += G_CROSS[:, :, lag - 1] @ _g_sat(m_lag)   # saturated transfer
        contrib += _bilinear(data, t)
        data[:, t] += W_plus @ contrib
    return data[:, burn:]


total_T = T + burn
_tau = [(-1.0/np.log(p) if 0 < p < 1 else float('inf')) for p in PHI]
print(f"\nGenerating {N_REALISATIONS} realisations  [ATMOREGIME: near-unit memory, {NG_DIST}]")
print(f"  Grid: {ny}x{nx}  L={L}  N={N}  T_fine={T}  burn={burn}")
print(f"  Self-loops phi: {[round(p,4) for p in PHI]}")
print(f"  Timescales tau: {[round(t,1) for t in _tau]}  (spread {max(_tau)/min(_tau):.1f}x)")
print(f"  Cross-coeff scale: {CROSS_SCALE}  (12 edges, same set+lags as hetdynamics)")
print(f"  Innov scales (ABSOLUTE): {[round(float(s),3) for s in INNOV_SCALE]}")
print(f"  DY_SCALE={DY_SCALE}  NL_ALPHA={NL_ALPHA}  NL_BETA={NL_BETA}")
print(f"  Linear-skeleton spectral radius: {spectral_radius:.4f}  (STABLE)")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()
max_abs_global = 0.0
all_stds = []

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)
    eps_x = draw_innovations(rng, (N, total_T))
    eps_x *= INNOV_SCALE[:, None]
    eps_y = EPS_Y_STD * rng.standard_normal((L, total_T))
    noise_field = W_plus @ eps_x + eps_y

    obs = generate_obs(noise_field)
    Z   = W_flat @ obs
    max_abs_global = max(max_abs_global, float(np.abs(Z).max()))
    all_stds.append(Z.std(1))

    np.savez_compressed(
        os.path.join(OUT_DIR, f"realisation_{seed:03d}.npz"),
        observations       = obs.astype(np.float32),
        latent_states      = Z.astype(np.float32),
        ground_truth_graph = G.astype(np.float32),
        fine_edges         = fine_edges_arr,
        W                  = W_flat.astype(np.float32),
        W_plus             = W_plus.astype(np.float32),
        nl_meta            = np.array([NL_ALPHA, NL_BETA], dtype=np.float32),
        ng_meta            = np.array([{"gaussian":0,"skewnorm":1,"t":2}[NG_DIST],
                                       NG_SKEW, NG_DF], dtype=np.float32),
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
    )
    if (seed + 1) % 5 == 0:
        elapsed = time.time() - t_start
        rate = (seed + 1) / elapsed
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  "
              f"ETA {(N_REALISATIONS-seed-1)/rate:.1f}s")

print(f"\nDone in {time.time()-t_start:.1f}s")
print(f"  Global max |Z|: {max_abs_global:.3f}")
print(f"  Per-mode Z std (mean over reals): {np.stack(all_stds).mean(0).round(3)}")
print(f"  (target ~1.23 across all modes = parent eqvar amplitude)")

# realized-memory verification gate (the v1 failure mode): ACF(1) must track phi
d_last = np.load(os.path.join(OUT_DIR, f"realisation_{N_REALISATIONS-1:03d}.npz"))
Zv = d_last["latent_states"].astype(np.float64)
ac1 = np.array([np.corrcoef(Zv[j, :-1], Zv[j, 1:])[0, 1] for j in range(N)])
print(f"  Realized ACF(1) per mode: {ac1.round(4)}")
print(f"  Designed phi:             {np.array(PHI).round(4)}")
print(f"  max |ACF(1) - phi| = {np.abs(ac1 - np.array(PHI)).max():.4f} "
      f"({'OK' if np.abs(ac1 - np.array(PHI)).max() < 0.05 else 'REGIME MISMATCH'})")
