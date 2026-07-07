"""
R1 OVERLAP rung generator (litext plan §3 E5, R1 row) — fork of
generate_hetdynamics.py (eqvar configuration).

DYNAMICS ARE BYTE-IDENTICAL TO hetdynamics_eqvar: same HD_PHI band
(0.15..0.92), same 12-edge set / fine lags / coefficients, same eqvar
empirical innovation scaling (0.97,...,0.69), same nonlinearity
(NL_ALPHA=0.5, NL_BETA=0.15), same skew-normal innovations, same T=2400,
100 realisations, same DY_SCALE=0.05, same per-realisation seeds. The known
saturated-self-loop caveat (g_sat applied to the φ self-loop ⇒ small-signal
τ_eff ≈ 3; see R6-v1 postmortem) applies equally to parent and R1 — this is
deliberate: comparability with the parent rung is the point.

ONLY THE EMISSION CHANGES: the aggregation map W. Instead of disjoint 16×16
blobs on the 3×3 lattice, each mode's footprint is an isotropic Gaussian at
the SAME lattice centre, with the width σ tuned numerically (bisection) so
that the MAX PAIRWISE W-ROW COSINE hits OV_COS (default 0.2 — the R1 primary;
OV_COS=0.5 gives the follow-up variant). Rows are truncated at OV_TRUNC of
their max (support stays finite so support-Jaccard / half-split machinery
stays meaningful) and L1-normalised (parent convention). W is FIXED across
realisations. The generator prints the full pairwise cosine matrix and the
support-Jaccard matrix.

Note on what stays invariant under the W change: W_flat @ W_plus = I_N
(pinv right-inverse, full row rank), so the mode-level recursion — and hence
Z = W_flat @ obs — follows the SAME VAR as the parent; the only leak is the
pixel-noise term W_flat @ eps_y (std EPS_Y_STD·||row||₂ ≈ 0.01 per mode,
negligible vs mode std 1.23, and now weakly correlated across modes ∝ row
cosines — part of what the R1 gates probe).

Env: OV_COS (0.2), OV_TRUNC (1e-3), N_REALISATIONS (100), NL_T (2400),
     OV_OUT_DIR (data/realisations_overlap<02|05|...>)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

# Pull in grid / modes / weights (N, ny, nx, L, positions, ...). W is REBUILT
# below with overlapping Gaussians; links_coeffs and G are overridden with the
# hetdynamics fine-lag graph.
exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

from savar.savar import dict_to_matrix
from savar.functions import check_stability, create_graph

# ── OVERLAPPING W (the one change vs the parent rung) ────────────────────────
OV_COS   = float(os.environ.get("OV_COS", 0.2))
OV_TRUNC = float(os.environ.get("OV_TRUNC", 1e-3))

# blob centres = centres of the SAME 3×3-lattice slots the parent uses
centres = []
for i in range(N):
    y1, y2, x1, x2 = positions[i]
    centres.append(((y1 + y2 - 1) / 2.0, (x1 + x2 - 1) / 2.0))
centres = np.array(centres)                    # (8, 2)

_yy, _xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

def build_overlap_W(sigma):
    Wn = np.zeros((N, ny, nx))
    for i in range(N):
        cy, cx = centres[i]
        g = np.exp(-((_yy - cy) ** 2 + (_xx - cx) ** 2) / (2.0 * sigma ** 2))
        g[g < OV_TRUNC * g.max()] = 0.0        # finite support
        Wn[i] = g / g.sum()                    # L1-normalise (parent convention)
    return Wn.reshape(N, L)

def max_pair_cos(Wf):
    Wu = Wf / np.linalg.norm(Wf, axis=1, keepdims=True)
    Cm = Wu @ Wu.T
    off = Cm[~np.eye(N, dtype=bool)]
    return float(off.max()), Cm

# bisection on sigma: max pairwise cosine is monotone-increasing in sigma
lo, hi = 1.0, 20.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    m, _ = max_pair_cos(build_overlap_W(mid))
    if m < OV_COS:
        lo = mid
    else:
        hi = mid
SIGMA_BLOB = 0.5 * (lo + hi)
W_flat = build_overlap_W(SIGMA_BLOB)
W = W_flat.reshape(N, ny, nx)
W_plus = np.linalg.pinv(W_flat)
MAXCOS, COSM = max_pair_cos(W_flat)

# support Jaccard + mass overlap (Block-G conventions)
supp = W_flat > 0
JAC = np.zeros((N, N)); MASS = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        inter = (supp[i] & supp[j]).sum(); union = (supp[i] | supp[j]).sum()
        JAC[i, j] = inter / union if union else 0.0
        MASS[i, j] = np.minimum(W_flat[i], W_flat[j]).sum()
_off = ~np.eye(N, dtype=bool)

print(f"\n[overlap W] target max pairwise cosine = {OV_COS}  "
      f"-> sigma = {SIGMA_BLOB:.3f} px  (truncation {OV_TRUNC} of row max)")
print(f"  achieved max cos = {MAXCOS:.4f}   mean off-diag cos = {COSM[_off].mean():.4f}")
print(f"  support Jaccard: max = {JAC[_off].max():.4f}  mean = {JAC[_off].mean():.4f}")
print(f"  mass overlap:    max = {MASS[_off].max():.4f}  mean = {MASS[_off].mean():.4f}")
print(f"  row L2 norms: {np.linalg.norm(W_flat, axis=1).round(4)}")
print(f"  ||W_plus|| column max-abs: {np.abs(W_plus).max():.3f} "
      f"(parent disjoint blobs ~ sharper)")
print("  pairwise cosine matrix:")
for i in range(N):
    print("   " + " ".join(f"{COSM[i, j]:5.2f}" for j in range(N)))
print("  support-Jaccard matrix:")
for i in range(N):
    print("   " + " ".join(f"{JAC[i, j]:5.2f}" for j in range(N)))

# ── FINE-LAG ground-truth graph — BYTE-IDENTICAL to generate_hetdynamics.py ──
_PHI_DEFAULT = "0.15,0.30,0.42,0.55,0.68,0.78,0.86,0.92"
PHI = [float(x) for x in os.environ.get("HD_PHI", _PHI_DEFAULT).split(",")]
assert len(PHI) == 8, f"HD_PHI needs 8 values, got {len(PHI)}"

# eqvar empirical innovation scaling — the hetdynamics_eqvar calibration
# (scratch/scratch_hetdyn_eqvar_pipeline.sh), hardcoded as the default here.
_EQVAR_SCALE_DEFAULT = "0.97,0.93,0.89,0.85,0.83,0.73,0.68,0.69"
_raw_scale = np.array([float(x) for x in os.environ.get(
    "HD_INNOV_SCALE", _EQVAR_SCALE_DEFAULT).split(",")])
assert len(_raw_scale) == 8
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

check_stability(links_coeffs)
G = dict_to_matrix(links_coeffs)
tau_max = G.shape[2]                           # = 6

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

# ── noise / lengths / nonlinearity — parent eqvar values ─────────────────────
DY_SCALE  = float(os.environ.get("DY_SCALE", 0.05))
EPS_Y_STD = DY_SCALE ** 0.5
T    = int(os.environ.get("NL_T", 2400))
burn = 300
NL_ALPHA = float(os.environ.get("NL_ALPHA", 0.5))
NL_BETA  = float(os.environ.get("NL_BETA",  0.15))
NG_DIST = os.environ.get("NG_DIST", "skewnorm").lower()
NG_SKEW = float(os.environ.get("NG_SKEW", 4.0))
NG_DF   = float(os.environ.get("NG_DF",   5.0))
N_REALISATIONS = int(os.environ.get("N_REALISATIONS", 100))

_SUFFIX = f"{OV_COS:g}".replace("0.", "0")     # 0.2 -> "02", 0.5 -> "05"
OUT_DIR = os.environ.get("OV_OUT_DIR", os.path.join("data", f"realisations_overlap{_SUFFIX}"))
os.makedirs(OUT_DIR, exist_ok=True)

_g   = create_graph(links_coeffs, return_lag=False)
_p   = _g.shape[2]
_top = np.hstack([_g[:, :, i] for i in range(_p)])
_bot = np.hstack([np.eye(N * (_p - 1)), np.zeros((N * (_p - 1), N))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))

cross_edges = [(j, i, lag, G[j, i, lag - 1])
               for j in range(N) for i in range(N) for lag in range(1, tau_max + 1)
               if i != j and G[j, i, lag - 1] != 0]


def _g_sat(m):
    return (1.0 - NL_ALPHA) * m + NL_ALPHA * np.tanh(m)


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
            contrib += G[:, :, lag - 1] @ _g_sat(m_lag)
        contrib += _bilinear(data, t)
        data[:, t] += W_plus @ contrib
    return data[:, burn:]


total_T = T + burn
_tau = [(-1.0/np.log(p) if 0 < p < 1 else float('inf')) for p in PHI]
print(f"\nGenerating {N_REALISATIONS} realisations  [R1 OVERLAP {OV_COS}, dynamics = hetdynamics_eqvar]")
print(f"  Grid: {ny}x{nx}  L={L}  N={N}  T_fine={T}  burn={burn}")
print(f"  Self-loops φ: {[round(p,2) for p in PHI]}  (τ spread {max(_tau)/min(_tau):.1f}×)")
print(f"  Innov scales (eqvar empirical): {[round(s,2) for s in INNOV_SCALE]}")
print(f"  NL_ALPHA={NL_ALPHA} NL_BETA={NL_BETA}  {NG_DIST}(α={NG_SKEW})  DY_SCALE={DY_SCALE}")
print(f"  Linear-skeleton spectral radius: {spectral_radius:.4f}  ({'STABLE' if spectral_radius<1 else 'UNSTABLE'})")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()
max_abs_global = 0.0

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)
    eps_x = draw_innovations(rng, (N, total_T))
    eps_x *= INNOV_SCALE[:, None]
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
        fine_edges         = fine_edges_arr,
        W                  = W_flat.astype(np.float32),
        W_plus             = W_plus.astype(np.float32),
        cos_matrix         = COSM.astype(np.float32),
        jaccard            = JAC.astype(np.float32),
        nl_meta            = np.array([NL_ALPHA, NL_BETA], dtype=np.float32),
        ng_meta            = np.array([{"gaussian":0,"skewnorm":1,"t":2}[NG_DIST],
                                       NG_SKEW, NG_DF], dtype=np.float32),
        ov_meta            = np.array([OV_COS, SIGMA_BLOB, MAXCOS, OV_TRUNC], dtype=np.float32),
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
    )

    if (seed + 1) % 10 == 0:
        elapsed = time.time() - t_start
        rate    = (seed + 1) / elapsed
        eta     = (N_REALISATIONS - seed - 1) / rate
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s")
print(f"  Global max |Z|: {max_abs_global:.3f}  "
      f"({'STABLE' if np.isfinite(max_abs_global) and max_abs_global < 1e3 else 'UNSTABLE'})")

# ── verification: dynamics-identity vs the parent rung ──────────────────────
d   = np.load(os.path.join(OUT_DIR, f"realisation_000.npz"))
Zo  = d["latent_states"].astype(np.float64)
print(f"\nVerification (realisation 0):")
print(f"  obs shape={d['observations'].shape}  Z std per mode: {Zo.std(1).round(3)}")
parent = os.path.join("data", "realisations_hetdynamics_eqvar", "realisation_000.npz")
if os.path.exists(parent):
    Zp = np.load(parent)["latent_states"].astype(np.float64)
    _Tm = min(Zp.shape[1], Zo.shape[1])
    cc = [float(np.corrcoef(Zo[j, :_Tm], Zp[j, :_Tm])[0, 1]) for j in range(N)]
    print(f"  corr(Z_overlap, Z_parent) per mode: {np.round(cc, 4)}")
    print(f"  (same seeds + same mode VAR; deviation = pixel-noise leak "
          f"W@eps_y through the new W only)")
