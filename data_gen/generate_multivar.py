"""
R3 MULTIVARIATE rung generator (litext plan §3 E5, R3 row) — fork of
generate_overlap.py.

WHAT CHANGES vs the parent rungs: each spatial node now emits **C observed
channels** (C = 2 or 3 coupled fields, e.g. "temperature / pressure / humidity"
co-located at the same footprint), and the coupling graph Φ gains **cross-channel
edges** (a mode in one channel drives a mode in another channel). This is the
"2–3 coupled observed channels per node, cross-channel edges in Φ" rung: Ŵ must
become (channel × space) and PCMCI+ conditioning must handle the cross-channel
edges.

DYNAMICS FAMILY IS UNCHANGED. The within-channel graph is the hetdynamics_eqvar
fine-lag graph (same HD_PHI band, same 12-edge set / fine lags / coefficients,
same eqvar innovation scaling, same g_sat / bilinear nonlinearity, same
skew-normal innovations). Each of the C channels carries an INDEPENDENT COPY of
that within-channel graph; the channels are then coupled ONLY through the small
cross-channel edge set (below). So with the cross-channel edges removed the rung
degenerates to C independent copies of the parent rung — the cross edges are the
one new thing the discovery/battery must cope with.

── CHANNEL / GRAPH LAYOUT (read this before touching the .npz) ───────────────
  N   spatial nodes (modes) per channel      (= 8, parent)
  C   observed channels per node             (env MV_C, default 2)
  NC  = N * C   total latent modes
  L   = ny * nx pixels per channel field

  STACKED MODE INDEX (used by ground_truth_graph, latent_states, fine_edges):
      m = channel * N + node        (channel-major)
      channel(m) = m // N ,  node(m) = m % N

  observations       (C, L, T)     one flattened 50×50 field PER CHANNEL
  latent_states      (NC, T)       Z[m] = W_flat[node(m)] @ observations[channel(m)]
  ground_truth_graph (NC, NC, tau) G_ext[effect, cause, lag-1]  (stacked index)
  fine_edges         rows (cause_m, effect_m, lag, coeff)   in STACKED index
  channel_edges      rows (cause_node, cause_ch, effect_node, effect_ch, lag,
                     coeff, is_cross)  — human-readable decode of every edge;
                     is_cross==1 iff cause_ch != effect_ch
  W                  (N, L)        spatial footprints, SHARED across channels
                     (the full aggregation map is Ŵ = I_C ⊗ W, block-diagonal
                     (channel × space) — that is exactly the "Ŵ must become
                     (channel × space)" change)
  W_plus             (L, N)        pinv right-inverse of W
  mv_meta            [N, C, L, NC, tau_max]

EMISSION. Footprints are the parent's DISJOINT Gaussian blobs (from
instantiate_model.py — NOT the overlap-Gaussian W). R3 isolates the multivariate
axis; spatial overlap is the orthogonal R1 axis and is deliberately left off here
so that Stage-1 footprint discovery stays trivial (F4 lossless within a channel)
and the only new stress is the cross-channel conditioning. Each channel reuses
the SAME footprints, so a discovered aggregation must separate co-located modes
by CHANNEL, not by position.

Env: MV_C (2), MV_CROSS_COEFF scale (1.0), N_REALISATIONS (100), NL_T (2400),
     HD_PHI, HD_INNOV_SCALE, DY_SCALE, NL_ALPHA, NL_BETA, NG_DIST, NG_SKEW,
     NG_DF, MV_OUT_DIR (data/realisations_multivar<C>)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

# Pull in grid / modes / weights (N, ny, nx, L, W, W_flat, W_plus, positions).
# We reuse the parent DISJOINT blobs for the per-channel footprints and override
# the graph below with the hetdynamics fine-lag graph.
exec(open(os.path.join(os.path.dirname(__file__), "instantiate_model.py")).read())

from savar.savar import dict_to_matrix
from savar.functions import check_stability

# ── multivariate config ──────────────────────────────────────────────────────
C = int(os.environ.get("MV_C", 2))
assert 2 <= C <= 3, f"R3 is a 2–3 channel rung; got MV_C={C}"
NC = N * C
CROSS_SCALE = float(os.environ.get("MV_CROSS_COEFF", 1.0))

# ── FINE-LAG within-channel graph — BYTE-IDENTICAL to generate_overlap.py ─────
_PHI_DEFAULT = "0.15,0.30,0.42,0.55,0.68,0.78,0.86,0.92"
PHI = [float(x) for x in os.environ.get("HD_PHI", _PHI_DEFAULT).split(",")]
assert len(PHI) == 8, f"HD_PHI needs 8 values, got {len(PHI)}"

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
G_base = dict_to_matrix(links_coeffs)          # (N, N, tau_base)
tau_base = G_base.shape[2]                      # = 6

# ── CROSS-CHANNEL edges (the R3 change to Φ) ──────────────────────────────────
# rows (cause_node, cause_ch, effect_node, effect_ch, lag, coeff). Small
# coefficients so the extended VAR stays stable; scaled by MV_CROSS_COEFF.
# Chosen to exercise the cases PCMCI+ conditioning must handle:
#   * same-node cross-field coupling  (0,ch0)->(0,ch1)  — co-located confounder
#   * cross-node cross-field, lagged  (3,ch0)->(5,ch1)  and (2,ch1)->(6,ch0)
_CROSS_C2 = [
    (0, 0, 0, 1, 1,  0.25),   # ch0 X0 → ch1 X0  (lag 1, same footprint)
    (3, 0, 5, 1, 2,  0.20),   # ch0 X3 → ch1 X5  (lag 2)
    (2, 1, 6, 0, 1,  0.18),   # ch1 X2 → ch0 X6  (lag 1, reverse direction)
]
# extra edges when a 3rd channel is present
_CROSS_C3 = [
    (1, 1, 4, 2, 1,  0.22),   # ch1 X1 → ch2 X4  (lag 1)
    (7, 2, 2, 0, 3,  0.15),   # ch2 X7 → ch0 X2  (lag 3, closes a cross-channel loop)
]
_cross_raw = list(_CROSS_C2) + (list(_CROSS_C3) if C >= 3 else [])
CROSS_EDGES = [(cn, cc, en, ec, lag, coeff * CROSS_SCALE)
               for (cn, cc, en, ec, lag, coeff) in _cross_raw]

tau_max = max([tau_base] + [lag for (_, _, _, _, lag, _) in CROSS_EDGES])

# ── build the extended (NC × NC × tau_max) graph ──────────────────────────────
def _m(node, ch):
    return ch * N + node

G_ext = np.zeros((NC, NC, tau_max))
for ch in range(C):                              # C independent within-channel copies
    G_ext[_m(0, ch):_m(0, ch) + N, _m(0, ch):_m(0, ch) + N, :tau_base] = G_base
for (cn, cc, en, ec, lag, coeff) in CROSS_EDGES: # cross-channel couplings
    G_ext[_m(en, ec), _m(cn, cc), lag - 1] += coeff

# stability of the extended companion matrix
_top = np.hstack([G_ext[:, :, i] for i in range(tau_max)])
_bot = np.hstack([np.eye(NC * (tau_max - 1)), np.zeros((NC * (tau_max - 1), NC))])
spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.vstack([_top, _bot])))))
assert spectral_radius < 1.0, f"extended VAR UNSTABLE (ρ={spectral_radius:.3f}); lower MV_CROSS_COEFF"

# fine_edges (stacked index) + channel_edges (decoded, human-readable)
fine_edges, channel_edges = [], []
for eff in range(NC):
    for cause in range(NC):
        if cause == eff:
            continue
        for lag in range(1, tau_max + 1):
            c = G_ext[eff, cause, lag - 1]
            if c != 0:
                fine_edges.append((cause, eff, lag, float(c)))
                channel_edges.append((cause % N, cause // N, eff % N, eff // N,
                                      lag, float(c), int(cause // N != eff // N)))
fine_edges_arr    = np.array(fine_edges, dtype=np.float32)
channel_edges_arr = np.array(channel_edges, dtype=np.float32)
n_cross = int(channel_edges_arr[:, 6].sum()) if len(channel_edges_arr) else 0
assert n_cross >= 1, "R3 requires at least one cross-channel edge"

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

OUT_DIR = os.environ.get("MV_OUT_DIR", os.path.join("data", f"realisations_multivar{C}"))
os.makedirs(OUT_DIR, exist_ok=True)

# cross_edges for the bilinear term, in stacked-index (effect, cause, lag, coeff)
# — every off-diagonal edge of G_ext, exactly as the parent bilinear used every
# off-diagonal edge of G.
bilin_edges = [(eff, cause, lag, G_ext[eff, cause, lag - 1])
               for eff in range(NC) for cause in range(NC)
               for lag in range(1, tau_max + 1)
               if eff != cause and G_ext[eff, cause, lag - 1] != 0]


def _g_sat(m):
    return (1.0 - NL_ALPHA) * m + NL_ALPHA * np.tanh(m)


def _modes_at(data, tcol):
    """Per-channel W-pool of the pixel field at time column tcol → (NC,)."""
    z = np.empty(NC)
    for ch in range(C):
        z[ch * N:(ch + 1) * N] = W_flat @ data[ch, :, tcol]
    return z


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
    """noise_field (C, L, total_T) → observations (C, L, T) after burn-in."""
    total_T = noise_field.shape[2]
    data = noise_field.copy()
    for t in range(tau_max, total_T):
        zl = {lag: _modes_at(data, t - lag) for lag in range(1, tau_max + 1)}
        contrib = np.zeros(NC)
        for lag in range(1, tau_max + 1):
            contrib += G_ext[:, :, lag - 1] @ _g_sat(zl[lag])
        if NL_BETA != 0.0 and bilin_edges:                     # bilinear nonlinearity
            q  = np.zeros(NC)
            m1 = zl[1]
            for eff, cause, lag, coef in bilin_edges:
                q[eff] += coef * zl[lag][cause] * m1[eff]
            contrib += NL_BETA * np.tanh(q)
        for ch in range(C):                                    # emit each channel
            data[ch, :, t] += W_plus @ contrib[ch * N:(ch + 1) * N]
    return data[:, :, burn:]


total_T = T + burn
_tau = [(-1.0 / np.log(p) if 0 < p < 1 else float('inf')) for p in PHI]
print(f"\nGenerating {N_REALISATIONS} realisations  [R3 MULTIVARIATE C={C}, "
      f"dynamics = hetdynamics_eqvar]")
print(f"  Grid: {ny}x{nx}  L={L}  N={N}  C={C}  NC={NC}  T_fine={T}  burn={burn}")
print(f"  Self-loops φ: {[round(p,2) for p in PHI]}  (τ spread {max(_tau)/min(_tau):.1f}×)")
print(f"  Cross-channel edges ({n_cross}):")
for (cn, cc, en, ec, lag, coeff) in CROSS_EDGES:
    print(f"    (X{cn},ch{cc}) → (X{en},ch{ec})  lag {lag}  coeff {coeff:+.3f}")
print(f"  Within-channel edges: {len(fine_edges)-n_cross} total "
      f"({(len(fine_edges)-n_cross)//C} per channel × {C})")
print(f"  NL_ALPHA={NL_ALPHA} NL_BETA={NL_BETA}  {NG_DIST}(α={NG_SKEW})  DY_SCALE={DY_SCALE}")
print(f"  Extended-VAR spectral radius: {spectral_radius:.4f}  "
      f"({'STABLE' if spectral_radius < 1 else 'UNSTABLE'})")
print(f"  Output: {OUT_DIR}/\n")

t_start = time.time()
max_abs_global = 0.0

for seed in range(N_REALISATIONS):
    rng = np.random.default_rng(seed)
    # per-channel innovations; INNOV_SCALE (length N) tiled across channels
    eps_x = draw_innovations(rng, (C, N, total_T)) * INNOV_SCALE[None, :, None]
    eps_y = EPS_Y_STD * rng.standard_normal((C, L, total_T))
    noise_field = np.stack([W_plus @ eps_x[ch] + eps_y[ch] for ch in range(C)], axis=0)

    obs = generate_obs(noise_field)                       # (C, L, T)
    Z   = np.concatenate([W_flat @ obs[ch] for ch in range(C)], axis=0)   # (NC, T)
    max_abs_global = max(max_abs_global, float(np.abs(Z).max()))

    np.savez_compressed(
        os.path.join(OUT_DIR, f"realisation_{seed:03d}.npz"),
        observations       = obs.astype(np.float32),          # (C, L, T)
        latent_states      = Z.astype(np.float32),            # (NC, T)
        ground_truth_graph = G_ext.astype(np.float32),        # (NC, NC, tau_max)
        fine_edges         = fine_edges_arr,                  # (cause_m, eff_m, lag, coeff)
        channel_edges      = channel_edges_arr,               # decoded + is_cross flag
        W                  = W_flat.astype(np.float32),       # (N, L) shared footprints
        W_plus             = W_plus.astype(np.float32),       # (L, N)
        nl_meta            = np.array([NL_ALPHA, NL_BETA], dtype=np.float32),
        ng_meta            = np.array([{"gaussian":0,"skewnorm":1,"t":2}[NG_DIST],
                                       NG_SKEW, NG_DF], dtype=np.float32),
        mv_meta            = np.array([N, C, L, NC, tau_max], dtype=np.int64),
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

# ── verification ─────────────────────────────────────────────────────────────
d  = np.load(os.path.join(OUT_DIR, "realisation_000.npz"))
Zo = d["latent_states"].astype(np.float64)
print(f"\nVerification (realisation 0):")
print(f"  observations shape={d['observations'].shape}  (C, L, T)")
print(f"  latent_states shape={d['latent_states'].shape}  (NC, T)")
print(f"  ground_truth_graph shape={d['ground_truth_graph'].shape}  (NC, NC, tau)")
print(f"  cross-channel edges in graph: {int(d['channel_edges'][:,6].sum())}")
print(f"  Z std per mode (chan-major): {Zo.std(1).round(3)}")
