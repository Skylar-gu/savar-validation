"""
R4 SCALE rung generator (litext plan, notes/literature_extension_experiments.md
R4 row) — fork of generate_overlap.py scaled to N=24 modes on a larger grid,
for the unknown-N discovery experiment (Leiden / N̂ estimation, Stage 1).

WHAT CHANGES vs the parent / R1 rung, and WHY:
  * MODE COUNT  N: 8 -> 24 (a "realistic" mode count; N is unknown a priori to
    the Stage-1 discovery routine).
  * GRID: 50×50 -> 80×80 (L = 6400).  find_mode_positions(res=(80,80),
    n_var=24) tiles the grid with a 5×5 lattice of size-16 slots (25, 24 used).
    The per-mode footprint is therefore the SAME 16×16 Gaussian blob the parent
    and R1 rungs emit (parent used size-16 blobs on 50×50); only the mode COUNT
    and grid EXTENT scale.  So R4 is a clean "scale" manipulation on top of a
    byte-identical emission family — not an overlap manipulation (that is R1).
  * COUPLING GRAPH: the 8-node hetdynamics_eqvar motif is replicated in 3
    disjoint blocks of 8 (offsets 0,8,16), each keeping the parent's within-
    block structure (hub X0, sink X7, converging X3, mixed lags 1..6, negative
    edges, long-range ℓ4/ℓ6 edges), PLUS three sparse forward inter-block
    bridges (block b hub -> block b+1 converging node, and one long ℓ5 bridge)
    so the 24-node graph has genuine cross-community causal structure for
    PCMCI+ parent search to recover.  τ_max stays 6.

WHAT STAYS BYTE-IDENTICAL to the parent dynamics family (comparability is the
point of a scale rung): the φ band (HD_PHI, 0.15..0.92, tiled ×3 across the 3
blocks so the per-block τ spread is unchanged), the eqvar empirical innovation
scaling (tiled ×3, renormalised to mean 1), DY_SCALE=0.05, the saturating +
bilinear nonlinearity (NL_ALPHA=0.5, NL_BETA=0.15), skew-normal innovations,
per-realisation seeds, and the W_flat/W_plus mode-recursion machinery.  The
known saturated-self-loop caveat (g_sat on the φ self-loop ⇒ small-signal
τ_eff shrinks) applies here exactly as in the parent — deliberately.

Emission W: disjoint (non-overlapping) 16×16 Gaussian blobs, one per lattice
slot, L1-normalised (parent convention), FIXED across realisations.  Because
the blobs are disjoint and full-row-rank, W_flat @ W_plus = I_N, so Z =
W_flat @ obs follows the SAME mode VAR; the only pixel leak is W_flat @ eps_y.

Env: N_MODES (24), NL_T (2400), N_REALISATIONS (100), DY_SCALE (0.05),
     HD_PHI, HD_INNOV_SCALE (8 base values each, tiled ×3), NL_ALPHA (0.5),
     NL_BETA (0.15), NG_DIST (skewnorm), NG_SKEW (4.0), NG_DF (5.0),
     GRID (80), SC_OUT_DIR (data/realisations_scale24).
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))

from savar.savar import dict_to_matrix
from savar.functions import check_stability, create_graph, create_random_mode
from savar.model_generator import SavarGenerator

# ── 1. Spatial domain + emission map W (disjoint Gaussian blobs) ─────────────
N    = int(os.environ.get("N_MODES", 24))
GRID = int(os.environ.get("GRID", 80))
ny = nx = GRID
L  = ny * nx

size, positions = SavarGenerator.find_mode_positions(res=(ny, nx), n_var=N)

W = np.zeros((N, ny, nx))
for i in range(N):
    y1, y2, x1, x2 = positions[i]
    blob_h, blob_w = y2 - y1, x2 - x1
    blob = create_random_mode((blob_w, blob_h), random=False)   # (h, w) Gaussian bump
    blob /= blob.sum()                                          # L1-normalise
    W[i, y1:y2, x1:x2] = blob

W_flat = W.reshape(N, L)              # (24, 6400)
W_plus = np.linalg.pinv(W_flat)       # (6400, 24)

# footprint diagnostics (disjoint ⇒ ~0 off-diag cosine; kept for parity with
# the overlap generator's npz schema so downstream loaders are agnostic)
Wu = W_flat / np.linalg.norm(W_flat, axis=1, keepdims=True)
COSM = Wu @ Wu.T
supp = W_flat > 0
JAC = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        inter = (supp[i] & supp[j]).sum(); union = (supp[i] | supp[j]).sum()
        JAC[i, j] = inter / union if union else 0.0
_off = ~np.eye(N, dtype=bool)
print(f"[scale W] N={N}  grid={ny}×{nx}  L={L}  blob={size}×{size}px  "
      f"({len(positions)} of {(ny//size)*(nx//size)} lattice slots used)")
print(f"  max off-diag cos = {COSM[_off].max():.4f}  (disjoint blobs ⇒ ≈0)")
print(f"  ||W_plus|| col max-abs: {np.abs(W_plus).max():.3f}")

# ── 2. 24-node coupling graph: 3× the hetdynamics motif + bridges ────────────
_PHI_DEFAULT = "0.15,0.30,0.42,0.55,0.68,0.78,0.86,0.92"
PHI = [float(x) for x in os.environ.get("HD_PHI", _PHI_DEFAULT).split(",")]
assert len(PHI) == 8, f"HD_PHI needs 8 base values, got {len(PHI)}"

_EQVAR_SCALE_DEFAULT = "0.97,0.93,0.89,0.85,0.83,0.73,0.68,0.69"
_base_scale = np.array([float(x) for x in os.environ.get(
    "HD_INNOV_SCALE", _EQVAR_SCALE_DEFAULT).split(",")])
assert len(_base_scale) == 8
N_BLOCKS = N // 8
assert N == 8 * N_BLOCKS, "N=24 rung: N must be a multiple of 8 (block motif)"

# per-block cross-edge template (cause, effect, lag, coeff) — the hetdynamics
# non-auto edge set (relative node indices 0..7 within a block).
_TEMPLATE = [
    (2, 0, 3,  0.22),   # X2→X0 (ℓ3, weak downstream feedback)
    (0, 1, 1,  0.35),   # X0→X1 (hub drive)
    (1, 2, 1,  0.40),   # X1→X2 (chain)
    (0, 3, 1,  0.30),   # X0→X3 (diverging)
    (2, 3, 2, -0.30),   # X2→X3 (ℓ2, converging, negative)
    (1, 4, 3,  0.25),   # X1→X4 (ℓ3)
    (4, 5, 2,  0.35),   # X4→X5 (ℓ2)
    (0, 5, 4, -0.20),   # X0→X5 (ℓ4, negative)
    (3, 6, 2,  0.30),   # X3→X6 (ℓ2)
    (5, 6, 6,  0.25),   # X5→X6 (ℓ6, long range)
    (6, 7, 4,  0.20),   # X6→X7 (ℓ4)
    (3, 7, 6, -0.15),   # X3→X7 (ℓ6, negative sink input)
]

# sparse forward inter-block bridges (cause_block, cause_node, eff_block,
# eff_node, lag, coeff) — hub of block b drives converging node of block b+1,
# plus one long ℓ5 bridge; forward-only keeps the block DAG stable.
_ALL_BRIDGES = [
    (0, 0, 1, 3, 2, 0.12),   # b0.X0 → b1.X3 (ℓ2)
    (1, 0, 2, 3, 3, 0.12),   # b1.X0 → b2.X3 (ℓ3)
    (0, 6, 2, 6, 5, 0.10),   # b0.X6 → b2.X6 (ℓ5, long cross-community)
]
# only keep bridges whose endpoints exist for the configured N_BLOCKS
_BRIDGES = [br for br in _ALL_BRIDGES if br[0] < N_BLOCKS and br[2] < N_BLOCKS]

links_coeffs = {j: [] for j in range(N)}
PHI_FULL   = np.array(PHI * N_BLOCKS)                       # (24,) φ, band tiled ×N_BLOCKS
_raw_scale = np.concatenate([_base_scale] * N_BLOCKS)      # (24,) innov scale, tiled
INNOV_SCALE = _raw_scale / _raw_scale.mean()               # renormalise to mean 1
for j in range(N):
    links_coeffs[j].append(((j, -1), float(PHI_FULL[j])))  # self-loop
for b in range(N_BLOCKS):
    o = 8 * b
    for cause, eff, lag, c in _TEMPLATE:
        links_coeffs[o + eff].append(((o + cause, -lag), c))
for cb, cn, eb, en, lag, c in _BRIDGES:
    links_coeffs[8 * eb + en].append(((8 * cb + cn, -lag), c))

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

# ── 3. noise / lengths / nonlinearity — parent eqvar values ──────────────────
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

OUT_DIR = os.environ.get("SC_OUT_DIR", os.path.join("data", "realisations_scale24"))
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


def _bilinear(m_by_lag):
    """bilinear cross term; m_by_lag[lag] = W_flat @ data[:, t-lag] (precomputed
    in generate_obs — identical math to the parent, just no redundant matmuls)."""
    if NL_BETA == 0.0 or not cross_edges:
        return np.zeros(N)
    q  = np.zeros(N)
    m1 = m_by_lag[1]
    for j, i, lag, c in cross_edges:
        q[j] += c * m_by_lag[lag][i] * m1[j]
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
        m_by_lag = {lag: W_flat @ data[:, t - lag] for lag in range(1, tau_max + 1)}
        contrib = np.zeros(N)
        for lag in range(1, tau_max + 1):
            contrib += G[:, :, lag - 1] @ _g_sat(m_by_lag[lag])
        contrib += _bilinear(m_by_lag)
        data[:, t] += W_plus @ contrib
    return data[:, burn:]


total_T = T + burn
_tau = [(-1.0/np.log(p) if 0 < p < 1 else float('inf')) for p in PHI]
print(f"\nGenerating {N_REALISATIONS} realisations  [R4 SCALE N={N}, dynamics = hetdynamics_eqvar ×{N_BLOCKS}]")
print(f"  Grid: {ny}×{nx}  L={L}  N={N}  T_fine={T}  burn={burn}  tau_max={tau_max}")
print(f"  Self-loops φ (band tiled ×{N_BLOCKS}): {[round(p,2) for p in PHI]}  (τ spread {max(_tau)/min(_tau):.1f}×)")
print(f"  Innov scales (eqvar empirical, mean-normalised): base {[round(s,2) for s in _base_scale]}")
print(f"  Cross-edges: {len(cross_edges)}  (12 within-block ×{N_BLOCKS} + {len(_BRIDGES)} bridges)")
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
        sc_meta            = np.array([N, GRID, size, N_BLOCKS], dtype=np.float32),
        metadata           = np.array([N, L, T, DY_SCALE, seed, spectral_radius]),
    )

    if (seed + 1) % 10 == 0 or seed == 0:
        elapsed = time.time() - t_start
        rate    = (seed + 1) / elapsed
        eta     = (N_REALISATIONS - seed - 1) / rate
        print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {elapsed:.1f}s  ETA {eta:.1f}s  ({rate:.2f} real/s)")

total = time.time() - t_start
print(f"\nDone. {N_REALISATIONS} realisations in {total:.1f}s")
print(f"  Global max |Z|: {max_abs_global:.3f}  "
      f"({'STABLE' if np.isfinite(max_abs_global) and max_abs_global < 1e3 else 'UNSTABLE'})")

d   = np.load(os.path.join(OUT_DIR, f"realisation_000.npz"))
Zo  = d["latent_states"].astype(np.float64)
print(f"\nVerification (realisation 0):")
print(f"  obs shape={d['observations'].shape}  Z std per mode (first 8): {Zo.std(1)[:8].round(3)}")
print(f"  Z std range over 24 modes: [{Zo.std(1).min():.3f}, {Zo.std(1).max():.3f}]")
