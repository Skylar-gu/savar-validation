"""
Generate MOVING-MECHANISM SAVAR realisations with fine sub-mechanisms
(spec: notes/moving_mechanism_subres_spec.md; fork of generate_hetdynamics.py).

Two changes vs generate_hetdynamics.py
--------------------------------------
1. MECHANISM IDENTITY IS DECOUPLED FROM LOCATION (MOVE=place).
   Mechanism k = (phi_k, causal parents) is IDENTICAL in every realisation
   (same links_coeffs as hetdynamics, eqvar innovation scaling), but each
   mechanism's Gaussian blob CENTRE is drawn freshly per realisation
   (uniform with margin; rejection-sampled so footprints stay disjoint —
   pairwise Jaccard 0 by construction, Block-G bookkeeping preserved).
   A positional code is therefore no longer optimal: to generalise the
   forecaster must encode "which mechanism" separately from "where".

2. FINE SUB-MECHANISMS + SUB-RESOLUTION OBSERVATION OPERATOR.
   The field is generated on a FINE grid (100x100) and observed through
     D_sub : keep every 2nd pixel  -> 50x50 (ALIASING regime, primary;
             matches every downstream script's 50x50 assumption)
     D_avg : 2x2 block average     -> 50x50 (LOW-PASS control)
   Within each parent blob sit TWO fine sub-sources at the SAME sub-position:
     sub (j,0) "fast":  phi=0.30, parent = mechanism j        (lag 1)
     sub (j,1) "slow":  phi=0.90, parent = mechanism (j+3)%8  (lag 2)
   with fine spatial patterns  envelope*cos(pi(x+y)) (checkerboard) and
   envelope*cos(pi x) (x-stripes). Both patterns sit at the EXACT fine-grid
   Nyquist, so under D_sub(even,even) BOTH alias to the same smooth +envelope
   — an exact spatial collision (the SAVAR embedding of testbed alpha's
   k_low/k_high pair): separating the pair is only possible through their
   distinct temporal/causal signatures. Under D_avg both are annihilated
   exactly (2x2 mean of either pattern = 0).
   NOTE the spec sketch says "distinct fine positions"; we co-locate the pair
   deliberately — at distinct positions each sub would be partially separable
   by SPACE alone (each aliases to its own envelope), which re-admits the
   position-readout confound the whole spec exists to remove.
   T4 identical-twin null: blob 7's pair instead has IDENTICAL phi (0.60) and
   IDENTICAL parent (mechanism 7, lag 1) — dynamically indistinguishable,
   must be unrecoverable individually everywhere (their sum is fair game).

Ground truth saved per realisation
----------------------------------
  Z_fine          (16, T)  INJECTED sub amplitude = contrib_m(t) + eps_m(t)
                           (well-defined even for the twins; primary target)
  Z_fine_measured (16, T)  fine-grid readout  W_sub_fine @ obs_fine
  latent_states   (8, T)   parent amplitudes  W_parent_fine @ obs_fine (compat)
  centres (8,2), sub_pos (16,2), sub_meta (16,4: phi,parent,lag,coeff)
  W (8,2500) coarse D_sub parent pooling maps (L1-norm; downstream compat)
  W_sub_coarse (16,2500) aliased sub envelopes under D_sub (L1-norm)
  W_fine (24, 10000) fine-grid mechanism patterns
  observations (2500,T) D_sub coarse field; observations_avg (2500,T) D_avg.

Everything else (saturating tanh, bilinear parent coupling, skew-normal
innovations, DY noise, T=2400, 100 realisations) inherited unchanged.

Knobs: MM_MOVE (place; advect not yet implemented), SUB_AMP (default 1.0),
       N_REALISATIONS, NL_T, DY_SCALE, HD_PHI, HD_INNOV_SCALE (eqvar default).
Output: data/realisations_movmech_place/
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "savar"))
from savar.savar import dict_to_matrix
from savar.functions import check_stability

# ── geometry ──────────────────────────────────────────────────────────────────
SUBS   = 2                     # sub-resolution factor
NYC = NXC = 50                 # coarse grid (all downstream scripts assume 50x50)
NYF, NXF = NYC * SUBS, NXC * SUBS
LF, LC = NYF * NXF, NYC * NXC
N      = 8                     # parent mechanisms
N_SUB  = 16                    # 2 per parent
N_ALL  = N + N_SUB

BLOB_SIGMA, BLOB_R = 5.0, 11   # parent envelope on the fine grid (23x23 support)
SUB_SIGMA,  SUB_R  = 3.0, 7    # sub envelope (15x15 support)
MIN_SEP = 2 * BLOB_R + 2       # min pairwise centre distance (disjoint + margin)
MARGIN  = BLOB_R + 1

MOVE = os.environ.get("MM_MOVE", "place")
assert MOVE == "place", "only MOVE=place implemented (advect deferred per spec)"

# ── mechanism dynamics (IDENTICAL every realisation — this is the invariant) ──
_PHI_DEFAULT = "0.15,0.30,0.42,0.55,0.68,0.78,0.86,0.92"
PHI = [float(x) for x in os.environ.get("HD_PHI", _PHI_DEFAULT).split(",")]
# eqvar empirical innovation scaling (same as scratch_hetdyn_eqvar_pipeline.sh)
_SCALE_DEFAULT = "0.97,0.93,0.89,0.85,0.83,0.73,0.68,0.69"
_raw_scale = np.array([float(x) for x in os.environ.get("HD_INNOV_SCALE", _SCALE_DEFAULT).split(",")])
INNOV_SCALE_PARENT = _raw_scale / _raw_scale.mean()

links_coeffs = {
    0: [((0, -1),  PHI[0]), ((2, -3),  0.22)],
    1: [((1, -1),  PHI[1]), ((0, -1),  0.35)],
    2: [((2, -1),  PHI[2]), ((1, -1),  0.40)],
    3: [((3, -1),  PHI[3]), ((0, -1),  0.30), ((2, -2), -0.30)],
    4: [((4, -1),  PHI[4]), ((1, -3),  0.25)],
    5: [((5, -1),  PHI[5]), ((4, -2),  0.35), ((0, -4), -0.20)],
    6: [((6, -1),  PHI[6]), ((3, -2),  0.30), ((5, -6),  0.25)],
    7: [((7, -1),  PHI[7]), ((6, -4),  0.20), ((3, -6), -0.15)],
}
check_stability(links_coeffs)
G_par  = dict_to_matrix(links_coeffs)           # (8, 8, 6)
TAU_MAX = G_par.shape[2]                         # 6

# sub-mechanism table: (phi, parent, lag, coeff, pattern) — pattern 0=checker, 1=xstripes
SUB_PHI_FAST, SUB_PHI_SLOW, SUB_COEFF = 0.30, 0.90, 0.45
TWIN_BLOB, TWIN_PHI = 7, 0.60
sub_meta = []
for j in range(N):
    if j == TWIN_BLOB:   # T4 identical-twin pair: same phi, same parent, same lag
        sub_meta.append((TWIN_PHI, j, 1, SUB_COEFF, 0))
        sub_meta.append((TWIN_PHI, j, 1, SUB_COEFF, 1))
    else:
        sub_meta.append((SUB_PHI_FAST, j, 1, SUB_COEFF, 0))
        sub_meta.append((SUB_PHI_SLOW, (j + 3) % N, 2, SUB_COEFF, 1))
sub_meta = np.array(sub_meta, dtype=np.float32)   # (16, 5)

# full graph G_all (24,24,6): parent block + sub rows (subs are sinks)
G_all = np.zeros((N_ALL, N_ALL, TAU_MAX), dtype=np.float64)
G_all[:N, :N, :] = G_par
for m in range(N_SUB):
    phi, par, lag = sub_meta[m, 0], int(sub_meta[m, 1]), int(sub_meta[m, 2])
    G_all[N + m, N + m, 0]   = phi                    # self-loop lag 1
    G_all[N + m, par, lag-1] = sub_meta[m, 3]         # parent drive

SUB_AMP = float(os.environ.get("SUB_AMP", 1.0))
INNOV_SCALE = np.concatenate([
    INNOV_SCALE_PARENT,
    SUB_AMP * np.sqrt(1.0 - sub_meta[:, 0].astype(np.float64) ** 2),
])

# parent cross-edge list for bilinear coupling (unchanged; subs excluded)
cross_edges = [(j, i, lag, G_par[j, i, lag - 1])
               for j in range(N) for i in range(N) for lag in range(1, TAU_MAX + 1)
               if i != j and G_par[j, i, lag - 1] != 0]
fine_edges_arr = np.array([[i, j, l, c] for (j, i, l, c) in cross_edges], dtype=np.float32)

# ── noise / lengths / nonlinearity / innovations (inherited) ─────────────────
DY_SCALE  = float(os.environ.get("DY_SCALE", 0.05))
EPS_Y_STD = DY_SCALE ** 0.5
T    = int(os.environ.get("NL_T", 2400))
burn = 300
NL_ALPHA = float(os.environ.get("NL_ALPHA", 0.5))
NL_BETA  = float(os.environ.get("NL_BETA",  0.15))
NG_SKEW  = float(os.environ.get("NG_SKEW", 4.0))
N_REALISATIONS = int(os.environ.get("N_REALISATIONS", 100))
OUT_DIR = os.environ.get("MM_OUT_DIR", os.path.join("data", f"realisations_movmech_{MOVE}"))
os.makedirs(OUT_DIR, exist_ok=True)


def draw_innovations(rng, shape, a=NG_SKEW):
    delta = a / np.sqrt(1.0 + a * a)
    z0 = np.abs(rng.standard_normal(shape))
    z1 = rng.standard_normal(shape)
    x = delta * z0 + np.sqrt(1.0 - delta * delta) * z1
    mean = delta * np.sqrt(2.0 / np.pi)
    var  = 1.0 - 2.0 * delta * delta / np.pi
    return (x - mean) / np.sqrt(var)


def _g_sat(m):
    return (1.0 - NL_ALPHA) * m + NL_ALPHA * np.tanh(m)


# ── spatial patterns ──────────────────────────────────────────────────────────
YY, XX = np.mgrid[0:NYF, 0:NXF]
CHECKER  = ((-1.0) ** (YY + XX)).astype(np.float64)      # cos(pi(x+y))
XSTRIPES = ((-1.0) ** XX).astype(np.float64)             # cos(pi x)


def gaussian_env(cy, cx, sigma, r):
    e = np.exp(-(((YY - cy) ** 2 + (XX - cx) ** 2) / (2 * sigma ** 2)))
    e[((YY - cy) ** 2 + (XX - cx) ** 2) > r ** 2] = 0.0
    return e


def draw_centres(rng):
    """8 blob centres, uniform with margin, pairwise distance >= MIN_SEP."""
    for _ in range(2000):
        pts = []
        tries = 0
        while len(pts) < N and tries < 500:
            p = rng.uniform(MARGIN, [NYF - MARGIN, NXF - MARGIN], size=2)
            if all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= MIN_SEP**2 for q in pts):
                pts.append(p)
            tries += 1
        if len(pts) == N:
            return np.array(pts)
    raise RuntimeError("centre rejection sampling failed")


def build_W(centres):
    """W_fine (24, LF): 8 parent blobs + 16 sub patterns (co-located pairs)."""
    Wf = np.zeros((N_ALL, LF))
    sub_pos = np.zeros((N_SUB, 2))
    for j in range(N):
        env = gaussian_env(centres[j, 0], centres[j, 1], BLOB_SIGMA, BLOB_R)
        Wf[j] = (env / env.sum()).ravel()
        # both subs of blob j at the same sub-position (centre of blob)
        cy, cx = centres[j]
        senv = gaussian_env(cy, cx, SUB_SIGMA, SUB_R)
        senv /= np.abs(senv).sum()
        for i in range(2):
            m = 2 * j + i
            pat = CHECKER if int(sub_meta[m, 4]) == 0 else XSTRIPES
            Wf[N + m] = (senv * pat).ravel()
            sub_pos[m] = (cy, cx)
    return Wf, sub_pos


# coarse observation operators (index arrays; fast)
_sub_rows = np.arange(0, NYF, SUBS)
def D_sub(field_fine):     # (..., NYF, NXF) -> (..., 50, 50)
    return field_fine[..., ::SUBS, ::SUBS]
def D_avg(field_fine):
    s = field_fine.reshape(*field_fine.shape[:-2], NYC, SUBS, NXC, SUBS)
    return s.mean(axis=(-3, -1))


def generate_one(seed):
    rng = np.random.default_rng(seed)
    centres = draw_centres(rng)
    W_fine, sub_pos = build_W(centres)                # (24, LF)
    W_plus = np.linalg.pinv(W_fine)                   # (LF, 24)

    total_T = T + burn
    eps_x = draw_innovations(rng, (N_ALL, total_T)) * INNOV_SCALE[:, None]
    data  = W_plus @ eps_x + EPS_Y_STD * rng.standard_normal((LF, total_T))

    M_hist = np.zeros((N_ALL, total_T))               # measured amplitudes cache
    for t in range(TAU_MAX):
        M_hist[:, t] = W_fine @ data[:, t]
    contrib_hist = np.zeros((N_ALL, total_T))
    for t in range(TAU_MAX, total_T):
        contrib = np.zeros(N_ALL)
        for lag in range(1, TAU_MAX + 1):
            contrib += G_all[:, :, lag - 1] @ _g_sat(M_hist[:, t - lag])
        # bilinear advective coupling on parent cross-edges (inherited)
        if NL_BETA != 0.0:
            q = np.zeros(N_ALL)
            m1 = M_hist[:N, t - 1]
            for j, i, lag, c in cross_edges:
                q[j] += c * M_hist[i, t - lag] * m1[j]
            contrib += NL_BETA * np.tanh(q)
        data[:, t] += W_plus @ contrib
        M_hist[:, t] = W_fine @ data[:, t]
        contrib_hist[:, t] = contrib

    obs_fine = data[:, burn:]                          # (LF, T)
    Z_par    = M_hist[:N, burn:]                       # measured parent amplitudes
    Zf_meas  = M_hist[N:, burn:]                       # measured sub amplitudes
    Z_fine   = contrib_hist[N:, burn:] + eps_x[N:, burn:]   # injected sub amplitudes

    ff = obs_fine.T.reshape(T, NYF, NXF)
    obs_sub = D_sub(ff).reshape(T, LC).T               # (2500, T)
    obs_avg = D_avg(ff).reshape(T, LC).T

    # coarse pooling maps
    Wp_c = D_sub(W_fine[:N].reshape(N, NYF, NXF)).reshape(N, LC)
    Wp_c = np.clip(Wp_c, 0, None)
    Wp_c /= Wp_c.sum(1, keepdims=True)
    Ws_c = D_sub(W_fine[N:].reshape(N_SUB, NYF, NXF)).reshape(N_SUB, LC)
    Ws_c = np.clip(Ws_c, 0, None)                      # aliased envelopes (>=0)
    Ws_c /= Ws_c.sum(1, keepdims=True) + 1e-12

    return dict(
        observations      = obs_sub.astype(np.float32),
        observations_avg  = obs_avg.astype(np.float32),
        latent_states     = Z_par.astype(np.float32),
        Z_fine            = Z_fine.astype(np.float32),
        Z_fine_measured   = Zf_meas.astype(np.float32),
        W                 = Wp_c.astype(np.float32),
        W_sub_coarse      = Ws_c.astype(np.float32),
        W_fine            = W_fine.astype(np.float32),
        centres           = centres.astype(np.float32),
        sub_pos           = sub_pos.astype(np.float32),
        sub_meta          = sub_meta,
        ground_truth_graph= G_par.astype(np.float32),
        fine_edges        = fine_edges_arr,
        nl_meta           = np.array([NL_ALPHA, NL_BETA], dtype=np.float32),
        metadata          = np.array([N, LC, T, DY_SCALE, seed, SUBS]),
    )


if __name__ == "__main__":
    _tau = [(-1.0/np.log(p) if 0 < p < 1 else np.inf) for p in PHI]
    print(f"\nMoving-mechanism SAVAR  [MOVE={MOVE}]  fine {NYF}x{NXF} -> coarse {NYC}x{NXC} (s={SUBS})")
    print(f"  parents: phi={[round(p,2) for p in PHI]}  tau={[round(t,2) for t in _tau]}")
    print(f"  subs: fast phi={SUB_PHI_FAST} (parent j, lag1) / slow phi={SUB_PHI_SLOW} "
          f"(parent (j+3)%8, lag2), coeff={SUB_COEFF}, co-located colliding pair "
          f"(checker vs x-stripes @ Nyquist)")
    print(f"  twin blob {TWIN_BLOB}: both subs phi={TWIN_PHI}, parent {TWIN_BLOB}, lag1 (T4 null)")
    print(f"  SUB_AMP={SUB_AMP}  T={T}  burn={burn}  n_real={N_REALISATIONS}")
    print(f"  Output: {OUT_DIR}/\n")

    t0 = time.time()
    for seed in range(N_REALISATIONS):
        out = generate_one(seed)
        np.savez_compressed(os.path.join(OUT_DIR, f"realisation_{seed:03d}.npz"), **out)
        if (seed + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{seed+1:3d}/{N_REALISATIONS}]  {el:.1f}s  ETA {(N_REALISATIONS-seed-1)*el/(seed+1):.1f}s")

    # verification on the last realisation
    d = out
    print(f"\nVerification (realisation {N_REALISATIONS-1}):")
    print(f"  obs_sub std={d['observations'].std():.4f}  obs_avg std={d['observations_avg'].std():.4f}")
    print(f"  parent Z std per mode: {np.round(d['latent_states'].std(1), 3)}")
    print(f"  Z_fine std per sub:    {np.round(d['Z_fine'].std(1), 3)}")
    print(f"  corr(Z_fine, Z_fine_measured) per sub: "
          f"{np.round([np.corrcoef(d['Z_fine'][m], d['Z_fine_measured'][m])[0,1] for m in range(N_SUB)], 3)}")
    # visibility of the collided sub pair in the D_sub observation
    for m in (0, 1):
        w = d['W_sub_coarse'][m]; w = w / (np.linalg.norm(w) + 1e-12)
        proj = w @ d['observations']
        cs = [abs(np.corrcoef(proj, d['Z_fine'][k])[0, 1]) for k in (0, 1)]
        print(f"  D_sub pooled envelope proj vs Z_fine[0]/[1]: {cs[0]:.3f}/{cs[1]:.3f}  (sub {m})")
    # Jaccard non-overlap bookkeeping
    sup = d['W_fine'][:N] != 0
    jac = max((sup[a] & sup[b]).sum() / max((sup[a] | sup[b]).sum(), 1)
              for a in range(N) for b in range(a+1, N))
    print(f"  max pairwise parent-footprint Jaccard: {jac:.4f} (0 = disjoint)")
