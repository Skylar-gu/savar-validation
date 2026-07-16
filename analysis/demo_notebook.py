# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SAVAR causal-interpretability — end-to-end demo
#
# **The claim in one sentence:** given a black-box forecaster trained on a spatial
# system, we can recover the model's causal wiring *and* attach a calibrated
# confidence to that recovery — **with no answer key**.
#
# The demo is fully self-contained: **cell 0** is a toggle config exposing the five
# realism mechanisms (seasonality, non-linearity, non-Gaussian innovations, moving
# mechanism, multivariate channels — all off by default, so the base case is the
# clean linear-Gaussian SAVAR parent), and **`generate_world(CFG)`** builds the
# whole world — observations, latent modes, mode maps, ground-truth graph — in one
# function whose math is lifted from the repo generators
# (`data_gen/instantiate_model.py`, `generate_finecadence.py`,
# `generate_movmech.py`, `generate_multivar.py`). The downstream sections then run
# the real pipeline on that generated world. The seven sections mirror the
# pipeline scripts one-to-one, and each cell contains the **actual computation** a
# reader would run (lifted/adapted from the scripts named in each header):
# 1. **DGP** — `generate_world(CFG)`: build the SAVAR world + true graph.
# 2. **Forecaster** — `train/gnn/gnn_forecaster.py`: the MeshGNN black box.
# 3. **E1** — `sae/discover_modes.py`: unsupervised discovery of candidate mode maps.
# 4. **Ĝ_int** — `pcmci/e4_agreement.py` (int stage): PCMCI+ on `Ŵ@obs` series.
# 5. **Ĝ_dyn** — `pcmci/e4_agreement.py` (dyn stage): perturb the forecaster.
# 6. **PX** — `pcmci/e4_agreement.py` (cal stage): pool-crossed agreement, no truth.
# 7. **Trust dial** — `analysis/e4_final_calibration.py`: agreement → predicted accuracy.
#
# **The spine of the recipe** (one row = one section): *make a world → train a
# black box → find its parts → read its wiring two independent ways → let their
# agreement pick the answer and predict its accuracy, all without the answer key.*
#
# **How to run.** The heavy steps (world generation, E1 discovery, the PCMCI +
# teacher-forced response loops) take many minutes to hours, so they are guarded by
# a single `RUN = False` flag: as a plain script the file just *defines* the real
# pipeline and executes nothing expensive. In Jupyter, uncomment the
# `# %matplotlib inline` magic and flip `RUN = True` to execute it. All
# plot-drawing lines are commented out so nothing renders headless.

# %%
# %matplotlib inline   # <- uncomment in Jupyter for inline figures

import os
import sys
from dataclasses import dataclass, field

import numpy as np

RUN = False   # <- flip to True in Jupyter to actually execute the heavy pipeline

# Resolve repo root whether the file is run from repo root or from analysis/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    if "__file__" in globals() else os.path.abspath("..")
sys.path.insert(0, os.path.join(ROOT, "train", "gnn"))
sys.path.insert(0, os.path.join(ROOT, "savar"))

# ============================ CELL 0 — WORLD CONFIG ============================
# One config, five realism toggles. Defaults are all off/neutral, so
# generate_world(CFG) reproduces the clean linear-Gaussian SAVAR parent. Each
# toggle stresses ONE assumption of the recovery recipe (see §1.8 below).
@dataclass
class WorldConfig:
    # ---- base world (parent values; geometry N/ny/nx is fixed in §1.7) -------
    T: int = 500              # usable timesteps per realisation
    burn: int = 200           # discarded transient
    n_real: int = 28          # realisations (last 4 held out for E1 discovery)
    lam: float = 1.0          # latent innovation strength λ (D_x = λ·I_N)
    dy_scale: float = 1.0     # per-pixel white-noise variance (D_y = dy_scale·I_L)
    seed0: int = 0            # base RNG seed (realisation r uses seed0 + r)

    # ---- 1. SEASONALITY — stresses stationarity + periodic confounding -------
    # Designated modes get an intrinsic sinusoid added to their driving
    # innovation; the cycle propagates to their causal children through G.
    seas_modes: list = field(default_factory=list)                  # e.g. [0, 4]
    seas_periods: list = field(default_factory=lambda: [24.0, 168.0, 730.0])
    seas_amp: float = 1.5     # amplitude relative to unit-variance innovations
    seas_phase: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # ---- 2. NON-LINEARITY — stresses ParCorr (a *linear* CI test) ------------
    # nl_alpha: saturating self-term g(m) = (1-α)m + α·tanh(m)  (identity at 0)
    # nl_beta:  bilinear advective coupling on lagged cross-edges, tanh-bounded,
    #           only on EXISTING edges (edge-set preserved by construction).
    nl_alpha: float = 0.0
    nl_beta: float = 0.0

    # ---- 3. NON-GAUSSIAN — stresses ParCorr's Gaussianity assumption ---------
    # Innovations drawn skewed / heavy-tailed, standardized to unit variance so
    # the VAR stationarity condition is untouched.
    ng_dist: str = "gaussian"  # one of {"gaussian", "skewnorm", "t"}
    ng_skew: float = 4.0       # skew-normal shape alpha  (ng_dist="skewnorm")
    ng_df: float = 5.0         # Student-t dof            (ng_dist="t")

    # ---- 4. MOVING MECHANISM — stresses the static-Ŵ assumption of Ŵ@obs -----
    # Footprints W advect: each mode's spatial support drifts over time
    # (periodic roll), so any single static Ŵ is wrong somewhere in the record.
    move: bool = False
    move_speed: float = 0.05   # drift speed in px per timestep
    move_angle_deg: float = 45.0  # base drift direction (per-mode spread added)
    move_every: int = 10       # rebuild the drifted W every k steps (speed knob)

    # ---- 5. MULTIVARIATE — stresses mode separability --------------------------
    # C observed channels per grid cell; block-diagonal Ŵ = I_C ⊗ W, coupled
    # only by a small set of cross-channel edges added to Φ.
    n_channels: int = 1
    cross_scale: float = 1.0   # scales the cross-channel coefficients

CFG = WorldConfig()

# The forecaster checkpoint (section 2). NOTE: each generated world needs its own
# trained forecaster; this default is the parent-world checkpoint and is only
# valid for the default (all-toggles-off) CFG. For a toggled world: save the
# generated realisations, split them (data_gen/data_split.py) and train with
# train/gnn/gnn_forecaster.py (env GNN_DATA_DIR / GNN_CKPT_DIR), then point CKPT here.
CKPT  = os.path.join(ROOT, "checkpoints", "hetdynamics_eqvar", "best.pt")
PLOTS = os.path.join(ROOT, "results", "plots")

# matplotlib is imported but no plt.show() is ever called (drawing lines commented).
import matplotlib
matplotlib.use("Agg")           # headless-safe; harmless in Jupyter
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=120)
print("repo root:", ROOT)
print("RUN =", RUN, "(heavy steps are gated; flip to True to execute)")
print("CFG =", CFG)


# %% [markdown]
# ## 1. The data-generating process (SAVAR)
#
# **SAVAR** = Spatially Aggregated Vector AutoRegression. A low-dimensional causal
# system is "painted" onto a 2-D grid, so the observed data are high-dimensional
# pixels but the true dynamics live among a few hidden **modes**. Every later score
# in the notebook is secretly checked against this world's known wiring — the
# sandbox that lets us validate the truth-free machinery.
#
# ### 1.1 Objects
#
# | symbol | meaning | parent value |
# |---|---|---|
# | $N$ | number of modes (hidden variables) | 8 |
# | $n_y\times n_x$ | grid resolution | $50\times50$ |
# | $L=n_y n_x$ | pixels (observed dim) | 2500 |
# | $W\in\mathbb{R}^{N\times L}$ | mode map: row $i$ = normalized Gaussian blob, $\sum_\ell W_{i\ell}=1$ | 8 disjoint $16\times16$ blobs |
# | $W^{+}=\operatorname{pinv}(W)\in\mathbb{R}^{L\times N}$ | lifts modes back to pixels | — |
# | $\Phi(\tau)\in\mathbb{R}^{N\times N}$ | causal coefficient matrix at lag $\tau$ (the ground-truth graph $G$) | $\tau\in\{1,2\}$ |
# | $\tau_{\max}$ | max lag | 2 |
# | $T,\ \text{burn}$ | usable steps, discarded transient | 500, 200 |
# | $\lambda$ | noise strength | 1.0 |

# %% [markdown]
# ### 1.2 Mode-space causal dynamics
#
# The hidden mode signal is $Z(t)=W\,X(t)\in\mathbb{R}^N$. It follows a linear VAR
# whose coefficients **are** the causal graph:
#
# $$
# Z(t)=\sum_{\tau=1}^{\tau_{\max}}\Phi(\tau)\,Z(t-\tau)+\xi(t).
# $$
#
# An entry $\Phi(\tau)_{j i}\neq0$ means **$X_i(t-\tau)\to X_j(t)$** (cause $i$ at
# lag $\tau$ drives effect $j$).
#
# ### 1.3 Pixel-space realisation (what is actually generated)
#
# The grid field $X(t)\in\mathbb{R}^L$ is evolved directly by projecting the mode
# dynamics down to pixels and back:
#
# $$
# X(t)=\sum_{\tau=1}^{\tau_{\max}} W^{+}\,\Phi(\tau)\,W\,X(t-\tau)\;+\;\eta(t).
# $$
#
# (This is the literal update loop; $W^{+}\Phi W$ is the $L\times L$ pixel
# operator.)
#
# ### 1.4 Coloured spatial noise
#
# Noise is added in pixel space with a mode-structured covariance:
#
# $$
# \eta(t)\sim\mathcal N(0,\ \Sigma_y),\qquad
# \Sigma_y=\lambda\,W^{+}D_x\,(W^{+})^{\top}+D_y,
# $$
#
# with latent-innovation cov $D_x=I_N$ and per-pixel cov $D_y=I_L$. $D_y=I_L$
# guarantees $\Sigma_y\succ0$. So each mode carries independent innovations,
# smeared onto the grid by its blob, plus white per-pixel noise. In code this is
# exactly $\eta(t) = W^{+}\varepsilon_x(t) + \varepsilon_y(t)$ — and
# $\varepsilon_x$ is where three of the realism toggles act (seasonal sinusoids,
# non-Gaussian draws, per-mode scaling).

# %% [markdown]
# ### 1.7 The ground-truth graph (the answer key)
#
# `links_coeffs` (format $\{\,j:[((i,-\tau),\text{coeff}),\dots]\}$); diagonal
# terms are autocorrelation, off-diagonal are the causal edges:
#
# ```
# X0 ← 0.45·X0(t-1) + 0.22·X2(t-2)          X4 ← 0.40·X4(t-1) + 0.25·X1(t-2)
# X1 ← 0.50·X1(t-1) + 0.35·X0(t-1)          X5 ← 0.30·X5(t-1) + 0.35·X4(t-1) − 0.20·X0(t-2)
# X2 ← 0.35·X2(t-1) + 0.40·X1(t-1)          X6 ← 0.50·X6(t-1) + 0.30·X3(t-1) + 0.25·X5(t-2)
# X3 ← 0.55·X3(t-1) + 0.30·X0(t-1) − 0.30·X2(t-1)   X7 ← 0.45·X7(t-1) + 0.20·X6(t-1) − 0.15·X3(t-2)
# ```
#
# Designed features: **X0 = hub** (high out-degree), **X3 = collider** (two
# parents, different signs), **X7 = pure sink** (no outgoing), mixed lag-1/lag-2
# edges, and negative edges (X0→X5, X2→X3, X3→X7). Stationarity is checked before
# generating. The block below is the real construction from
# `data_gen/instantiate_model.py`: deterministic blobs for `W`, the hand-designed
# `links_coeffs` graph, and a stationarity check.

# %%
# --- SAVAR ground-truth parameters (verbatim logic from instantiate_model.py) ----
from savar.savar import dict_to_matrix
from savar.functions import check_stability, create_random_mode
from savar.model_generator import SavarGenerator

N, ny, nx = 8, 50, 50
L = ny * nx                                   # 2500 pixels

# Mode map W: deterministic non-overlapping Gaussian blobs (3x3 lattice, first 8).
size, positions = SavarGenerator.find_mode_positions(res=(ny, nx), n_var=N)
W = np.zeros((N, ny, nx))
for i in range(N):
    y1, y2, x1, x2 = positions[i]
    blob = create_random_mode((x2 - x1, y2 - y1), random=False)
    W[i, y1:y2, x1:x2] = blob / blob.sum()    # L1-normalize each footprint
W_flat = W.reshape(N, L)                       # (8, 2500)
W_plus = np.linalg.pinv(W_flat)                # (2500, 8) lifts modes back to pixels

# Ground-truth causal graph, links_coeffs format { j: [((i, -tau), coeff), ...] }.
# Designed features: X0 = hub (high out-degree), X3 = collider (two signed parents),
# X7 = pure sink, mixed lag-1/lag-2 edges, negative edges (X0->X5, X2->X3, X3->X7).
links_coeffs = {
    0: [((0, -1), 0.45), ((2, -2), 0.22)],
    1: [((1, -1), 0.50), ((0, -1), 0.35)],
    2: [((2, -1), 0.35), ((1, -1), 0.40)],
    3: [((3, -1), 0.55), ((0, -1), 0.30), ((2, -1), -0.30)],
    4: [((4, -1), 0.40), ((1, -2), 0.25)],
    5: [((5, -1), 0.30), ((4, -1), 0.35), ((0, -2), -0.20)],
    6: [((6, -1), 0.50), ((3, -1), 0.30), ((5, -2), 0.25)],
    7: [((7, -1), 0.45), ((6, -1), 0.20), ((3, -2), -0.15)],
}
check_stability(links_coeffs)                  # asserts the VAR is stationary
G = dict_to_matrix(links_coeffs)               # (N, N, tau_max) coefficient stack

# ground-truth off-diagonal edge set (cause, effect, lag) — the answer key
GT_EDGES = {(i, j, -lag) for j, parents in links_coeffs.items()
            for (i, lag), c in parents if i != j}
print(f"SAVAR parent: N={N}, grid={ny}x{nx}, L={L}, tau_max={G.shape[2]}")
print(f"blob size {size}x{size}px; W rows sum to 1; W_plus shape {W_plus.shape}")
print(f"{len(GT_EDGES)} ground-truth causal edges (cause -> effect @lag):")
for i, j, lag in sorted(GT_EDGES):
    print(f"   X{i} -> X{j}  @lag {lag}")

# %%
# --- VIZ: publication-style true-graph figure --------------------------------
# Rendered by analysis/plot_true_graph.py -> results/plots/true_graph.png.
# Recompute:  python analysis/plot_true_graph.py
# In Jupyter, uncomment to display the saved figure:
# from IPython.display import Image
# Image(filename=os.path.join(PLOTS, "true_graph.png"))
print("true-graph figure:", os.path.join(PLOTS, "true_graph.png"))


# %% [markdown]
# ### 1.5 Seasonality as *seasonal modes* (not a global forcing)
#
# Seasonality is **not** an additive field applied uniformly to every pixel — that
# would contaminate all modes identically and just create a trivial common trend.
# Instead, **designated modes carry an intrinsic periodic component in their own
# driving signal**, so their footprint is a genuine spatial mode whose time-course
# is diurnal / weekly / annual:
#
# $$
# \varepsilon_i(t)\mathrel{+}= A\,\sin\!\Big(\tfrac{2\pi}{P_i}\,t+\phi_i\Big)
# \quad\text{for }i\in\text{SEAS\_MODES},
# $$
#
# added to that mode's innovation before the VAR recurrence. The cycle then
# **propagates to the mode's causal children** through $G$ (a seasonal parent
# imposes its period on its effects) — creating realistic **periodic confounding**
# between co-periodic modes, which is exactly the recovery stress we want to test.
# Knobs: `seas_modes`, `seas_periods`, `seas_amp`, `seas_phase`.
#
# ### 1.6 Optional external forcing (used in the regime rung)
#
# Piecewise step forcing on chosen modes over time windows $[t_1,t_2]$: a constant
# $f$ added through a forcing map $w_f$ during the window — models regime shifts.
# (Not part of this demo's toggle set; it lives in the regime rung.)

# %% [markdown]
# ### 1.8 Realistic demo world (GraphCast-like — the actual demo target)
#
# The clean linear-Gaussian parent (§1.1–1.7) is only the *first, easiest* pass. A
# linear-Gaussian world makes the optimal forecaster essentially linear, so it is
# a weak stand-in for GraphCast. The demo's headline world layers on five realism
# mechanisms — exactly the five toggles in `WorldConfig` (cell 0):
#
# 1. **Seasonality** — *seasonal modes*: designated modes carry an intrinsic
#    diurnal/annual oscillation in their driving signal (§1.5), propagating to
#    their causal children. Stresses **stationarity** + **periodic confounding**.
# 2. **Non-linearity** — in mode space, *before* the linear map: saturating
#    self-term $g(m)=(1-\alpha)m+\alpha\tanh(m)$ plus **bilinear advective**
#    coupling on lagged cross-edges (scaled by $\beta$, tanh-bounded). Stresses
#    the **linear CI test (ParCorr)**. The edge set is preserved by construction
#    (bilinear terms sit only on existing edges) — *that preservation is a claim
#    to be tested, not assumed.*
# 3. **Non-Gaussian innovations** — mode innovations drawn skew / heavy-tailed
#    (`ng_dist ∈ {skewnorm, t}`), standardized to unit variance. Stresses
#    **ParCorr's Gaussianity**.
# 4. **Moving mechanism** — footprints **advect** (spatial support drifts over
#    time). Stresses the **static-$\hat W$** assumption behind the projection
#    $\hat W X(t)$.
# 5. **Multivariate** — $C$ coupled channels per grid cell (block-diagonal
#    $\hat W = I_C\otimes W$), coupled only through a small cross-channel edge
#    set. Stresses **mode separability** (the activation arm can't split
#    co-located channels — the R3 degeneracy).
#
# **Test order (isolate before combine):** non-linearity and non-Gaussianity are
# tested **first and separately** (they most directly break the CI test), then
# moving mechanism, then multivariate, then the fully-combined world. Each
# mechanism needs its **own trained forecaster** (for `Ĝ_dyn`), so each test is a
# full generate → train → E1 → E4 rung (see the robustness appendix).
#
# `generate_world(CFG)` below composes **all** the toggles in a single mode-space
# recurrence; with the default CFG it reproduces the parent world of §1.1–1.7.

# %%
# ======================= CELL 1 — SELF-CONTAINED GENERATOR =====================
# Math lifted from the repo generators:
#   data_gen/instantiate_model.py    — base links_coeffs graph, W blobs, Σ_y noise
#   data_gen/generate_finecadence.py — _g_sat, _bilinear, draw_innovations,
#                                      seasonal-mode injection into eps_x
#   data_gen/generate_movmech.py     — footprint advection (here: periodic roll
#                                      drift; the repo generator ships MOVE=place)
#   data_gen/generate_multivar.py    — C-channel block-diagonal Ŵ = I_C ⊗ W,
#                                      cross-channel edges, per-channel emission

def generate_world(cfg=CFG, verbose=True):
    """Generate a SAVAR world composing all five realism toggles.

    Returns a dict with:
      reals    : list of cfg.n_real dicts {observations (L_obs, T) float32,
                 latent_states (NC, T) float32} — channel-major stacking when
                 n_channels > 1 (mode m = ch*N + node; pixel p = ch*L + px)
      W        : (NC, L_obs) effective block-diagonal mode map  I_C ⊗ W
      W_plus   : (L_obs, NC) its pseudo-inverse
      G        : (NC, NC, tau_max) ground-truth coefficient stack
      GT_EDGES : off-diagonal edge set {(cause, effect, -lag)}
      tau_max, spectral_radius, cfg
    """
    C = cfg.n_channels
    NC, L_obs = C * N, C * L
    total_T = cfg.T + cfg.burn

    # -- ground-truth graph: C within-channel copies + cross-channel edges ------
    # (generate_multivar.py; stacked index m = channel*N + node, channel-major)
    G_base = dict_to_matrix(links_coeffs)                  # (N, N, 2) parent graph
    tau_base = G_base.shape[2]
    cross = []      # (cause_node, cause_ch, effect_node, effect_ch, lag, coeff)
    if C >= 2:
        cross += [(0, 0, 0, 1, 1, 0.25),   # ch0 X0 -> ch1 X0 (same footprint)
                  (3, 0, 5, 1, 2, 0.20),   # ch0 X3 -> ch1 X5 (lag 2)
                  (2, 1, 6, 0, 1, 0.18)]   # ch1 X2 -> ch0 X6 (reverse direction)
    if C >= 3:
        cross += [(1, 1, 4, 2, 1, 0.22),   # ch1 X1 -> ch2 X4
                  (7, 2, 2, 0, 3, 0.15)]   # ch2 X7 -> ch0 X2 (closes a loop)
    cross = [(cn, cc, en, ec, lag, co * cfg.cross_scale)
             for (cn, cc, en, ec, lag, co) in cross]
    tau_max = max([tau_base] + [lag for (_, _, _, _, lag, _) in cross])

    G_ext = np.zeros((NC, NC, tau_max))
    for ch in range(C):                      # C independent within-channel copies
        G_ext[ch*N:(ch+1)*N, ch*N:(ch+1)*N, :tau_base] = G_base
    for (cn, cc, en, ec, lag, co) in cross:  # cross-channel couplings
        G_ext[ec*N + en, cc*N + cn, lag - 1] += co

    # linear-skeleton companion-matrix stability check (generate_multivar.py)
    top = np.hstack([G_ext[:, :, i] for i in range(tau_max)])
    bot = np.hstack([np.eye(NC * (tau_max - 1)), np.zeros((NC * (tau_max - 1), NC))])
    rho = float(np.max(np.abs(np.linalg.eigvals(np.vstack([top, bot])))))
    assert rho < 1.0, f"linear skeleton unstable (rho={rho:.3f}); lower cross_scale"

    gt_edges = {(c, e, -lag) for e in range(NC) for c in range(NC) if c != e
                for lag in range(1, tau_max + 1) if G_ext[e, c, lag - 1] != 0}

    # -- effective aggregation map: block-diagonal Ŵ = I_C ⊗ W ------------------
    W_blk = np.kron(np.eye(C), W_flat)                     # (NC, C*L)
    Wp_blk = np.linalg.pinv(W_blk)                         # (C*L, NC)

    # -- moving mechanism: footprints advect by periodic roll -------------------
    # (spec of generate_movmech.py; each mode drifts in its own direction, the
    #  drifted map is rebuilt every cfg.move_every steps and held in between)
    _Wcache = {}
    base_ang = np.deg2rad(cfg.move_angle_deg)

    def W_at(t):
        """(W_t, pinv(W_t)) at absolute step t; static unless cfg.move."""
        if not cfg.move:
            return W_blk, Wp_blk
        k = (t // cfg.move_every) * cfg.move_every
        if k not in _Wcache:
            Wt = np.zeros_like(W_flat)
            for i in range(N):
                th = base_ang + 2.0 * np.pi * i / N        # per-mode direction
                dy = int(round(cfg.move_speed * k * np.sin(th)))
                dx = int(round(cfg.move_speed * k * np.cos(th)))
                Wt[i] = np.roll(W[i], (dy, dx), axis=(0, 1)).ravel()
            Wtb = np.kron(np.eye(C), Wt)
            _Wcache[k] = (Wtb, np.linalg.pinv(Wtb))
        return _Wcache[k]

    # -- non-linearity (generate_finecadence.py) --------------------------------
    def g_sat(m):
        """Saturating self-term; exact identity when nl_alpha == 0."""
        return (1.0 - cfg.nl_alpha) * m + cfg.nl_alpha * np.tanh(m)

    # lagged cross-edge list for the bilinear coupling — EXISTING edges only,
    # so the toggle preserves the edge set by construction
    cross_lag_edges = [(e, c, lag, G_ext[e, c, lag - 1])
                       for e in range(NC) for c in range(NC)
                       for lag in range(1, tau_max + 1)
                       if e != c and G_ext[e, c, lag - 1] != 0]

    # -- non-Gaussian innovations (generate_finecadence.py, verbatim) -----------
    def draw_innovations(rng, shape):
        """Zero-mean, UNIT-VARIANCE innovations (stationarity untouched)."""
        if cfg.ng_dist == "gaussian":
            return rng.standard_normal(shape)
        if cfg.ng_dist == "t":
            x = rng.standard_t(cfg.ng_df, size=shape)
            return x / np.sqrt(cfg.ng_df / (cfg.ng_df - 2.0))
        if cfg.ng_dist == "skewnorm":
            a = cfg.ng_skew
            delta = a / np.sqrt(1.0 + a * a)
            z0 = np.abs(rng.standard_normal(shape))
            z1 = rng.standard_normal(shape)
            x = delta * z0 + np.sqrt(1.0 - delta * delta) * z1
            mean = delta * np.sqrt(2.0 / np.pi)
            var = 1.0 - 2.0 * delta * delta / np.pi
            return (x - mean) / np.sqrt(var)
        raise ValueError(f"unknown ng_dist={cfg.ng_dist!r}")

    # -- the recurrence (all toggles composed) -----------------------------------
    reals = []
    for seed in range(cfg.n_real):
        rng = np.random.default_rng(cfg.seed0 + seed)
        eps_x = np.sqrt(cfg.lam) * draw_innovations(rng, (NC, total_T))
        if cfg.seas_modes:               # seasonal modes: sinusoid in eps_x (§1.5)
            tv = np.arange(total_T)
            for k, mi in enumerate(cfg.seas_modes):
                P = cfg.seas_periods[k % len(cfg.seas_periods)]
                ph = cfg.seas_phase[k % len(cfg.seas_phase)]
                s = cfg.seas_amp * np.sin(2.0 * np.pi * tv / P + ph)
                for ch in range(C):      # seasonal node in every channel
                    eps_x[ch*N + mi] += s
        eps_y = np.sqrt(cfg.dy_scale) * rng.standard_normal((L_obs, total_T))

        # coloured noise field η(t) = W_t^+ ε_x(t) + ε_y(t)   (§1.4)
        if cfg.move:
            data = eps_y
            for t in range(total_T):
                data[:, t] += W_at(t)[1] @ eps_x[:, t]
        else:
            data = Wp_blk @ eps_x + eps_y                  # (L_obs, total_T)

        M = np.zeros((NC, total_T))      # measured mode amplitudes Z(t) = W_t X(t)
        for t in range(tau_max):
            M[:, t] = W_at(t)[0] @ data[:, t]
        for t in range(tau_max, total_T):
            contrib = np.zeros(NC)
            for lag in range(1, tau_max + 1):              # §1.2 / §1.3 recurrence
                contrib += G_ext[:, :, lag - 1] @ g_sat(M[:, t - lag])
            if cfg.nl_beta != 0.0:       # bilinear advective coupling (bounded)
                q = np.zeros(NC)
                m1 = M[:, t - 1]
                for e, c, lag, co in cross_lag_edges:
                    q[e] += co * M[c, t - lag] * m1[e]
                contrib += cfg.nl_beta * np.tanh(q)
            Wt, Wpt = W_at(t)
            data[:, t] += Wpt @ contrib                    # emit through W_t^+
            M[:, t] = Wt @ data[:, t]

        reals.append(dict(observations=data[:, cfg.burn:].astype(np.float32),
                          latent_states=M[:, cfg.burn:].astype(np.float32)))
        if verbose and (seed + 1) % 10 == 0:
            print(f"  generated {seed + 1}/{cfg.n_real} realisations")

    if verbose:
        toggles = [n for n, on in [("seasonal", bool(cfg.seas_modes)),
                                   ("nonlinear", cfg.nl_alpha or cfg.nl_beta),
                                   ("non-gaussian", cfg.ng_dist != "gaussian"),
                                   ("moving", cfg.move),
                                   (f"multivar C={C}", C > 1)] if on]
        print(f"world: NC={NC} modes, L_obs={L_obs}, T={cfg.T}, "
              f"{len(gt_edges)} true edges, rho={rho:.3f}, "
              f"toggles: {toggles or ['none (linear-Gaussian parent)']}")
    return dict(reals=reals, W=W_blk, W_plus=Wp_blk, G=G_ext, GT_EDGES=gt_edges,
                tau_max=tau_max, spectral_radius=rho, cfg=cfg)

# Gated generation: nothing heavy auto-executes. Downstream cells consume
# REALS / W_TRUE / GT_EDGES / N_MODES / L_OBS, all rewired to the generated world.
if RUN:
    WORLD = generate_world(CFG)
    REALS = WORLD["reals"]
    W_TRUE = WORLD["W"]                        # (NC, L_obs) block-diagonal truth
    GT_EDGES = WORLD["GT_EDGES"]               # stacked-index answer key
    N_MODES, L_OBS = W_TRUE.shape
else:
    N_MODES, L_OBS = CFG.n_channels * N, CFG.n_channels * L
    W_TRUE = np.kron(np.eye(CFG.n_channels), W_flat)
    print("[skipped: RUN=False] generate_world(CFG) builds the full world "
          f"({CFG.n_real} realisations, NC={N_MODES}, L_obs={L_OBS}); "
          "default CFG = clean linear-Gaussian parent.")


# %% [markdown]
# ## 2. The forecaster (MeshGNN)
#
# Train a graph neural net (~890K params) to predict the next frame from the past
# 3. It learns the dynamics but **never sees the causal graph** — it is the black
# box that steps 5–6 interpret, a stand-in for a real weather model like GraphCast.
# It runs on a deliberately **heterogeneous** multi-scale mesh (local 8-neighbour
# edges plus long-range hub edges), so some nodes are structurally
# better-connected — the "grid-lock" substrate a real model has. The cell loads a
# trained checkpoint onto CPU (`map_location="cpu"`, so no GPU needed).
#
# **Per-world training.** Training is *not* done in this notebook. For the default
# CFG the parent checkpoint at `CKPT` applies; for any toggled world, save the
# `generate_world` output as realisations, split, and train with
# `train/gnn/gnn_forecaster.py` (multichannel worlds: `gnn_forecaster_multivar.py`),
# then point `CKPT` (cell 0) at the new `best.pt`.

# %%
# --- load the trained forecaster (real logic from the E1/E4 extract functions) ---
import torch
from gnn_forecaster import MeshGNN            # train/gnn/gnn_forecaster.py

def load_forecaster(ckpt_path=CKPT, ny=ny, nx=nx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshGNN(ny=ny, nx=nx).to(device)      # defaults: gcn mesh, k=3, hidden=256
    ck = torch.load(ckpt_path, map_location=device)   # CPU-safe load
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck, device

if RUN:
    model, ck, DEVICE = load_forecaster()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MeshGNN loaded: {n_params:,} params (~890K), "
          f"epoch {ck['epoch']}, val_rmse {ck['val_rmse']:.4f}")
    print(f"mesh: {model.n_hubs} hubs, {model.edge_src.shape[0]} directed edges")
else:
    print("[skipped: RUN=False] load_forecaster() reads", CKPT)
    print("MeshGNN(ny=50, nx=50): gcn mesh, k=3 past frames, hidden=256, ~890K params")


# %% [markdown]
# ## 3. E1 — mode discovery (candidates)
#
# From data + the frozen model's internals only (no `W`, no `Z`), recover
# candidate mode maps `Ŵ` — i.e. find the hidden regions (the blobs) unsupervised.
# The honest operators are varimax-rotated PCA and k-means, run on either raw
# pixels or the GNN's per-node activation field (DMD is a third). We also build
# deliberately **corrupted** anchors (merge / coarsen / split / shift / blur of
# the true `W`) so the selector in step 6 has a spread of quality to rank —
# keeping good *and* bad candidates is the point. The builder functions below are
# lifted from `sae/discover_modes.py` and now consume the **generated** world's
# held-out discovery realisations.

# %%
# --- E1 candidate builders (verbatim logic from sae/discover_modes.py) -----------
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

C0 = 12          # initial component/cluster count (pruned to coherent modes)
COH_MIN = 0.25   # coherence floor for keeping a discovered mode
SEED = 0

def varimax(Phi, gamma=1.0, q=200, tol=1e-8):
    p, k = Phi.shape
    R = np.eye(k); dd = 0.0
    for _ in range(q):
        Lm = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lm ** 3 - (gamma / p) * Lm @ np.diag((Lm ** 2).sum(0))))
        R = u @ vt
        d_new = s.sum()
        if d_new < dd * (1 + tol):
            break
        dd = d_new
    return Phi @ R

def loading_to_footprint(load):
    """signed loading (L_OBS,) -> nonneg, L1-normalized footprint row."""
    if load[np.argmax(np.abs(load))] < 0:
        load = -load
    fp = np.clip(load, 0, None)
    fp[fp < 0.05 * fp.max()] = 0.0
    s = fp.sum()
    return fp / s if s > 0 else fp

def cand_varimax(field):
    """varimax-rotated PCA on a (R, L_OBS, T) scalar field -> footprints (C0, L_OBS)."""
    X = field.transpose(0, 2, 1).reshape(-1, L_OBS)       # (R*T, L_OBS)
    X = X - X.mean(0)
    pca = PCA(n_components=C0, random_state=SEED).fit(X)
    loads = (pca.components_ * np.sqrt(pca.explained_variance_)[:, None]).T
    rot = varimax(loads)                                  # (L_OBS, C0)
    return np.stack([loading_to_footprint(rot[:, c]) for c in range(C0)])

def cand_kmeans(field):
    """k-means on z-scored per-node time courses -> hard-assignment footprints."""
    Xn = field.transpose(1, 0, 2).reshape(L_OBS, -1)      # (L_OBS, R*T)
    Xn = (Xn - Xn.mean(1, keepdims=True)) / (Xn.std(1, keepdims=True) + 1e-9)
    emb = PCA(n_components=50, random_state=SEED).fit_transform(Xn)
    km = KMeans(n_clusters=C0, n_init=4, random_state=SEED).fit(emb)
    What = np.stack([(km.labels_ == c).astype(np.float64) for c in range(C0)])
    return What / np.maximum(What.sum(1, keepdims=True), 1)

def build_corrupted(W_true):
    """deliberately-wrong anchors that populate the quality axis for the selector.

    Built in base (single-channel) space from the true W; lifted to the
    block-diagonal (channel x space) layout via kron when n_channels > 1."""
    def norm(M): return M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    out = {}
    out["merge01"] = norm(np.vstack([(W_true[0] + W_true[1])[None], W_true[2:]]))
    out["coarse4"] = norm(np.stack([W_true[2*i] + W_true[2*i+1] for i in range(N // 2)]))
    out["shift5"]  = norm(np.stack([np.roll(W_true[j].reshape(ny, nx), 5, axis=1)
                                    .reshape(L) for j in range(N)]))
    out["blur"]    = norm(np.stack([gaussian_filter(W_true[j].reshape(ny, nx), sigma=6)
                                    .reshape(L) for j in range(N)]))
    if CFG.n_channels > 1:                                 # lift: Ŵ_blk = I_C ⊗ Ŵ
        out = {k: np.kron(np.eye(CFG.n_channels), v) for k, v in out.items()}
    return out

# In the real run the honest builders consume a field extracted from the generated
# realisations (S_PIX = raw pixels; S_ACT = per-node GNN activations via a forward
# hook on model.layers[-1]). Here we assemble the candidate dict; `oracle` = true W.
if RUN:
    DISC = list(range(len(REALS) - 4, len(REALS)))        # held-out discovery reals
    S_PIX = np.stack([REALS[r]["observations"].astype(np.float32)
                      for r in DISC])                     # (R, L_OBS, T)
    CANDS = {"vmax_pix": cand_varimax(S_PIX),
             "km_pix":   cand_kmeans(S_PIX)}
    CANDS.update(build_corrupted(W_flat))
    CANDS["oracle"] = W_TRUE / W_TRUE.sum(1, keepdims=True)
    print("candidates:", {k: v.shape[0] for k, v in CANDS.items()})
else:
    print("[skipped: RUN=False] E1 builders would run on the generated pixel/"
          "activation field; honest ops: vmax_act/vmax_pix/km_act/km_pix/dmd_act, "
          "plus corrupted anchors merge01/coarse4/split7/fine16/shift5/diag8/blur, "
          "plus oracle.")


# %% [markdown]
# ## 4. Integration view — Ĝ_int (PCMCI on projected mode series)
#
# The first, observational read of the wiring: whose past predicts whose future in
# the *recordings*. For each candidate map `Ŵ` we project the pixel movie onto its
# modes (`Ŵ @ obs`) to get mode time-series, then run **PCMCI+** causal discovery
# (default CI test `RobustParCorr`, `tau_max` = max true lag, `pc_alpha = 0.05`).
# Per realisation this yields a directed edge set; the consensus graph keeps a
# pair detected in ≥50% of realisations. This is the exact int-stage logic from
# `pcmci/e4_agreement.py` — pure observation, data only; the model is never touched.

# %%
# --- PCMCI+ on Ŵ-pooled series (verbatim logic from e4_agreement.py int stage) ---
PC_ALPHA = 0.05
TAU_MAX = max(-lag for _, _, lag in GT_EDGES)      # max true lag (parent: 2)

def detect(graph):
    """tigramite graph array -> set of directed (cause, effect, tau) edges."""
    Nn, _, T1 = graph.shape
    return sorted((c, e, tau) for c in range(Nn) for e in range(Nn) if c != e
                  for tau in range(1, T1) if graph[c, e, tau] == "-->")

def pcmci_edges(series, tau_max=TAU_MAX, pc_alpha=PC_ALPHA):
    """PCMCI+ on a (T, C) mode-series array -> directed edge set."""
    from tigramite.data_processing import DataFrame
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.robust_parcorr import RobustParCorr
    pc = PCMCI(dataframe=DataFrame(series), cond_ind_test=RobustParCorr(), verbosity=0)
    res = pc.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    return detect(res["graph"])

def consensus_pairs(dets_per_real, C, cons_frac=0.5):
    """pair-level consensus: keep (c,e) present in >= cons_frac of realisations."""
    cnt = np.zeros((C, C))
    for det in dets_per_real.values():
        for (c, e) in {(c, e) for (c, e, _) in det}:
            cnt[c, e] += 1
    thr = cons_frac * len(dets_per_real)
    return sorted((c, e) for c in range(C) for e in range(C)
                  if c != e and cnt[c, e] >= thr)

def integration_graph(What, reals, n_real=24):
    """Ĝ_int for one candidate: PCMCI+ per real on Ŵ@obs, then consensus pairs."""
    dets = {}
    for ri in range(min(n_real, len(reals))):
        obs = reals[ri]["observations"].astype(np.float64)            # (L_OBS, T)
        dets[ri] = pcmci_edges((What @ obs).T)                        # series (T, C)
    return consensus_pairs(dets, What.shape[0]), dets

if RUN:
    G_INT = {}
    for name, What in CANDS.items():
        G_INT[name], _ = integration_graph(What, REALS)
        print(f"  Ĝ_int[{name}]: {len(G_INT[name])} consensus edges")
else:
    print("[skipped: RUN=False] integration_graph() runs PCMCI+ (RobustParCorr, "
          f"tau_max={TAU_MAX}, alpha={PC_ALPHA}) per realisation, ~minutes/candidate.")


# %% [markdown]
# ## 5. Dynamics view — Ĝ_dyn (perturb the model)
#
# The second, independent read — from *behavior*, not observation: poke the
# trained forecaster on one mode and watch which modes respond. We inject a small
# `W`-free impulse (from `pinv(Ŵ)`) into one mode's slot, roll the forecaster
# forward teacher-forced, and record which modes respond (`R = Ŵ @ ΔX`). An edge
# is kept when the integral response `Σ_τ|R|` beats a pixel-permutation null
# (`alpha=0.01`, 1000 perms). Different blind spots than step 4: this is an
# experiment *on the model*. Code is the dyn-stage core of `pcmci/e4_agreement.py`,
# with perturbation windows drawn from the generated realisations.

# %%
# --- teacher-forced response graph (verbatim logic from e4_agreement.py dyn stage) --
N_WIN, N_STEPS, N_PERM, ALPHA_DYN, SIGMA = 240, 24, 1000, 0.01, 1.0

def dynamics_graph(What, model, traj, pix_std, device):
    """Ĝ_dyn for one candidate via teacher-forced impulse responses."""
    C, K = What.shape[0], 3
    Wp = np.linalg.pinv(What)                                  # (L_OBS, C) W-free impulse
    patterns = Wp / np.maximum(np.abs(Wp).max(0, keepdims=True), 1e-12)
    imps = (SIGMA * pix_std * patterns.T).astype(np.float32)   # (C, L_OBS)

    @torch.no_grad()
    def batched_model(x_np, BS=48):
        outs = []
        for i in range(0, len(x_np), BS):
            xb = torch.from_numpy(x_np[i:i + BS]).to(device)
            outs.append(model(xb)[:, 0].cpu().numpy())
        return np.concatenate(outs, 0)

    @torch.no_grad()
    def teacher_dX():
        mean_dX = np.zeros((C, N_STEPS, L_OBS))
        delta = np.zeros((C, len(traj), K, ny, nx), dtype=np.float32)
        for j in range(C):
            delta[j, :, -1] = imps[j].reshape(ny, nx)
        for s in range(N_STEPS):
            base = traj[:, s:s + K]
            y_base = batched_model(base)
            for j in range(C):
                dy = batched_model(base + delta[j]) - y_base
                mean_dX[j, s] += dy.reshape(len(traj), -1).sum(0)
                delta[j] = np.concatenate([delta[j, :, 1:], dy[:, None]], axis=1)
        return mean_dX / len(traj)

    mean_dX = teacher_dX()
    R = np.einsum("il,jtl->ijt", What, mean_dX)                # (C, C, S)
    rng = np.random.default_rng(1)                             # pixel-permutation null
    null_int = np.stack([np.abs(np.einsum("il,jtl->ijt",
                        What[:, rng.permutation(L_OBS)], mean_dX)).sum(2)
                        for _ in range(N_PERM)])
    th_int = np.quantile(null_int, 1 - ALPHA_DYN, axis=0)
    s_int = np.abs(R).sum(2)
    return sorted((j, i) for i in range(C) for j in range(C)
                  if i != j and s_int[i, j] > th_int[i, j])

if RUN:
    # The demo dyn arm drives the single-field MeshGNN; multichannel worlds need
    # the multivar forecaster (train/gnn/gnn_forecaster_multivar.py) instead.
    assert CFG.n_channels == 1, "dyn-arm demo path assumes n_channels=1"
    # frame movies (T, ny, nx) from the generated realisations
    frames = [r["observations"].T.reshape(-1, ny, nx) for r in REALS[:10]]
    pix_std = float(np.concatenate([f.reshape(len(f), -1) for f in frames]).std())
    rng = np.random.default_rng(0)
    win = [(i % len(frames),
            int(rng.integers(0, frames[i % len(frames)].shape[0] - 3 - N_STEPS)))
           for i in range(N_WIN)]
    traj = np.stack([frames[r][t0:t0 + 3 + N_STEPS] for r, t0 in win]).astype(np.float32)
    G_DYN = {name: dynamics_graph(What, model, traj, pix_std, DEVICE)
             for name, What in CANDS.items()}
    for name in G_DYN:
        print(f"  Ĝ_dyn[{name}]: {len(G_DYN[name])} edges")
else:
    print("[skipped: RUN=False] dynamics_graph() teacher-forces the forecaster "
          f"({N_WIN} windows x {N_STEPS} steps) + {N_PERM}-perm null per candidate.")


# %% [markdown]
# ## 6. PX selector — pool-crossed agreement (the answer-key-free signal)
#
# **PX(A) = mean over B≠A of pair-F1( Ĝ_int(A), Ĝ_dyn(B) )** — each candidate's
# observational wiring scored against the *other* candidates' dynamics wiring.
# "Crossed" so no candidate grades its own homework; high PX ⇒ the two independent
# lenses concur ⇒ a faithful decomposition, computed with **no answer key**. It
# picks the best mode set and flags the bad ones. To compare candidates we first
# map each to true-mode space via **behavior matching** (`|corr(Ŵ@obs, Z)|`), then
# check whether PX ranks the (secretly known) truth-F1 — the headline
# `Spearman(PX, truth-F1) ≥ 0.8`. Code mirrors the cal stage of
# `pcmci/e4_agreement.py`, reading `Z` from the generated world's latent states.

# %%
# --- behavior matching, cross-agreement, PX (verbatim logic from e4_agreement.py cal) --
from scipy.optimize import linear_sum_assignment
from scipy import stats

MATCH_R = 0.3          # keep a candidate<->mode match iff mean |corr| >= 0.3
GT_PAIRS = {(i, j) for i, j, _ in GT_EDGES}

def behavior_mapping(What, reals, n_match=6):
    """Hungarian on mean |corr(Ŵ@obs, true Z)| -> {cand_var: true_mode}."""
    C = What.shape[0]
    M = np.zeros((C, N_MODES))
    for ri in range(n_match):
        d = reals[ri]
        P = What @ d["observations"].astype(np.float64)
        Z = d["latent_states"].astype(np.float64)
        Pc, Zc = P - P.mean(1, keepdims=True), Z - Z.mean(1, keepdims=True)
        den = np.sqrt((Pc**2).sum(1))[:, None] * np.sqrt((Zc**2).sum(1))[None]
        M += np.abs(Pc @ Zc.T / np.maximum(den, 1e-12))
    M /= n_match
    a, b = linear_sum_assignment(-M)
    return {int(i): int(j) for i, j in zip(a, b) if M[i, j] >= MATCH_R}

def set_f1(A, B):
    A, B = set(map(tuple, A)), set(map(tuple, B))
    tp = len(A & B)
    p = tp / len(B) if B else 0.0
    r = tp / len(A) if A else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def to_mode_space(pairs, mp):
    return {(mp[c], mp[e]) for (c, e) in pairs if c in mp and e in mp}

def truth_f1_pair(dets_per_real, mp):
    """behavior-matched, Hungarian-strict pair-level truth-F1, micro-averaged."""
    tp = fp = fn = 0
    for det in dets_per_real.values():
        mapped = {(mp[c], mp[e]) for (c, e, _) in det if c in mp and e in mp}
        un = len({(c, e) for (c, e, _) in det if c not in mp or e not in mp})
        tp += len(GT_PAIRS & mapped); fp += len(mapped - GT_PAIRS) + un
        fn += len(GT_PAIRS - mapped)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def pool_crossed_px(G_int, G_dyn, BEH):
    """PX(A) = mean_{B!=A} F1( G_int(A) in mode space, G_dyn(B) in mode space )."""
    names = [n for n in G_int if len(BEH.get(n, {})) >= 4]      # need >=4 matches
    MI = {n: to_mode_space(G_int[n], BEH[n]) for n in names}
    MD = {n: to_mode_space(G_dyn[n], BEH[n]) for n in names}
    PX = {}
    for n in G_int:
        if n in names:
            PX[n] = float(np.mean([set_f1(MI[n], MD[b]) for b in names if b != n]))
        else:
            PX[n] = 0.0                                          # unmatched -> 0
    return PX

if RUN:
    BEH = {n: behavior_mapping(W_, REALS) for n, W_ in CANDS.items()}
    # per-candidate truth-F1 needs the per-real int detections (kept from step 4):
    DETS = {n: integration_graph(CANDS[n], REALS)[1] for n in CANDS}
    TRUTH_F1 = {n: truth_f1_pair(DETS[n], BEH[n]) for n in CANDS}
    PX = pool_crossed_px(G_INT, G_DYN, BEH)
    names = sorted(CANDS, key=lambda n: -TRUTH_F1[n])
    print(f"{'candidate':10s} {'PX':>6s} {'truth-F1':>9s}")
    for n in names:
        print(f"{n:10s} {PX[n]:6.3f} {TRUTH_F1[n]:9.3f}")
    px_arr = np.array([PX[n] for n in names]); f1_arr = np.array([TRUTH_F1[n] for n in names])
    print(f"\nSpearman(PX, truth-F1) = {stats.spearmanr(px_arr, f1_arr).statistic:+.3f}"
          "   [selector passes at >= 0.8]")
else:
    print("[skipped: RUN=False] PX = mean_{B!=A} F1(Ĝ_int(A), Ĝ_dyn(B)) after behavior "
          "matching; headline check is Spearman(PX, truth-F1) >= 0.8 (no answer key used).")

# %%
# --- VIZ: PX vs truth-F1 scatter — drawing commented for headless ------------
# if RUN:
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.scatter(px_arr, f1_arr, c="#4c72b0")
#     for n, x, y in zip(names, px_arr, f1_arr): ax.annotate(n, (x, y), fontsize=7)
#     ax.set_xlabel("PX (pool-crossed agreement, truth-free)"); ax.set_ylabel("truth-F1")
#     ax.set_title("PX ranks accuracy"); plt.tight_layout(); plt.show()
print("PX-vs-truthF1 scatter defined (uncomment to draw in Jupyter).")


# %% [markdown]
# ## 7. E4-final — the trust dial (agreement → predicted accuracy)
#
# The deliverable. Pool the `(PX, truth-F1)` points from *every* pre-registered
# rung (parent, overlap, rollout, scale, multivar) and fit **one line**. That line
# turns an observed agreement into a *predicted* accuracy for a model you cannot
# check — a calibrated confidence read-out (`accuracy ≈ 1.48·PX + 0.13`, ±0.13 on
# the current pool). The cell mirrors `analysis/e4_final_calibration.py`: it reads
# each rung's E4 output, pools the pairs, and fits the dial with Spearman /
# linregress. (This step still reads the precomputed rung outputs under
# `results/` — a fresh generated world would contribute one more row to the pool
# after its own E1→E4 battery.)

# %%
# --- pooled cross-rung calibration (verbatim logic from e4_final_calibration.py) -----
import collections

# Each pre-registered rung re-runs steps 1-6 with a different data tag; the parent is
# the first row. E4 writes results/litext_e4_agreement{tag}.npy per rung.
RUNGS = [("parent", ""), ("R1_overlap", "_overlap02"), ("R5_rollout", "_overlap02_rollout"),
         ("R4_scale", "_scale24"), ("R3_multivar", "_multivar")]

def build_calibration(rungs=RUNGS):
    """pool (PX, truth-F1) across rungs, fit the trust-dial line."""
    recs = []      # (rung, cand, PX, agree, f1_pair)
    for name, tag in rungs:
        f = os.path.join(ROOT, "results", f"litext_e4_agreement{tag}.npy")
        if not os.path.exists(f):
            print(f"[skip] {name}: {f} missing"); continue
        d = np.load(f, allow_pickle=True).item()
        px, rows = d["px"], d["rows"]
        for cand in px:
            if cand in rows and np.isfinite(rows[cand].get("f1_pair", np.nan)):
                recs.append((name, cand, float(px[cand]),
                             float(rows[cand].get("agree", np.nan)),
                             float(rows[cand]["f1_pair"])))
    PX = np.array([r[2] for r in recs]); F1 = np.array([r[4] for r in recs])
    sp = stats.spearmanr(PX, F1).statistic
    sl, ic, rr, _, _ = stats.linregress(PX, F1)
    resid_std = (F1 - (sl * PX + ic)).std()
    return recs, dict(spearman=sp, slope=sl, icept=ic, r2=rr**2, resid_std=resid_std)

if RUN:
    recs, pooled = build_calibration()
    print(f"pooled over {len(set(r[0] for r in recs))} rungs, {len(recs)} candidates:")
    print(f"  TRUST DIAL:  accuracy ~= {pooled['slope']:.2f} * PX + {pooled['icept']:.2f}"
          f"  (+/- {pooled['resid_std']:.2f})")
    print(f"  Spearman = {pooled['spearman']:+.3f}   R2 = {pooled['r2']:.3f}")
else:
    print("[skipped: RUN=False] build_calibration() pools every rung's E4 output and")
    print("fits accuracy ~= 1.48*PX + 0.13 (+/-0.13); the parent is the first rung.")

# %%
# --- VIZ: the trust-dial figure (v2, 2026-07-16) -------------------------------
# Rendered by analysis/plot_trust_dial.py -> results/plots/trust_dial.png.
# v2 adds the four robustness worlds as an OUT-OF-SAMPLE test of the Jul-8 dial,
# then refits pooled (results/litext_e4_final_calibration_v2.npy).
# In Jupyter, uncomment to display the saved figure:
# from IPython.display import Image
# Image(filename=os.path.join(PLOTS, "trust_dial.png"))
print("trust-dial figure:", os.path.join(PLOTS, "trust_dial.png"))
print("""
TAKEAWAY: from a forecaster we cannot check, PX (measured with NO answer key)
predicts recovery accuracy. Out-of-sample on the 4 robustness worlds the
RANKING transfers cleanly (Spearman +0.81, Pearson +0.93) but the ABSOLUTE
calibration shifted: new points sit ~+0.14 F1 above the Jul-8 line (only 8/52
inside its +/-0.13 band), so v2 refits pooled over 9 worlds:
accuracy ~ 1.76*PX + 0.13 (Spearman +0.79, R2 0.72, residual +/-0.16).
Caveat: the offset is confounded with the CI-test switch (old rungs ParCorr,
new rungs RobustParCorr) -- quote the dial as rank-reliable with +/-0.16
absolute error, not a universal constant.""")


# %% [markdown]
# ## Appendix — robustness: does the method survive realistic mechanisms?
#
# The linear-Gaussian parent is the cleanest signal; the five toggles of cell 0
# each stress one recipe assumption on its own before combining. **The CI-test
# default is now `RobustParCorr`** (rank-transformed ParCorr) everywhere — it
# matched plain ParCorr on the skewnorm worlds (Jaccard 0.909, one fewer false
# edge) and is strictly safer under non-Gaussian marginals at negligible cost.
#
# Every mechanism stresses a specific assumption in the recipe. The **primary
# test is always the same**: does `Spearman(PX, truth-F1) ≥ 0.8` still hold? Each
# row adds one **mechanism-specific diagnostic** that localizes *why* if it
# fails. All require a forecaster trained on that world (needed for `Ĝ_dyn`), so
# each row is a full generate → train → E1 → E4 rung — one `WorldConfig` per row.
#
# | mechanism | assumption stressed | how it could break PX/F1 | mechanism-specific test | pass criterion |
# |---|---|---|---|---|
# | **Non-linearity** (test 1st) | ParCorr = *linear* CI test | linear CI misses curved dependencies → missing/spurious edges in Ĝ_int | ablate CI test: **ParCorr vs GPDC / CMIknn**; check edge-set preservation vs true G | Spearman ≥ 0.8 **and** ParCorr-F1 ≈ nonlinear-CI-F1 (confirms "edge-preserving") |
# | **Non-Gaussianity** (test 1st) | ParCorr assumes Gaussian residuals | miscalibrated p-values → false edges under skew/heavy tails | **ParCorr vs rank/robust CI**; check CI test false-positive rate on null pairs | Spearman ≥ 0.8; FP rate near α (calibrated) |
# | **Seasonality** | stationarity; independent innovations | shared periodic trend → all modes co-vary → spurious common-cause edges | PX/F1 **with vs without deseasonalization** (regress out $s(t)$) | Spearman ≥ 0.8; quantify whether detrend is a *required* preprocessing step |
# | **Moving mechanism** | static footprint `Ŵ` | advecting region → fixed `Ŵ` yields time-varying mode mixtures → discovery + both graphs degrade | E1 **coherence** on movmech; **static vs time-windowed `Ŵ`**; PX/F1 both ways | Spearman ≥ 0.8; discovery stays coherent (or windowing rescues it) |
# | **Multivariate** | mode separability; activation arm | one hidden/node can't split co-located channels (R3) | PX/F1 **pixel-side vs activation-side**; per-channel recovery | pixel-side Spearman ≥ 0.8 (R3 got +0.943); acts-arm failure documented |
# | **All combined** | all of the above at once | interacting failures | full PX/F1 on the combined world | Spearman ≥ 0.8 = method survives realistic regime |
#
# **Cross-cutting checks that ride along every row:**
# - **CI-test ablation** (ParCorr vs GPDC/CMIknn) is the single most important new
#   experiment — nonlinearity and non-Gaussianity both attack the *same* linear-CI
#   assumption, so this ablation covers two rows at once.
# - **Truth-free alignment** (align candidates to each other, not true Z) must hold
#   under each mechanism — it's the precondition for the GraphCast claim.
# - **Preconditions recheck** per world: dynamics *live* (else PX inapplicable, cf.
#   R6), pool *resolution-homogeneous* (cf. R4).
#
# ### Results (run 2026-07-15/16; `out/orchestrate_robust.sh`, results committed)
#
# | mechanism | world tag | PX Spearman | Pearson | verdict |
# |---|---|---|---|---|
# | **Non-Gaussianity** (skewnorm innovations) | `_linskew` | **+0.909** | +0.946 | **PASS** |
# | **All combined** (realistic regime) | `_finecadence` | **+0.869** | +0.951 | **PASS** |
# | **Non-linearity** (bilinear, α = 0.5) | `_nlgauss` | +0.624 | +0.972 | MISS (rank only — see reading) |
# | **Seasonality, raw** | `_overlapseas_raw` | +0.132 | +0.333 | MISS |
# | **Seasonality, deseasonalized** | `_overlapseas_deseas` | +0.769 | +0.789 | near-miss |
#
# **Mechanism diagnostics:**
# - **Non-linearity CI ablation** (`pcmci/ci_test_ablation.py`): ParCorr ≡
#   RobustParCorr (Jaccard 1.000, F1 0.815 vs truth); CMIknn reaches F1 0.957 by
#   dropping 4 false-positive edges while keeping recall. The linear CI test is
#   **edge-recall-preserving** under this non-linearity, at a cost of ~4 spurious
#   edges — and CMIknn costs 51,652 s vs 1 s, so RobustParCorr stays the default.
# - **Non-Gaussianity**: RobustParCorr F1 0.923 vs ParCorr 0.889 on the skewnorm
#   world; **FP calibration near-nominal for both** (α = 0.05 → observed
#   0.060–0.065 on 200 cross-realisation null pairs) — ParCorr p-values are not
#   badly miscalibrated under skew. Pass criterion met.
#
# **Reading the misses** (per-candidate diagnostics, dyn *live* on all four worlds
# so PX applies everywhere):
# - **`nlgauss`** — the mid-pool is 8 candidates within **0.017 of PX** spanning
#   **0.29 of truth-F1**: in-cluster ranking is noise (excluding the two PX = 0
#   degenerates, Spearman drops to +0.382). But the *top-pick cost is small*:
#   PX picks `shift5` (F1 0.826) vs true best `blur` (0.883), −0.057. Under
#   strong non-linearity PX still **screens** out bad decompositions
#   (coarse4/diag8/fine16) and its calibration line holds (Pearson +0.97), but it
#   is not a fine **ranker** among near-ties — the E2 lesson, now for PX.
# - **Seasonality** — deseasonalization is a **required preprocessing step**
#   (+0.132 raw → +0.769 deseasonalized). The remaining near-miss is two
#   placements: `dmd_act` over-ranked (PX 1st, truth 7th; top-pick cost 0.760 vs
#   0.971) and `km_pix` under-ranked (PX 12th, truth 8th).
#
# **Takeaway:** the selector + trust dial survive non-Gaussian innovations and
# the fully combined realistic world; seasonal worlds demand deseasonalized
# inputs; under strong non-linearity PX remains a reliable screen with a
# calibrated accuracy line, but near-tie ranking is unreliable.

# %%
# ===== APPENDIX — robustness scorecard (post-processing of committed results) ==
# Loads results/litext_e4_agreement_<tag>.npy from the 2026-07-15/16 robustness
# pipeline (out/orchestrate_robust.sh) and renders the scorecard + PX-vs-truth
# scatters. No heavy compute; runs with RUN=False.
from scipy.stats import spearmanr

RES = os.path.join(ROOT, "results")
ROBUST_TAGS = [
    ("linskew",            "non-Gaussian (skewnorm)",  "PASS"),
    ("finecadence",        "all combined",             "PASS"),
    ("nlgauss",            "non-linear (alpha=0.5)",   "MISS (rank)"),
    ("overlapseas_raw",    "seasonal, raw",            "MISS"),
    ("overlapseas_deseas", "seasonal, deseasonalized", "near-miss"),
]

print(f"{'world':26s} {'PX Spearman':>12s} {'Pearson':>8s}  [bar >= 0.8]")
robust = {}
for tag, label, note in ROBUST_TAGS:
    d = np.load(os.path.join(RES, f"litext_e4_agreement_{tag}.npy"),
                allow_pickle=True).item()
    robust[tag] = d
    print(f"{label:26s} {d['spearman_px_pair']:+12.3f} "
          f"{d['pearson_px_pair']:+8.3f}  {note}")

fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
for ax, (tag, label, note) in zip(axes, ROBUST_TAGS):
    d = robust[tag]
    names = [n for n in d["px"] if n in d["rows"]]
    x = np.array([d["px"][n] for n in names])
    y = np.array([d["rows"][n]["f1_pair"] for n in names])
    ax.scatter(x, y, s=42, color="#4269D0", alpha=0.85, edgecolors="white",
               linewidths=1.5, zorder=3)
    for idx in {int(np.argmax(x)), int(np.argmax(y))}:  # label PX pick + true best
        ax.annotate(names[idx], (x[idx], y[idx]), textcoords="offset points",
                    xytext=(5, -9), fontsize=8, color="#555555")
    ax.set_title(f"{label}\nSpearman {d['spearman_px_pair']:+.2f} · "
                 f"Pearson {d['pearson_px_pair']:+.2f}", fontsize=9)
    ax.set_xlabel("PX (no answer key)", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("truth-F1 (pair)", fontsize=9)
fig.suptitle("Robustness rungs — PX vs. truth accuracy per candidate decomposition",
             fontsize=11, y=1.06)
fig.tight_layout()
FIG = os.path.join(PLOTS, "robustness_scatter.png")
fig.savefig(FIG, dpi=110, bbox_inches="tight")
print("figure saved ->", FIG)  # display in Jupyter: Image(filename=FIG)

# nlgauss near-tie diagnostic: rank signal is carried by the degenerate tail
d = robust["nlgauss"]
names = [n for n in d["px"] if n in d["rows"]]
x = np.array([d["px"][n] for n in names])
y = np.array([d["rows"][n]["f1_pair"] for n in names])
keep = x > 0
sp_live, _ = spearmanr(x[keep], y[keep])
i_px, i_f1 = int(np.argmax(x)), int(np.argmax(y))
print(f"\nnlgauss: Spearman excluding PX=0 degenerates = {sp_live:+.3f} "
      f"(headline +0.624 leans on the easy tail)")
print(f"  PX top pick {names[i_px]} F1={y[i_px]:.3f} vs true best {names[i_f1]} "
      f"F1={y[i_f1]:.3f} -> top-pick cost {y[i_f1]-y[i_px]:.3f}")
print("\n=== demo notebook defined (RUN=False: heavy pipeline steps gated) ===")
