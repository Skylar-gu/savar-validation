# Demo notebook plan — "what this project is about" for a collaborator

A minimal, end-to-end walk-through: from a known synthetic world, train a black-box
forecaster, then recover its causal wiring **and** a trust dial **with no answer
key**. Each row below becomes one notebook section. Ground-truth data-generating
process (with equations) is documented first so every later step has a reference.

---

## 1. Data-generating process (SAVAR)

**SAVAR** = Spatially Aggregated Vector AutoRegression. A low-dimensional causal
system is "painted" onto a 2-D grid, so the observed data are high-dimensional
pixels but the true dynamics live among a few hidden **modes**.

### 1.1 Objects
| symbol | meaning | parent value |
|---|---|---|
| $N$ | number of modes (hidden variables) | 8 |
| $n_y\times n_x$ | grid resolution | $50\times50$ |
| $L=n_y n_x$ | pixels (observed dim) | 2500 |
| $W\in\mathbb{R}^{N\times L}$ | mode map: row $i$ = normalized Gaussian blob, $\sum_\ell W_{i\ell}=1$ | 8 disjoint $16\times16$ blobs |
| $W^{+}=\operatorname{pinv}(W)\in\mathbb{R}^{L\times N}$ | lifts modes back to pixels | — |
| $\Phi(\tau)\in\mathbb{R}^{N\times N}$ | causal coefficient matrix at lag $\tau$ (the ground-truth graph $G$) | $\tau\in\{1,2\}$ |
| $\tau_{\max}$ | max lag | 2 |
| $T,\ \text{burn}$ | usable steps, discarded transient | 500, 200 |
| $\lambda$ | noise strength | 1.0 |

### 1.2 Mode-space causal dynamics
The hidden mode signal is $Z(t)=W\,X(t)\in\mathbb{R}^N$. It follows a linear VAR
whose coefficients **are** the causal graph:
$$
Z(t)=\sum_{\tau=1}^{\tau_{\max}}\Phi(\tau)\,Z(t-\tau)+\xi(t).
$$
An entry $\Phi(\tau)_{j i}\neq0$ means **$X_i(t-\tau)\to X_j(t)$** (cause $i$ at lag
$\tau$ drives effect $j$).

### 1.3 Pixel-space realisation (what is actually generated)
The grid field $X(t)\in\mathbb{R}^L$ is evolved directly by projecting the mode
dynamics down to pixels and back:
$$
X(t)=\sum_{\tau=1}^{\tau_{\max}} W^{+}\,\Phi(\tau)\,W\,X(t-\tau)\;+\;\eta(t).
$$
(This is the literal update loop; $W^{+}\Phi W$ is the $L\times L$ pixel operator.)

### 1.4 Coloured spatial noise
Noise is added in pixel space with a mode-structured covariance:
$$
\eta(t)\sim\mathcal N(0,\ \Sigma_y),\qquad
\Sigma_y=\lambda\,W^{+}D_x\,(W^{+})^{\top}+D_y,
$$
with latent-innovation cov $D_x=I_N$ and per-pixel cov $D_y=I_L$. $D_y=I_L$
guarantees $\Sigma_y\succ0$. So each mode carries independent innovations, smeared
onto the grid by its blob, plus white per-pixel noise.

### 1.5 Optional seasonality (used in the diurnal rung)
An additive periodic trend, same phase everywhere (optionally spatially weighted):
$$
s(t)=A\,\sin\!\Big(\tfrac{2\pi}{P}\,t\Big),\qquad X(t)\mathrel{+}= s(t)\,[\,\odot\ w_{\text{seas}}\,].
$$
$A$ = amplitude, $P$ = period, $w_{\text{seas}}$ = optional per-pixel weight.

### 1.6 Optional external forcing (used in the regime rung)
Piecewise step forcing on chosen modes over time windows $[t_1,t_2]$: a constant
$f$ added through a forcing map $w_f$ during the window — models regime shifts.

### 1.7 Ground-truth graph (parent)
`links_coeffs` (format $\{\,j:[((i,-\tau),\text{coeff}),\dots]\}$); diagonal terms
are autocorrelation, off-diagonal are the causal edges:
```
X0 ← 0.45·X0(t-1) + 0.22·X2(t-2)          X4 ← 0.40·X4(t-1) + 0.25·X1(t-2)
X1 ← 0.50·X1(t-1) + 0.35·X0(t-1)          X5 ← 0.30·X5(t-1) + 0.35·X4(t-1) − 0.20·X0(t-2)
X2 ← 0.35·X2(t-1) + 0.40·X1(t-1)          X6 ← 0.50·X6(t-1) + 0.30·X3(t-1) + 0.25·X5(t-2)
X3 ← 0.55·X3(t-1) + 0.30·X0(t-1) − 0.30·X2(t-1)   X7 ← 0.45·X7(t-1) + 0.20·X6(t-1) − 0.15·X3(t-2)
```
Designed features: **X0 = hub** (high out-degree), **X3 = collider** (two parents,
different signs), **X7 = pure sink** (no outgoing), mixed lag-1/lag-2 edges, and
negative edges (X0→X5, X2→X3, X3→X7). Stationarity is checked before generating.

---

## 2. Minimal experiment set (one row = one notebook section)

The spine of the recipe: **make a world → train a black box → find its parts →
read its wiring two independent ways → let their agreement pick the answer and
predict its accuracy, all without the answer key.**

| # | Experiment | What it is (plain language) |
|---|---|---|
| **1** | **Generate SAVAR world** | • Build a small known causal graph and paint it onto a 50×50 grid.<br>• Output: pixel movies whose *true* wiring we secretly know (the answer key).<br>• This is the sandbox — every later score is checked against it. |
| **2** | **Train the forecaster** (MeshGNN) | • Train a graph neural net (~890K params) to predict the next frame from the past.<br>• It learns the dynamics but never sees the causal graph — it's the black box we interpret.<br>• Stand-in for a real weather model like GraphCast. |
| **3** | **E1 — discover the modes** | • From data alone, unsupervised, recover the hidden regions (the blobs).<br>• Try several candidate methods (clustering, varimax, Leiden…) — some good, some bad.<br>• We deliberately keep good *and* bad candidates so the selector has something to choose between. |
| **4** | **Integration view — Ĝ_int** | • Take each candidate's mode time-series and run causal discovery (PCMCI).<br>• Reads the wiring from *patterns in the recordings* — whose past predicts whose future.<br>• Pure observation; the model is never touched. |
| **5** | **Dynamics view — Ĝ_dyn** | • Poke the trained forecaster on one mode and watch which modes respond.<br>• Reads the wiring from *behavior* — what actually moves what.<br>• An experiment on the model, with different blind spots than view 4. |
| **6** | **PX selector — pool-crossed agreement** | • Score each candidate by how well its integration-wiring matches the *other* candidates' dynamics-wiring.<br>• High agreement ⇒ faithful decomposition — computed with **no answer key**.<br>• Picks the best mode set and flags bad ones; "crossed" stops a candidate grading its own homework. |
| **7** | **E4-final — the trust dial** | • Pool (agreement, true-accuracy) points across worlds and fit one line.<br>• Turns an observed agreement into a *predicted* accuracy (≈1.48·PX+0.13, ±0.13).<br>• The deliverable: a calibrated confidence read-out for a model you can't check. |

**Optional appendix rows** (show robustness, not core story): overlap / rollout /
multivariate / scale rungs re-run steps 1–6 on harder worlds; each is one extra
call with a different data tag.

---

## 3. Notebook build notes
- One section per row; each ends with a figure (true graph, learned modes, the two
  wiring diagrams side by side, the PX ranking, the trust-dial scatter).
- Steps 1–2 are precomputed (data + checkpoint on disk) — the notebook loads them;
  a `--smoke` path can regenerate a tiny world live.
- Steps 3–6 reuse `sae/discover_modes.py` and `pcmci/e4_agreement.py`; step 7 uses
  `analysis/e4_final_calibration.py`. Keep the parent world for the demo (fastest,
  cleanest signal); mention the harder rungs as the robustness story.
- Final cell = the trust-dial plot (`results/plots/trust_dial.png`) with the
  one-sentence takeaway.
