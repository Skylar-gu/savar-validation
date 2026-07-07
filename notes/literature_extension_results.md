# litext session results — 2026-07-06

Executes [[literature_extension_experiments]] steps 1–4: E1 (mode-discovery bake-off), E2
(Adag-selection test), E3 (dynamical-arm upgrade). Numbers first. Anchors
throughout: PCMCI+ on true Z F1 = 0.853 (graph, exact-lag protocol) / 0.823
(pair-level Block-B protocol); oracle-W pooled activations F1 = 0.855 (FU1).

---

## E3 — internals-free dynamical arm upgraded (Block B F1 0.500 → …)

`pcmci/impulse_response_v2.py`. Three arms + statistic variants, all under
Block B's detection protocol (W-projected response, pixel-permutation null
α=0.01, 1000 perms, pair-level scoring vs the 12 gt cross edges). New:
**ancestor-level scoring** vs the transitive closure (29 ordered pairs) —
dynamical probing measures *total* effects, so a detected 2-hop path is not a
model error.

**Run 1** (240 windows, 12 steps, σ=1: `results/litext_e3_dynarm.npy`):

| arm | direct P/R/F1 | ancestor P/R/F1 | lag-ok | τ-Spearman |
|---|---|---|---|---|
| A baseline (Block B replica) | 0.625/0.417/**0.500** | 1.00/0.28/0.43 | 5/5 | 0.738 |
| B teacher-forced | 0.600/0.500/**0.545** | 1.00/0.34/0.51 | 6/6 | **0.952** |
| C sustained forcing | 0.625/0.417/0.500 | 1.00/0.28/0.43 | 0/5 | n/a |
| C sustained ×3σ | 0.556/0.417/0.476 | 1.00/0.31/0.47 | 0/5 | n/a |

- Arm A reproduces Block B byte-for-byte (TP=5 FP=3 FN=7, same sets) —
  protocol validated.
- **Arm B fixes the timescale channel**: e-folding times become monotone in φ
  (Spearman 0.738 → 0.952 at 12 steps; no saturation), causally confirming
  FU1's diagnosis — Block B's flat τ̂ was rollout damping, not representation.
- Sustained forcing (C) is a null for recall and destroys lag info: not the
  lever.
- Dose linearity (A): response-shape corr(1×,3×) = 0.999, amplitude ratio
  3.01 — the model's response is linear in the impulse over ±3σ.

**Run 2** (240 windows, **24 steps**, + integral statistic Σ_τ|R| —
slow-mode responses are broad and low, exactly what a max-statistic misses;
`results/litext_e3_dynarm_s24.npy`):

| arm/stat | direct P/R/F1 | ancestor P/R/F1 |
|---|---|---|
| A max | 0.625/0.417/0.500 | 1.00/0.28/0.43 |
| A integral | 0.389/0.583/0.467 | 0.83/0.52/0.64 |
| B max | 0.600/0.500/0.545 | 1.00/0.34/0.51 |
| **B integral** | 0.529/**0.750**/**0.621** | **0.941/0.552/0.696** |
| C ×3σ | 0.700/0.583/0.636 | 1.00/0.34/0.51 |

- **B+integral: recall 0.417 → 0.750** (9/12 direct edges, 8/9
  lag-consistent). Missed: (2→0), (2→3), (5→6) — the X2-sourced edges and the
  slow–slow lag-6 edge.
- Every direct-level FP but one is a true 2–3-hop path (ancestor precision
  0.941; the single non-path detection is (3,4)). **The arm essentially never
  hallucinates influence** — the remaining direct-FP problem is
  direct-vs-indirect separation, not detection error.
- At 24 steps the free-rollout arms degrade (A τ-Spearman flips negative —
  damping artifact, consistent with Block B's 24-step sensitivity run), while
  teacher-forced stays clean (0.810). Teacher-forcing is what makes longer
  horizons usable.

**Run 3** (arm B only, 720 windows, 24 steps, + **deconvolution scoring**:
the measured R[i,j,τ] is the model's total-effect Green's function; the
Volterra recursion B[τ] = T[τ] − Σ_{s<τ} B[s]·T[τ−s] extracts direct kernels,
with the permutation null pushed through the same recursion;
`results/litext_e3_dynarm_B720.npy`):

| variant | direct P/R/F1 | ancestor P/R/F1 |
|---|---|---|
| B max | 0.600/0.500/0.545 | 1.00/0.34/0.51 |
| B integral | 0.529/0.750/0.621 | 0.94/0.55/0.70 |
| **B deconv** | **1.000**/0.417/0.588 | 1.00/0.17/0.29 |

- 720 windows = 240 windows, byte-identical detection sets → the misses are
  NOT a noise-floor problem.
- **Deconvolution does exactly its job: zero false positives** — every
  detection is a direct, lag-correct edge (5/5). Its recall cost is the
  null-noise amplification through the recursion (X1→X4 sits at ratio 0.99 of
  its α=0.01 threshold — just under).

**The residual misses are the model's, not the method's.** Analytic control:
propagate impulses through the TRUE companion dynamics (ground_truth_graph;
note `ground_truth_graph[effect, cause, lag]` orientation) and compare
integral response strengths per edge:

| missed edge | true-system rank (1=weakest of 12) | model response (stat/thresh) |
|---|---|---|
| X2→X0 | 1 | 0.01 |
| X2→X3 | 5 | 0.01 |
| X5→X6 | **11** (second-strongest, sum\|R_true\|=7.1) | 0.02 |

All three have response at 1–2% of the null threshold — the frozen GNN
implements **no transfer at all** on these edges (X6→X7, same slowness class,
shows a clean bump peaking at its designed lag 4). X2→X0 is also the weakest
edge in the true system (natural sensitivity limit), but X5→X6 is the
second-strongest — **the emulator genuinely failed to internalize the
slow–slow lag-6 coupling.** Meanwhile PCMCI-on-activations (FU1, R=1.00)
recovers all three: the model *encodes* these correlations without
*implementing* them in its forward map.

**E3 verdict.** On the 9 edges the model actually implements, the upgraded arm
detects **9/9** (B+integral), 8/9 lag-consistent; the deconv variant supplies
a zero-FP direct core; ancestor precision 0.94–1.00 across every variant (the
arm never hallucinates influence). Raw direct F1 0.621 misses the 0.75 bar,
but the entire shortfall is now attributed to genuine zeros of the model's
transfer function — which is itself the payoff: **representation-graph ≠
implemented-dynamics-graph, and the two channels' disagreement localizes
exactly which physical couplings the emulator failed to learn.** That
dissociation is the E4 calibration's raw material, and on GraphCast it is the
process-level model-evaluation product (Nowack-style): edges present in the
data/reanalysis graph but absent from the model-response graph = dynamics the
emulator misrepresents. Teacher-forced propagation (not free rollout) is what
makes the response channel trustworthy: it fixed the timescale readout
(Spearman 0.95 vs 0.74, no saturation) and stays clean at 24 steps where free
rollout degrades.

---

## E1 — unsupervised mode-discovery bake-off

`sae/discover_modes.py`. Discovery fit on 4 held-out realisations (96–99);
PCMCI eval on realisations 0–23 (Block-G protocol, exact-lag scoring,
Hungarian-strict variable mapping: edges touching unmatched discovered
variables = FP, gt edges at unmatched true modes = FN).

**Footprint recovery** (Hungarian-matched cosine to true W rows; N̂ selected
by coherence floor 0.25 from C0=12 initial components):

| candidate | N̂ | matched | mean cos | note |
|---|---|---|---|---|
| vmax_act | 12 | 8/8 | 0.701 | varimax-PCA on per-node activation scalar field |
| **vmax_pix** | **8** | 8/8 | **0.999** | varimax-PCA on raw pixels — near-exact, N̂ exactly 8 (coherence eigengap 0.99→0.20) |
| km_act | 10 | 4/8 | 0.610 | k-means on activation time courses |
| km_pix | 9 | 8/8 | 0.776 | |
| dmd_act | 8 | 4/8 | 0.405 | k-means on \|DMD mode\| loadings |
| merge01/coarse4/split7/fine16/shift5/diag8/blur | — | — | 0.96/0.71/0.96/0.71/0.00/0.00/0.50 | corrupted variants for E2's quality axis |

**Graph recovery** (pixels pooled through each Ŵ → PCMCI+, 24 reals;
anchors: true-Z / oracle-W pixels = 0.853, oracle-W activations = 0.855):

| candidate | F1 | P | R | reading |
|---|---|---|---|---|
| vmax_pix | **0.853** | 0.75 | 1.00 | = oracle, edge-for-edge (R=1.00, fn=1/288) |
| km_pix | **0.853** | 0.75 | 1.00 | = oracle |
| **vmax_act** | **0.819** | 0.83 | 0.81 | fully-internal discovery, −0.036 vs oracle despite cos 0.70 + 4 surplus components |
| km_act | 0.280 | 0.37 | 0.23 | 4/8 footprints → collapse |
| dmd_act | 0.177 | 0.18 | 0.17 | weakest discovery |
| blur (corrupt) | **0.892** | 0.81 | 0.99 | *above* oracle: smoothed footprints average more pixels → less observation noise; footprint sharpness is not what the graph needs |
| split7 | 0.776 | 0.72 | 0.84 | one split mode: mild damage |
| merge01 | 0.587 | 0.53 | 0.66 | one merged pair: moderate |
| coarse4 | 0.207 | 0.29 | 0.16 | merged pairs: severe |
| fine16 | 0.013 | 0.09 | 0.01 | every blob split: near-duplicate variables condition each other's edges away — textbook faithfulness violation |
| shift5 / diag8 | 0.000 | 0.00 | 0.00 | misplaced footprints: total loss (402/340 FPs) |

**Fully-internal path** (activations pooled through Ŵ, per-mode PC1 readout —
unsupervised end to end, no Z and no pixels in the series):

| variant | F1 | P | R |
|---|---|---|---|
| acts:oracle (oracle W + PC1 readout) | 0.813 | 0.69 | 0.99 |
| acts:vmax_act (discovered W + PC1 readout) | 0.641 | 0.68 | 0.60 |

**E1 readings.**
1. **The W-free price on this rung is ≈0.00–0.03 F1** when discovery uses the
   varimax operator: from raw pixels it is exactly oracle (0.853, R=1.00);
   from internals alone 0.819. The existence proof (FU1) converts to a method.
2. **The W-free price decomposes additively until the ends are combined**:
   unsupervised readout costs 0.04 (0.855 ridge → 0.813 PC1 at oracle W);
   discovery costs 0.03 (0.853 → 0.819 at pixel pooling); the fully-internal
   combination compounds to 0.641 (imperfect footprints × imperfect readout).
   Consequence for GraphCast: the **hybrid route** — footprints discovered
   from *internals*, series pooled from *data* — is the strong configuration,
   and it is fully available there (ERA5 is the data). The all-internal route
   is the fallback when the question is specifically "what graph does the
   model encode."
3. Discovery quality is strongly method-dependent (varimax ≫ k-means ≫ DMD on
   activations) — and footprint cosine is NOT a sufficient predictor of graph
   F1 (blur cos 0.50 → F1 0.892; km_act cos 0.61 → F1 0.28). What matters is
   whether the pooled series preserve the independence model — precisely the
   property E2's consistency scores test.
4. The corruption ladder orders as theory predicts: blur (benign) > split-one
   > merge-one > merge-all ≈ weak-discovery > split-all ≈ misplaced (fatal),
   giving E2 a full-range quality axis (F1 0.00–0.89).

---

## E2 — Adag consistency scores as unsupervised selector

**v1 — pure level-consistency: FAILS as a selector, in two instructive ways**
(`pcmci/aggregation_selection.py`, `results/litext_e2_adag.npy`; scores per
candidate = agreement between aggregate-level and pixel-level (in)dependence
verdicts + a sufficiency test; 6 reals, α=0.01):

| score | Spearman vs truth-F1 (13 candidates) |
|---|---|
| S_dep | −0.277 |
| S_indep | +0.572 |
| S_suff | −0.180 |
| S_joint | **+0.174** |

1. **Consistency is gameable by signal destruction.** diag8 (truth-F1 0.000)
   scored S_joint 0.973: a map that pools noise has *no dependencies at either
   level* and is vacuously consistent. fine16 (duplicate halves, F1 0.013)
   scored 0.977 — its faithfulness pathology is invisible to level-agreement.
   Consistency is necessary, not sufficient; selection needs an
   informativeness term and a non-redundancy term.
2. **Over-conditioning bug with a lesson.** v1's "pragmatic FullCI" included
   the tested pair's own lags in the conditioning set — which conditions away
   the lagged dependence under test (the oracle showed n_dep = 0: its 12 true
   edges all erased). Residual dependence under that protocol measures
   aggregate *impurity*, inverting the intended ranking. General lesson for
   any consistency-score implementation: the aggregate-level and micro-level
   tests must target the dependence the *discovery* stage will use, with the
   candidate cause's past left out of the conditioning set.

**v2 — composite selector** (`pcmci/aggregation_selection_v2.py`): partial
aggregation (random half-splits) as micro-variables for power; four components
— S_info (dependency density; vacuous maps → 0), S_dup (near-deterministic
pair penalty; catches fine16), S_agree (micro–macro agreement counted only on
pairs with signal — no vacuous credit), S_suff (halves residualized on own
aggregate must be independent of other aggregates); S_total = √S_info · S_dup
· S_agree · S_suff. *(running)*
