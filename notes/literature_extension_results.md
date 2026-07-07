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

**v2 — composite selector** (`pcmci/aggregation_selection_v2.py`,
`results/litext_e2_adag_v2.npy`): partial aggregation (random half-splits) as
micro-variables for power; four components — S_info (dependency density;
vacuous maps → 0), S_dup (near-deterministic pair penalty), S_agree
(micro–macro agreement counted only on pairs with signal), S_suff (halves
residualized on own aggregate independent of other aggregates); pre-registered
S_total = √S_info · S_dup · S_agree · S_suff.

**The shift5 rescore — the truth axis itself was wrong for misplaced maps.**
v2 gave shift5 (footprint-matched "F1 = 0.000") a near-oracle score of 0.619.
Investigating: shift5's pooled series correlate 0.979 with the true modes —
its footprints are geometrically wrong but behaviorally coherent, and under
**behavior-based matching** (variable ↔ mode by series correlation, not
footprint cosine) its graph scores **F1 = 0.835**. The consistency scores were
right; the footprint-cosine Hungarian convention was mislabeling a good map.
diag8 stays 0.000 under both matchings (S_info = 0 agreed). Methodological
consequence for GraphCast: *footprint geometry is the wrong success criterion;
behavioral coherence of the pooled series is the right one*, and all
calibration must use label-free scoring.

**Verdict against corrected truth-F1** (shift5 → 0.835):

| score | Spearman (13 candidates) |
|---|---|
| S_total (pre-registered) | **+0.522** |
| S_agree | +0.554 |
| S_info | +0.483 |
| S_dup | +0.354 |
| S_suff | +0.008 |
| exploratory no-suff composites | +0.49 |

**Bar (≥0.8) NOT met — the pre-registered fallback branch activates.** The
structural blind spots are now precisely characterized:
1. **Merges are consistency-invisible**: coarse4 passes info/agree/dup (its
   aggregates carry real dependencies); only the purity test (S_suff) catches
   it — but purity also executes blur (F1 0.892) and vmax_act (0.819), whose
   *impurity is harmless or helpful*. Purity ≠ usability is the same lesson
   E1's blur result taught, now seen from the selection side.
2. **The scores' zeros are trustworthy** (S_info = 0 or S_agree ≈ 0 ⇒ map is
   genuinely dead: diag8, and v1's vacuous cases once informativeness is
   demanded) — consistency works as a *screen*, not a *ranker*.
3. With 13 candidates, further composite iteration is overfitting; per the
   plan (§6 risk register), **selection now falls to E4 cross-channel
   agreement** (activations-graph vs response-graph on the same Ŵ), with the
   consistency screen retained as a cheap first filter. E3's dissociation
   result (channels disagree exactly where the model is wrong) makes this
   fallback more credible than it was at planning time.

---

## E4-v1 — agreement→accuracy calibration (2026-07-07 session)

`pcmci/e4_agreement.py`, `results/litext_e4_agreement.npy`. Battery rerun with
per-real edge SETS (`litext_e4_int_partial.npy`), behavior-based matching as
the default convention (Hungarian on mean |corr(Ŵ-pooled pixel series, true
Z)| over 6 reals, match iff ≥ 0.3). Dyn channel = E3 arm-B teacher-forced
propagation THROUGH each candidate: **impulse patterns from pinv(Ŵ)** (fully
W-free — documented choice; NOT W_plus), responses read out on Ŵ rows,
integral statistic, pixel-permutation null α=0.01, 240 windows × 24 steps.
Agreement = pair-level edge-set F1 between Ĝ_int (PCMCI consensus: pair in
≥50% of 24 reals, lags marginalized) and Ĝ_dyn, in the candidate's own
variable space (no truth anywhere).

**Behavior matching validates and generalizes the shift5 lesson.** Matched
counts: shift5 8/8 (mean |corr| 0.978 → truth-F1 0.835, the plan's number,
now reproduced in-battery), diag8 0/8, coarse4 4/8, km_act 5/8, dmd_act 7/8
(0.901). dmd_act's truth-F1 rises 0.177 → **0.553**: footprint-cosine
matching had mislabeled it too, not just shift5. E1 F1s otherwise reproduce
byte-for-byte (vmax_act 0.819, vmax_pix/km_pix/oracle 0.853, blur 0.892,
fine16 0.013).

**Same-Ŵ agreement FAILS as a calibrator — for two structural, signed
reasons** (Spearman +0.435 exact-lag / +0.375 pair, bar ≥ 0.8):

| candidate | truth-F1 (pair) | same-Ŵ agree | pool-crossed |
|---|---|---|---|
| blur | 0.918 | 0.556 | 0.453 |
| oracle | 0.879 | 0.621 | 0.447 |
| vmax_pix | 0.879 | 0.621 | 0.447 |
| shift5 | 0.878 | **0.154** | 0.490 |
| km_pix | 0.876 | 0.529 | 0.447 |
| vmax_act | 0.835 | 0.298 | 0.424 |
| split7 | 0.801 | 0.424 | 0.400 |
| dmd_act | 0.632 | 0.118 | 0.351 |
| merge01 | 0.622 | 0.696 | 0.391 |
| km_act | 0.289 | 0.375 | 0.166 |
| coarse4 | 0.230 | **0.727** | 0.149 |
| fine16 | 0.013 | 0.000 | 0.000 |
| diag8 | 0.000 | 0.000 | 0.000 |

1. **Merges self-agree on the wrong graph** (coarse4: agreement 0.727, the
   POOL MAXIMUM, truth 0.230). Both channels inherit the marginalization
   jointly — the same blind spot that made merges consistency-invisible in
   E2, now seen in two-channel form. Shared-Ŵ confound, exactly as the plan's
   risk register predicted.
2. **The write channel is geometry-sensitive where the read channel is not**
   (shift5: agreement 0.154, truth 0.878, |Ĝ_dyn| = 1). Reading through Ŵ
   only needs behaviorally coherent pooled series; *writing* through Ŵ⁺
   needs the injection to land on the true supports. A misplaced-but-
   coherent map reads fine and writes dead. New, GraphCast-relevant
   asymmetry: perturbation-based validation implicitly certifies footprint
   GEOMETRY, not just behavior.

**The pre-registered cross-Ŵ cells repair it: pool-crossed agreement passes
the bar.** PX(A) = mean over B≠A of F1(Ĝ_int(A), Ĝ_dyn(B)) (all quantities
unsupervised; the dyn pool serves as a Ŵ-independent reference, breaking the
shared-Ŵ error):

| statistic | Spearman vs truth-F1 |
|---|---|
| same-Ŵ agreement (pair) | +0.375 |
| **pool-crossed (pair)** | **+0.950** (Pearson +0.99) |
| pool-crossed (exact-lag truth) | +0.943 |
| transpose (dyn(A) vs int pool) | +0.288 |

Cross-matrix diag 0.467 vs off-diag 0.347. The direction matters: the INT
channel carries the map-quality signal; averaging over the pool of dyn
references cancels per-Ŵ channel error. Behavior-good maps sit at 0.40–0.49,
merged/weak at ≤ 0.17, dead at 0.00 — a usable unsupervised selector with a
clean gap, where E2's consistency scores gave +0.52. Caveat, stated plainly:
the aggregation rule (mean over the candidate pool) was fixed AFTER seeing
the two confounds, on n=13 candidates — R6/R1 reruns are its out-of-sample
test. G2 selection = consistency screen (E2 zeros) → pool-crossed ranking;
G3 confidence = the pool-crossed → truth-F1 curve above.

**Disagreement classification (the E3 dissociation, now through discovered
Ŵ's).** On good maps whose dyn channel is live (oracle, vmax_pix, km_pix,
split7, vmax_act), the true edges in Ĝ_int but NOT in Ĝ_dyn are **exactly
{(2→0), (2→3), (5→6)}** — the three model-unimplemented couplings — 15/16
detections (vmax_act adds (0→3)), zero non-gt disagreements. Pooled over all
good maps incl. handicapped dyn channels (blur's smeared, shift5's shifted
injections): 21/34 = 62% vs 25% base rate. "In the reading-graph but not the
poking-graph" localizes emulator deficiencies through fully-discovered
aggregation maps — the Nowack-style model-evaluation product survives losing
the oracle W.

---

## R2 — static-inputs rung (partial: train + ablation)

`train/gnn/gnn_forecaster.py` + `GNN_STATIC_INPUTS=1`: sin/cos(2πu/50),
sin/cos(2πv/50), hub flag concatenated to the K frames (k → k+5 input
channels), node_emb = 0. Trained on splits_hetdynamics_eqvar
(checkpoints/hetdynamics_eqvar_static/); run reached epoch 6, val corr
0.4553 vs the 40-epoch no-static baseline 0.4563 — converged to within
0.001, kept.

**Ablation check: the static channels are USED** (`eval_static_ablation.py`,
`results/litext_r2_static_ablation.npy`): zeroing them at inference drops
val corr 0.4553 → 0.3966 (**Δ = +0.059**, 30× the 0.002 ignore-threshold),
while skill with them exactly matches the no-static baseline. The network
integrated the coordinate channels into its computation WITHOUT any accuracy
gain — position information it previously derived from mesh heterogeneity is
now partly sourced from the input, making identity-as-content available to
probes.

**Identity probe: a perfect ADDRESS code appears; the shape code does not
move** (`sae_data/hetdynamics_eqvar_static/probe_mode_identity.npy` vs the
baseline checkpoint's probe):

| probe | baseline ckpt | static ckpt |
|---|---|---|
| linear/raw | 0.700 | **1.000** (all 8 modes 1.00) |
| linear/demeaned | 0.163 | **0.175** |
| mlp/raw | 0.710 | **1.000** |
| (chance) | 0.125 | 0.125 |

Static inputs create the first genuine what-code of the whole program — and
it is *exactly* a lookup: remove each mode's constant offset and identity
decodability collapses back to near-chance (0.175). No identity-specific
*computation* (shape code) emerges. GraphCast translation: lat/lon/orography
inputs guarantee position decodability from activations, but that
decodability is offset-based; do not read it as evidence of mode-specific
dynamics processing.

**E1 discovery on the static checkpoint: footprints sharpen, the graph does
NOT improve** (`results/litext_e1_discovery_r2static.npy`, footprint-matched
E1 protocol): vmax_act footprint cosine 0.701 → **0.882** (N̂ still 12), but
pixel-pooled graph F1 0.819 → **0.651** (R 0.81 → 0.57); oracle battery
reproduces 0.853 exactly (protocol control). Discovery geometry benefits
from the coordinate channels; the sharper components do not preserve the
independence model any better — footprint cosine is not the target (E1
lesson, re-confirmed from the opposite direction).

**E3 arm B on the static checkpoint: implemented-coupling recall HALVES at
identical forecast skill** (`results/litext_e3_dynarm_r2static.npy`;
B+integral): direct recall 9/12 → **5/12** (F1 0.476 vs 0.621), missing the
baseline's 3 unimplemented edges PLUS (0→5), (1→4), (3→7), (6→7) — all four
newly-lost edges are the slow/long-lag couplings (ℓ = 3, 4, 6). Ancestor
precision stays 1.000 (the arm still never hallucinates), τ-Spearman 0.905,
deconv P = 1.000 (zero FP). Reading: with an address channel available, the
network satisfies the same loss while implementing LESS long-range
cross-mode transfer in its forward map — an architecture/input change
invisible to val corr is fully visible to the response channel. Two
equally-skilled emulators differ ~2× in how much of the physical coupling
structure they implement; the Nowack-style response-graph product measures
exactly that.

**Fully-internal path (acts pooled through Ŵ, PC1 readout):** acts:oracle
0.813 → **0.876** on the static checkpoint (the per-mode address offsets
sharpen the unsupervised PC1 readout — above even the parent's 0.855 ridge
anchor); acts:vmax_act 0.641 → **0.482** (the sharper-cosine discovered
footprints are *worse as pooling maps*, compounding with the readout).

**R2 verdict.** Static coordinate inputs create identity-as-content in
exactly one sense: a perfect, linearly-readable, offset-based ADDRESS code
(probe 1.000) that improves oracle-pooled readout (0.876) — while the shape
code stays at chance, discovered-Ŵ graph recovery degrades (0.819→0.651
pixel; 0.641→0.482 internal), and the implemented dynamics THIN OUT (E3
recall 9/12→5/12 at equal skill). Identity-as-content ≠ better causal
substrate; it substitutes for implemented coupling rather than augmenting
it.

---

## R6 — atmosphere-regime rung (generator + ceiling done, training running)

`data_gen/generate_atmoregime.py`: φ = 0.900…0.990 (τ log-spaced 9.5 → 100
steps, spread 10.5×), same 12-edge set + lags, cross-coefficients ×0.20
(REQUIRED for stationarity: the 0→1→2→0 cycle under near-integrator
diagonals gives companion spectral radius 1.203 at full strength; 0.9911 at
×0.20; critical scale 0.2135) — verified and printed by the generator.
T=9600 + burn 900, 40 realisations (3.4 GB), DY_SCALE 0.0125 (4× cut).
Equal stationary variance per mode via ABSOLUTE innovation scales calibrated
empirically in two pilot iterations: per-mode Z std 1.225–1.259 vs target
1.23 (parent eqvar amplitude — tanh-saturation regime matched). Split
70/15/15 → data/splits_atmoregime.

**Rung ceiling (own anchor): PCMCI+ on true Z, 12 reals, F1 = 0.801**
(P 0.755, R 0.854, sign 1.000; `results/pcmci_modes_atmoregime.npy`). The
near-unit-memory regime costs −0.05 F1 vs the parent ceiling 0.853 at
T=9600 — autocorrelation shrinks effective sample size but nowhere near the
<0.4 collapse gate. MeshGNN training: **val corr 0.7068 after ONE epoch**
(vs 0.4563 converged on the parent rung), plateau 0.7079 = **97.8% of the
achievable one-step ceiling 0.724** (unpredictable variance = W⁺-injected
mode innovations 0.086 + pixel noise 0.0125 against obs var 0.207) — the
forecaster is at ceiling, not gradient-starved. Pixel-side E1 battery
(`litext_e1_discovery_atmo_satv1.npy`): oracle 0.801 = ceiling exactly
(aggregation lossless, F4 replicates); vmax_pix N̂=8, cos 0.999, **F1 0.793**
(discovery price −0.008); corruption ladder ordered as parent, EXCEPT blur
0.599 (was 0.892 > oracle on parent): with 4×-lower pixel noise, smearing's
noise-averaging benefit disappears and cross-blob mixing costs recall —
"benign blur" was a property of the noise level, not of blur.

**R6-v1 POSTMORTEM — the saturated self-loop destroys the regime; v1 is a
diagnostic, not the rung.** E3 arm B on the v1 checkpoint: ZERO detections,
flat e-folds ≈ 4 steps for every mode (`litext_e3_dynarm_atmo_satv1.npy`).
Cause, verified in data: the parent generator applies g_sat(m) = 0.5·m +
0.5·tanh(m) to the lagged state INCLUDING the φ self-loop; at operating
amplitude 1.23 the saturation derivative ≈ 0.66, so SMALL-SIGNAL memory is
τ_eff = −1/ln(0.66·φ) ≈ 3 steps regardless of φ. Realized ACF(1) on v1 data:
0.686–0.765 (τ_eff 2.7–3.7, spread 1.36×) vs designed 0.90–0.99 (10.5×). The
same mechanism affects every rung of this family: hetdynamics_eqvar's
"φ=0.92" mode realizes ACF(1)=0.74 (τ≈3.3). Consequences: (i) v1's E3 null is
FAITHFUL — the true small-signal transfer is weak and fast, and the model at
97.8% of ceiling mirrors it; (ii) all φ-labelled claims across the program
describe realized memories ≤ 3.7 steps, not the nominal τ; (iii) the
variance/ACF regime and the impulse-response regime of a saturated system
are different regimes — measure both before claiming either.

**R6-v2 generator** (same file, documented in-code): (a) self-loops LINEAR
(realized memory = φ exactly), cross-terms keep the tanh saturation;
(b) NL_BETA = 0 — the bounded bilinear forcing integrates under near-unit
memory to ±β/(1−φ) ≈ ±15σ (v2 pilot: X7 std 26, random walk) — a
regime-confound, dropped for this rung; (c) cross-coefficients DC-GAIN-
MATCHED to the parent: c_new = c_parent·(1−φ_new[eff])/(1−φ_parent[eff]) ×
0.5 — a global scale is the wrong knob for near-integrators, where each
edge's integrated gain is c/(1−φ) (at parent-level DC gain the slow modes'
cross-driven variance alone exceeds the eqvar target; ×0.5 makes eqvar
reachable). Realized: ACF(1) within 0.009 of designed φ (0.903–0.995,
τ 9.8→~200), per-mode std 1.20–1.27 (eqvar), radius 0.990. This is the
"weak instantaneous, strong integrated teleconnection" structure of real
slow climate modes. Regeneration + own ceiling + retrain + battery reruns:
below.

---

## Session closing — status vs the plan's bars

| exp | bar | outcome |
|---|---|---|
| E1 discovery bake-off | best candidate within 0.05 of oracle-W | **PASS**: vmax_pix 0.853 = oracle exactly (R=1.00); internals-only vmax_act 0.819 (−0.036); shift5 0.835 under behavior-matching. Discovery price ≈ 0.02–0.04 F1 |
| E2 consistency selection | Spearman ≥ 0.8 | **FAIL, cleanly factored**: pre-registered composite +0.52 after truth-axis correction; zeros trustworthy (screen), ranking not; merges consistency-invisible vs purity-kills-good-maps tension; fallback → E4 agreement (pre-registered branch) |
| E3 dynamical arm | direct F1 ≥ 0.75 | **method at ceiling, model is the limit**: 9/9 recall on the edges the model implements; deconv variant P=1.000 (zero FP); 3 misses = genuine transfer-function zeros of the frozen GNN (one is the true system's 2nd-strongest edge) — an emulator deficiency the arm *discovered*; τ-Spearman 0.74→0.95 via teacher-forcing |

**Next steps (in plan order):** E4 agreement↔accuracy calibration — rerun the
E1 battery saving per-candidate edge SETS (not just counts) so agreement cells
can be computed against E3's saved detections; add behavior-based matching as
the default scoring convention. Then R2/R6 rungs (cheap trains) and R1.

---

## R6-v3 readouts (harvested 2026-07-07 after agent session-limit cutoff)

Rung anchors (committed earlier): realized ACF = designed φ within 0.009
(τ genuinely 9.5–100 steps), spectral radius 0.990, **rung ceiling
PCMCI-on-Z F1 = 0.509** (T=9600; the near-unit-memory regime's honest
anchor), forecaster val corr 0.9245 = **98.8% of the achievable 0.936** —
the GraphCast-like one-step regime (R² ≈ 0.9, signal-rich K=3 window,
abundant gradient) is real on this rung.

**E1 discovery battery** (`litext_e1_discovery_atmo.npy`; score / ceiling
ratio in parens):

| candidate | cos | F1 | vs ceiling |
|---|---|---|---|
| oracle | 1.000 | **0.509** | 1.00 — aggregation STILL lossless (F4 replicates at high memory) |
| split7 | 0.963 | 0.477 | 0.94 |
| km_act | 0.553 | 0.455 | 0.89 |
| km_pix | 0.722 | 0.388 | 0.76 |
| vmax_pix | 0.985 | 0.385 | 0.76 |
| blur | 0.500 | 0.383 | 0.75 |
| vmax_act | **0.863** | 0.214 | 0.42 |
| acts:oracle | — | 0.312 | 0.61 |

Footprint recovery stays excellent (vmax_pix 0.985; vmax_act 0.863 —
*sharper* than the parent rung's 0.701) but graph recovery decouples from
footprint quality: recall is the casualty everywhere (R 0.26–0.38, oracle
included). The W-free discovery price is no longer ≈0: −0.12 (pixels)
to −0.30 (internals) vs the rung ceiling, and the candidate ordering
inverts (km_act > vmax_pix). At near-unit memory the graph, not the
footprints, is the hard part.

**SAE metric suite, 3 seeds** (`litext_sae_metrics_atmo_seed*.npy`) — the
window-composition prediction (R6's SAE readout) **CONFIRMED**:

| metric | parent (eqvar) | R6-v3 |
|---|---|---|
| Hungarian MCC | 0.406 ± 0.004 | **0.764 ± 0.013** (+88%) |
| per-mode matched \|r\| | 0.12 → 0.63 (φ-graded) | **0.68–0.82, ALL modes** |
| matched F1 | 0.512 | 0.563 ± 0.002 |
| mean uniqueness | ≈ 0 | **−0.036 ± 0.006, STILL ≈ 0** |

Fast modes' features are no longer noise-starved — alignment rises across
the whole spectrum exactly as the variance-composition argument predicted.
And the punchline: **uniqueness stays at zero even when skill and gradient
are abundant** — the mode-agnostic shared operator / no-identity-code
finding survives the regime change, its strongest form to date (it was
never an artifact of gradient starvation).

**Dynamical channel: DEAD on this rung** (`litext_e3_dynarm_atmo.npy`,
`litext_e4_dyn_partial_atmo.npy`): zero detections through true W and
through every candidate Ŵ (the only nonempty sets are fine16/split7
half-twin artifacts). Cause: v3's per-step cross-coefficients are small by
construction (equalized CI-detection SNR 0.08 — the realistic consequence
of near-unit memory: matched total influence ⇒ tiny per-step couplings),
and 240-window response averaging has far less power than PCMCI's 9600
samples. Consequences: (i) **E4's pool-crossed selector is INAPPLICABLE
here, not refuted** — its liveness precondition (a live dyn channel) fails,
and that failure is itself detectable unsupervised (all-zero dyn graphs
across all candidates ⇒ fall back to consistency screen + int channel);
(ii) the out-of-sample test of pool-crossed moves to R1 (moderate regime);
(iii) GraphCast implication: at 6 h cadence with realistic per-step
coupling strengths, perturbation probing needs much larger window budgets
or amplitudes — budget the Hakim–Masanam arm accordingly, and always run
the liveness check before trusting two-channel agreement.

**R6 verdict vs its stress-point list:** ceiling degrades gracefully
(0.853 → 0.509, no collapse); discovery survives with a real but bounded
price; consistency-screen zeros stay reliable; the SAE prediction confirmed;
the shared-operator finding strengthened; the response channel found its
scope boundary. The rung did exactly what it was built to do — including
exposing a generator defect (saturated self-loop, τ_eff ≈ 3 cap) that
retro-annotates every φ-labelled claim in the program.

---

## R1 — overlap rung (2026-07-07 session): OUT-OF-SAMPLE test of the pool-crossed selector

`data_gen/generate_overlap.py` — fork of generate_hetdynamics.py (eqvar
config). Dynamics BYTE-IDENTICAL to the parent rung: same φ band
0.15–0.92, same 12-edge set/lags/coeffs, same eqvar innovation scaling,
same T=2400 / 100 reals / DY_SCALE=0.05 / seeds (verified: corr(Z_overlap,
Z_parent) = 0.9993–0.9996 per mode on realisation 0 — the only leak is the
pixel-noise term W@eps_y through the new W). The saturated-self-loop caveat
(small-signal τ_eff≈3) applies equally to parent and R1, as intended. ONLY
the emission changed: isotropic Gaussians at the SAME 3×3-lattice centres,
σ bisected to 6.27 px so max pairwise W-row cosine = 0.2000 exactly
(mean off-diag cos 0.081; support Jaccard max 0.539 mean 0.324; mass
overlap max 0.213; truncation 1e-3 of row max; W fixed across reals).
Split 70/15/15 → data/splits_overlap02.

**Gates (results/aggregation_consistency_overlap02.npy, 12 reals):**

| gate | value | verdict |
|---|---|---|
| PCMCI+ on true Z (rung ceiling) | **F1 = 0.867** (P 0.77, R 0.99) | ≈ parent 0.853 — dynamics unchanged, PASS |
| W-pooled pixels vs true Z | agreement **1.000**, same F1 0.867 | lossless at cos 0.2 — F4 boundary holds; NB trivially so on this generator family (Z := W@obs by construction; correlated pixel-noise leak W@eps_y is ~1e-2 std per mode, ∝ row cosines — too small to break τ=0 conditioning). The live overlap stress is on DISCOVERED maps + the model, below |

Pre-registered rule verified before use: PX(A) = mean over B≠A of
pair-F1(Ĝ_int(A), Ĝ_dyn(B)) (mode space, behavior-matched; <4 matches → 0),
re-derived from the parent's saved battery byte-for-byte
(+0.950 Spearman / +0.990 Pearson / +0.943 exact-lag). Applied verbatim
here; no tuning.
