# Agent session results — 2026-07-03

Branch `agent/session-2026-07-03`. Plan: `notes/agent_progress.md` +
`notes/agent_session_plan.md`. Numbers first, interpretation second.
Closing section (status table, verdicts, follow-ups) at the bottom.

---

## Block A — VPD on the hetdynamics_eqvar checkpoint

**Question.** Did VPD components collapse on finecadence because the GNN computes
one mechanism (data problem), or because of VPD config (method problem)?
hetdynamics_eqvar has a 22.8× per-mode timescale spread → separable mechanisms
should exist. Baselines (finecadence run `p-0625f7dd`, layer0.mlp0):
PR = 1.14/64, median pairwise gate-map |r| = 0.79, blob-usage CV = 0.04.
Success bar: PR meaningfully > 3/64 AND blob CV ≫ 0.04 with mode-preferential
components.

**Primary run** (`vpd_out/runs/p-6b9ba3ba`, M4 config, C=64, 5000 steps,
byte-identical loss/CI block to p-0625f7dd; only checkpoint + split changed).
Results: `results/vpd_eqvar_redundancy.npy`.

| module | PR (/64) | med pairwise r | med blob CV | #mode-pref (/64) | sp(φ, blob-entropy) |
|---|---|---|---|---|---|
| layers_0_mlp_0 | 1.08 | 0.81 | 0.038 | 0 | −0.08 |
| layers_0_mlp_2 | 1.18 | 0.54 | 0.066 | 1 | −0.32 |
| layers_1_mlp_0 | 1.15 | 0.03 | 0.074 | 6 | −0.30 |
| layers_1_mlp_2 | 1.27 | 0.24 | 0.052 | 7 | −0.66 |
| layers_2_mlp_0 | 1.31 | 0.24 | 0.058 | 4 | −0.48 |
| layers_2_mlp_2 | 1.54 | 0.35 | 0.117 | 12 | −0.31 |
| layers_3_mlp_0 | 1.27 | 0.54 | 0.098 | 6 | −0.05 |
| layers_3_mlp_2 | 1.29 | 0.36 | 0.229 | 14 | +0.34 |

**Outcome: success criterion NOT met.** All 8 modules sit at PR 1.08–1.54, far
below the >3/64 bar; layer-0 blob CV (0.038) is *identical* to the finecadence
baseline (0.04–0.05). The 22.8× timescale spread did not make VPD components
differentiate. There is only mild late-layer movement (layers_2_mlp_2 PR
1.29→1.54, mode-preferential count 8→12; layers_3_mlp_2 7→14 with the
φ-vs-entropy Spearman flipping sign to +0.34 — slow modes attracting slightly
more dedicated components at the output end).

**Secondary run** (config test: C=16, ImportanceMinimality ×30 — the loss-balance
amplifier flagged in results_gnn §5). Run `out/runs/p-d67e2ebb` (5000 steps).
Results: `results/vpd_eqvar_redundancy_c16.npy`.

| module | PR (/16) | med pairwise r | med blob CV | #mode-pref (/16) | sp(φ, ent) |
|---|---|---|---|---|---|
| layers_0_mlp_0 | 1.23 | 0.60 | 0.091 | 0 | +0.37 |
| layers_0_mlp_2 | 1.09 | 0.53 | 0.119 | 3 | −0.16 |
| layers_1_mlp_0 | 1.18 | 0.15 | 0.146 | 6 | −0.62 |
| layers_1_mlp_2 | 1.17 | 0.40 | 0.160 | 7 | −0.35 |
| layers_2_mlp_0 | 1.15 | 0.19 | 0.102 | 5 | −0.10 |
| layers_2_mlp_2 | 1.14 | 0.31 | 0.230 | 10 | −0.26 |
| layers_3_mlp_0 | 1.21 | 0.33 | 0.358 | 4 | +0.56 |
| layers_3_mlp_2 | 1.01 | 0.44 | **0.851** | 12 | +0.50 |

Config amplification does NOT fix the participation ratio: PR stays 1.01–1.23
even with 4× fewer components and 30× stronger minimality pressure. What it
*does* change is where the (single) shared gate pattern sits: in the last MP
MLP the blob-usage CV explodes to 0.85 with 12/16 components mode-preferential
and dominant-mode counts piled on the slowest modes ([2,1,0,0,1,0,**6,6**] for
modes X0..X7) — i.e. all components jointly concentrate on the slow-mode blobs,
but they do not differentiate *from each other* (PR 1.01 there — literally one
pattern). The positive Spearman(φ, blob-entropy) in layer 3 (+0.56/+0.50, sign
flip vs finecadence) says slow modes attract the gates, matching the SAE
finding that slow-mode content is what the network makes decodable.

**BLOCK A VERDICT: data-problem hypothesis NOT confirmed — the evidence now
points at the method/objective.** A 22.8× timescale spread (primary) plus a
4×-fewer-components / 30×-minimality config (secondary) both leave every
decomposed module at PR ≈ 1: VPD's minimality objective collapses to one
shared mechanism on this GNN regardless of whether the data contains separable
mechanisms. The residual mode signal (late-layer blob concentration, slow-mode
preference) shows the *representation* differentiates — the decomposition just
doesn't track it. Follow-up candidates: per-layer C, gate regularization that
penalizes pairwise gate correlation directly, or SPD/APD-style stochastic
masking instead of ImportanceMinimality.

---

## Block B — Impulse-response dynamical test of the frozen eqvar GNN

**Setup.** Frozen `checkpoints_hetdynamics_eqvar/best.pt`; impulse = mode-j
spatial pattern (W_plus column, unit peak) × 1σ pixel std added to the last
input frame; 240 windows × 8 impulses; 12-step autoregressive rollout; response
projected onto W rows → R[i,j,τ]; edges by permutation null (α=0.01, 1000
pixel-permutations). Script `pcmci/impulse_response_gnn.py`, results
`results/impulse_response.npy`.

**Graph recovery (pair level, cross edges only):**
TP=5 FP=3 FN=7 → **P=0.625, R=0.417, F1=0.500** (PCMCI+ on true Z: 0.823).

- Every detected true edge peaks at (or 1 step after) its designed lag:
  X0→X1 (τ̂=1, ℓ=1), X0→X3 (1,1), X1→X2 (2,1), X3→X6 (2,2), X4→X5 (3,2).
- Missed edges are dominated by slow-mode effects: (0→5), (1→4), (3→7),
  (5→6), (6→7) all have effects in X4–X7 (φ≥0.68) whose responses build
  slowly; plus (2→0), (2→3).

**Self-response timescales:** Spearman(τ_design, τ_efold) = **0.738** (n=8),
but e-folding saturates at ≈1.8–2.1 steps for the slow modes (X5–X7 designed
4.0–12.0) — the 12-step rollout window and model damping truncate slow decays.

**Interpretation.** The frozen GNN demonstrably internalized a *directed,
lag-correct* subset of Φ (all five detected edges lag-consistent; precision
0.625 ≫ chance ≈ 0.21), and it orders the modes' intrinsic timescales
correctly, but the internals-free channel recovers only half the graph at this
budget — well short of the PCMCI-on-Z ballpark the success bar asked for.
A 24-step / 480-window sensitivity run (targeting the slow-mode misses) is
reported below.

**Sensitivity run** (24 steps, 480 windows, `results/impulse_response_s24.npy`):
edge detection is *byte-identical* — same TP/FP/FN (F1=0.500), same detected /
missed / false-positive sets, same peak lags. The graph-recovery result is
robust to doubling both the window count and the rollout horizon; the misses
are not a budget artifact. The e-folding estimates, however, are NOT robust:
with a 24-step fit window every mode collapses to τ̂ ≈ 0.8–1.2 (Spearman flips
to −0.71). Beyond ~10 rollout steps the model's response decays at a uniform,
model-imposed damping rate regardless of the designed φ — so (1) the frozen GNN
does not carry the designed slow timescales through long autoregressive
rollouts, and (2) the 12-step Spearman 0.74 should be read as "ordering visible
in the early response only", not as quantitative timescale recovery.

---

## Block C — SAE metric-suite upgrade (SynthSAEBench-style)

**Setup.** `sae/eval_sae_metrics.py`: Hungarian-matched MCC
(`linear_sum_assignment` on the feature×mode |Pearson| matrix — 8 globally
*distinct* features, no double counting), per-mode matched F1 (feature-active
vs |Z| above median), feature uniqueness (matched-mode |r| minus best
other-mode |r|). Per-mode variant scores the union pool of all 8 per-mode SAEs'
features encoded on every stream. Aggregate: `results/sae_metrics_suite.npy`.

| dataset / variant | Hungarian MCC | mean uniqueness | mean matched F1 | old best-\|r\| mean |
|---|---|---|---|---|
| finecadence / per-mode | 0.341 | −0.007 | 0.447 | 0.305 |
| hetdynamics / mixed | 0.387 | +0.030 | 0.485 | 0.393 |
| hetdynamics / per-mode | 0.421 | −0.042 | 0.555 | 0.359 |
| eqvar / mixed | 0.416 | +0.043 | 0.493 | 0.419 |
| eqvar / per-mode | 0.463 | −0.109 | 0.517 | 0.385 |
| eqvar / mixed, 3-seed retrain (final ckpts) | **0.4064 ± 0.0036** | −0.000 ± 0.031 | 0.512 ± 0.012 | — |

Per-mode matched |r| (eqvar mixed): X0 0.12, X1 0.23, X2 0.33, X3 0.37,
X4 0.45, X5 0.58, X6 0.63, X7 0.63 — monotone in φ.

**Reading.**
1. The literature-standard global metric *confirms* rather than deflates the
   old story: Hungarian matching costs almost nothing vs the best-|r| scalars
   (eqvar mixed 0.419 → 0.416), so the old numbers were not double-counting
   features. The dataset ordering finecadence < hetdynamics < eqvar (0.34 →
   0.42 → 0.46 per-mode) survives the metric upgrade — timescale heterogeneity
   genuinely improves feature–mode alignment.
2. Seed variance is tiny (±0.004 MCC): the mixed-SAE result is stable, not a
   lucky init. (Old `sae_mixed.pt` was best-val selected: 0.416; final-ckpt
   seeds: 0.406 — selection on recon MSE was inflating MCC only ~0.01.)
3. **Uniqueness ≈ 0 everywhere** is the new, sharper statement of the failure:
   even the matched feature for a mode correlates almost as strongly with its
   best *other* mode. Features track shared slow content, not mode identity —
   quantitative setup for Block D's dilution prediction.

## Block C2 — KAN-SAE bake-off

**Setup.** `SAE_ARCH=kan` flag on `sae/train_sae_mixed.py` (per-feature 8-basis
RBF spline gates on encoder pre-activations, identity-initialized; TopK +
linear decoder unchanged). Identical data, identical seeds {0,1,2} as Block C's
retrains; scored on FINAL checkpoints (never selected on recon MSE). Paired
table: `results/kan_sae_bakeoff.npy`.

| metric | TopK (mean±std) | KAN (mean±std) | paired Δ (KAN−TopK) |
|---|---|---|---|
| Hungarian MCC | 0.4064 ± 0.0036 | 0.4055 ± 0.0085 | −0.0008 ± 0.0068 |
| matched F1 | 0.5117 ± 0.0123 | 0.4879 ± 0.0496 | −0.0238 ± 0.0511 |
| uniqueness | −0.000 ± 0.031 | −0.004 ± 0.028 | −0.004 ± 0.056 |

**Gate decision: KAN does NOT win beyond seed error bars (it ties MCC, is
noisier on F1) → TopK stays primary.** No Block D/F reruns with KAN codes.
Roadmap consequence: the extractor for GraphCast activations stays the plain
TopK SAE (keeps comparability with the MacMillan & Ouellette GraphCast SAEs);
the saturating-AR nonlinearity is evidently not the binding constraint at this
activation→Z fidelity — consistent with Block C showing the ceiling is mode
identity, not encoder expressivity.

---

## Block E — τ=0 orientation benchmark (Gong et al. 2015 / LiNGAM)

**Question.** Our skew-normal innovations make aliased τ=0 edges (fine lag
ℓ < stride s) identifiable *in principle*. Can existing non-Gaussian estimators
actually orient the edges PCMCI+ leaves `o-o`?

**VAR-LiNGAM** (`pcmci/run_varlingam_orientation.py`,
`results/orientation_benchmark.npy`; 40 realisations, lags = coarse τ_max):

| stride | orient acc (threshold-free \|B0\| contest, GT aliased pairs) | CON-F1 undirected @thr .01 | PCMCI+ CON-F1 (existing sweep) | LAG-F1 | PCMCI+ overall |
|---|---|---|---|---|---|
| 2 | 0.532 (33/62) | 0.629 | 0.63 | 0.633 | 0.66 |
| 3 | 0.519 (40/77) | 0.452 | 0.49 | 0.436 | — |
| 4 | 0.554 (31/56) | 0.280 | 0.38 | 0.345 | — |

**Hybrid one-shot** (plan-sanctioned single follow-up,
`pcmci/run_hybrid_orientation.py`, `results/orientation_hybrid.npy`): PCMCI+
skeleton → residualize on lagged parents → DirectLiNGAM on residuals:
orientation accuracy 0.519 / 0.541 / 0.559 (B0 contest), 0.500 / 0.558 / 0.528
(causal order) for s=2/3/4.

**Reading.** Orientation of the aliased edges is at CHANCE (~0.5) for both the
off-the-shelf VAR-LiNGAM and the hybrid, at every stride. Adjacency recovery
roughly matches PCMCI+ at s=2 and falls behind at s=3,4. So Gong-style
theoretical identifiability does NOT cash out at our T (~ hundreds of coarse
steps), skewness level, and aliasing mix — the non-Gaussian signal surviving
subsampling + the ℓ<s aliasing is too weak in practice. The PCMCI+ 0.08–0.30
orient-rate problem stays open; heavier-tailed innovations or longer series are
the levers if orientation matters downstream.

---

## Block D — Dilution diagnosis via inverse-Ising + community structure

**Question.** Goodfire concept-manifold taxonomy: do modes land in capture /
tiling / dilution? Prediction: mode *identity* in dilution, slow-mode *content*
approaching capture. (`sae/ising_regimes.py`, `results/ising_regimes.npy`;
mixed eqvar SAE, 484/512 features kept at >0.5% activity, pseudo-likelihood
Ising fit on 120k samples, Louvain on |J|.)

Louvain finds **3 communities** (sizes 116 / 247 / 121); within-community
mean |J| = 0.234 / 0.197 / 0.195 vs global 0.169 — real but weak modularity.

| mode | regime | top1 \|r\| | top1/top2 | best comm | top-20 share |
|---|---|---|---|---|---|
| X0 | tiling | 0.118 | 1.18 | 1 | 0.70 |
| X1 | tiling | 0.249 | 1.07 | 1 | 0.85 |
| X2 | tiling | 0.327 | 1.06 | 1 | 0.80 |
| X3 | tiling | 0.373 | 1.06 | 1 | 0.70 |
| X4 | tiling | 0.455 | 1.02 | 1 | 0.80 |
| X5 | tiling | 0.578 | 1.03 | 1 | 0.85 |
| X6 | tiling | 0.628 | 1.11 | 1 | 0.70 |
| X7 | tiling | 0.629 | 1.03 | 1 | 0.75 |

**Reading.** No capture anywhere (top1/top2 ratios 1.02–1.18, capture threshold
1.5). All 8 modes are classified tiling — but they all tile the **same**
community (comm 1, the 247-feature one). This is not eight mode-specific tiles;
it is one shared subspace that every mode's top features live in — the Ising
view of the same phenomenon as uniqueness ≈ 0 (Block C) and steering leakage
(Block F). Content correlation strength is monotone in slowness (0.118 → 0.629
with φ), so slow-mode content is *better represented* but never captured by a
dominant feature.

**Identity signal.** Feature-activation ~ mode-label R²: median 0.030, mean
0.070 — identity is diluted across the dictionary — but with a thin dedicated
tail: 9 features have id-R² > 0.5 (max 0.896), and all of them sit in comm 1
as well. Per-community mean id-R²: comm 1 = 0.119 vs 0.026 / 0.015. So identity
is not a separate community; a few identity-coder features are embedded inside
the one shared content community.

**Caveat (variant design flaw).** The "demeaned = content-only" table is
bit-identical to raw *by construction*: per-stream demeaning subtracts a
constant per column and Pearson correlation is shift-invariant, so the
classifier cannot distinguish the two variants. The identity/content split is
carried entirely by the id-R² analysis above, which is unaffected.

**Prediction scorecard:** "identity in dilution" — half-right (diluted in the
median, but with dedicated identity coders); "slow content approaches capture" —
NO: correlations rise with slowness but stay in tiling.

---

## Block F — Steering dose–response (causal validation of SAE features)

**Question.** Are the Hungarian-matched slow-mode SAE features causal steering
handles? Steer H ← H + α·d_f at the last MP layer (all nodes),
α ∈ {−3,−1,+1,+3}·σ_f, decode, project Δpred onto every mode pattern.
(`sae/steering_dose_response.py`, `results/steering_dose_response.npy`;
240 test windows, targets X5→f216, X6→f475, X7→f143.)

| mode | slope | linearity R² | leakage (off/on) | active-window slope / R² | inactive slope / R² |
|---|---|---|---|---|---|
| X5 (f216) | +0.0161 | 0.670 | **0.812** | +0.0272 / 0.975 | +0.0086 / 0.559 |
| X6 (f475) | +0.0124 | 0.975 | **0.944** | +0.0131 / 0.989 | +0.0119 / 0.968 |
| X7 (f143) | −0.0149 | 0.941 | **0.841** | −0.0171 / 0.992 | −0.0124 / 0.922 |

**Reading.** Dose–response is monotone and linear — in the time-aware variant
(windows where the feature is naturally active) linearity R² reaches 0.97–0.99,
and the active/inactive slope gap is large for X5 (3.2×), confirming the
CFD-GNN paper's point that steering works best in-distribution. X7's negative
slope is just an anti-aligned decoder direction. **But the success bar fails on
specificity: leakage is 0.81–0.94** — at +3σ, steering X5's feature moves X5 by
+0.185 and the *other seven modes* by +0.09…+0.17. Since every W row is
L1-normalized, near-equal pooled responses mean Δpred is nearly uniform across
nodes: the decoder's Jacobian is spatially homogeneous, and the feature acts as
a **global amplitude knob, not a mode-local handle**. Consistent with Blocks
C/D: the features tile one shared subspace, so pushing any of them pushes the
shared direction.

**Outcome: success bar NOT met** (monotonicity yes, leakage ≪ 1 no).

---

## Block G — Aggregation-consistency check on W-pooling (Adag)

**Question.** Does pixel→mode aggregation itself distort conditional-independence
structure, or is the pipeline's causal-signal loss in the representation?
PCMCI+ (ParCorr, τ_max = max fine lag, α = 0.05) on (a) true Z, (b) W-pooled
pixels, (c) ridge readout of W-pooled GNN activations; 24 realisations.
(`pcmci/aggregation_consistency.py`, `results/aggregation_consistency.npy`.)

W row supports are **perfectly disjoint** (max off-diag Jaccard = 0.0000, mass
overlap 0.0000) — the Adag consistency conditions hold trivially for this rung.

| series | F1 vs GT | P | R | F1 vs (a) |
|---|---|---|---|---|
| (a) true Z | 0.853 | 0.75 | 1.00 | 1.000 |
| (b) W-pooled pixels | 0.853 | 0.75 | 1.00 | **1.000** |
| (c) W-pooled GNN activations | **0.020** | 0.02 | 0.02 | 0.032 |

**Reading.** The cleanest localization result of the session. Pooling destroys
*nothing*: (b) recovers the exact same graph as true Z, edge for edge, on every
realisation. The pipeline's actual object — pooled GNN activations — collapses
to F1 0.020. **All causal-signal loss is representation-level, not
aggregation-level.** Caveats: (c)'s ridge readout is itself weak on fast modes
(in-sample |r| 0.17 for X0 vs 0.74 for X7), and the GNN's K=3 input window
smears lags by design — both are properties of the representation pathway being
measured, which is the point. This gates the future overlapping-modes rung:
when blobs overlap, redo the Jaccard check before blaming the representation.

---

## Block H — Koopman/DMD timescale cross-check

**Question.** Architecture-free linear-surrogate readout: does rank-20 DMD on
raw eqvar **pixel data** recover the designed φ timescale spectrum?
(`pcmci/dmd_timescales.py`, `results/dmd_timescales.npy`; 20 realisations,
medians of per-blob matched eigenvalues.)

| mode | X0 | X1 | X2 | X3 | X4 | X5 | X6 | X7 |
|---|---|---|---|---|---|---|---|---|
| τ design | 0.53 | 0.83 | 1.15 | 1.67 | 2.59 | 4.03 | 6.63 | 11.99 |
| τ DMD | 0.33 | 0.30 | 0.70 | 1.23 | 1.33 | 2.10 | 2.79 | 3.15 |
| blob cos | 0.64 | 0.69 | 0.82 | 0.78 | 0.67 | 0.90 | 0.82 | 0.90 |

**Spearman(τ_design, τ_DMD) = 0.976** — the timescale *ordering* is fully
recoverable from pixels by a linear surrogate, and the leading DMD modes'
spatial supports match the W blob footprints (cos 0.64–0.90, best on slow
modes). Absolute timescales are compressed, increasingly so for slow modes
(X7: 12.0 → 3.1, ~3.8×) — consistent with the known DMD eigenvalue-shrinkage
bias under measurement noise rather than anything model-related. Note this ran
on pixel data per the plan (architecture-free baseline); DMD on GNN activation
streams is a cheap unfinished follow-up that would directly test how much of
the dynamics the GNN linearizes internally — and whether the activation
pathway's τ compression matches Block B's rollout damping.

---

## Closing — session status and verdicts

*(Blocks A–E run + written up by the overnight agent; D/F/H computations
launched by the agent and completed ~06:28, analyzed and written up in the
follow-up session; Block G run entirely in the follow-up session.)*

| block | status | headline |
|---|---|---|
| A — VPD on eqvar | **done** | PR ≈ 1 at C=64 *and* C=16/IM×30 → **data-problem verdict NOT confirmed**; collapse is method/objective-level (but layer-3 blob CV 0.85, gates see the timescale lever) |
| B — impulse response | done | F1 0.500 vs PCMCI-on-Z 0.823 (bar not met); all detected edges lag-correct; rollout damping caps recall |
| C — SAE metric suite | done | Hungarian MCC 0.341 (fc) < 0.421 (het) < 0.463 (eqvar), seed-stable ±0.004; uniqueness ≈ 0 |
| C2 — KAN-SAE bake-off | done | paired ΔMCC −0.001 ± 0.007 → tie; **TopK stays primary**, no propagation |
| D — Ising regimes | done | all 8 modes = tiling **in one shared community**; identity diluted (median id-R² 0.03) with 9 dedicated coders |
| E — τ=0 orientation | done | VAR-LiNGAM + hybrid both at chance (0.50–0.56); Gong identifiability doesn't cash out at our T/skew |
| F — steering | done | monotone, linear (gated R² up to 0.99) but leakage 0.81–0.94 → **global knob, not mode handle** |
| G — aggregation consistency | done | pooling lossless (F1 0.853 = true-Z, agreement 1.000); pooled activations F1 0.020 → **all distortion is representation-level** |
| H — DMD timescales | done | Spearman 0.976 vs designed φ spectrum from raw pixels; slow-mode τ compressed ~3.8× (noise shrinkage) |

**Block-A verdict, stated plainly:** the data-problem hypothesis is NOT
confirmed. Giving VPD a checkpoint trained on 22.8× heterogeneous timescales
does not make components differentiate (PR ≈ 1 under both the standard config
and the 4×-fewer-components / 30×-minimality amplifier). The redundant-shredding
collapse is a property of the VPD objective as configured, not of the data.
The gates *do* register the data lever (blob CV 0.038 → 0.851 at the last MP
MLP, slow-mode concentration, sp(φ, entropy) = +0.50) — heterogeneity shows up
inside the single shared gate pattern but minimality never splits it apart.

**The session's through-line.** Every representation-side *measurement* works
(probes, Hungarian MCC ordering, DMD ordering, dose monotonicity), and every
*mechanism-isolation* attempt fails for its own reason (VPD objective collapse,
impulse-response damping, LiNGAM sample inefficiency, steering leakage,
PCMCI-on-activations collapse) — and D/F/G triangulate the same cause: the GNN
concentrates mode information in **one shared, spatially-homogeneous subspace**
that supports linear readout but not per-mode surgery.

**Top-3 follow-ups:**
1. **Explain the G collapse (biggest lever for the GraphCast goal).** Pooling
   is lossless, so the F1 0.853 → 0.020 drop is entirely in the activation
   pathway. Decompose it: (i) rerun (c) with per-lag readouts / longer horizon
   to separate K=3 window smearing from information loss; (ii) DMD on
   activation streams (the cheap H follow-up) to see whether the GNN's internal
   τ compression mirrors Block B's rollout damping. If activations
   fundamentally lack lag structure, no SAE/VPD variant downstream can recover
   the graph.
2. **VPD objective surgery.** The verdict says method-level: gates see the
   lever but minimality never splits components. Try an explicit
   anti-redundancy term (pairwise gate-map decorrelation or orthogonality on
   rank-1 factors) and/or per-component data attribution, on the eqvar
   checkpoint where layer-3 gates already concentrate on slow modes.
3. **Break the shared subspace before the SAE.** C/D/F all point at one global
   amplitude direction absorbing everything (f495-style features, one Ising
   community, uniform steering Jacobian). Project out the shared direction (or
   use group-sparse / decorrelated-dictionary objectives) before training the
   SAE, and re-score with the Block-C suite — uniqueness and leakage are the
   metrics to move.
