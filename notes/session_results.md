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
aggregation-level.** *[CORRECTED by Follow-up 1 below: the (c) collapse was a
stride-5 cadence artifact in the activation extraction, not representation
loss — at stride 1 the pooled activations recover F1 0.855 = true-Z.]* Caveats: (c)'s ridge readout is itself weak on fast modes
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

simplified summary: 
 Block A — VPD decomposition test
  Tried to split the model's inner workings into separate mechanisms, hoping the built-in speed differences (the 22.8×) would force clean separation. It failed — the method collapsed everything
  into one blob. Conclusion: the failure is the method's fault, not the data's.

  Block B — Poke-and-watch test
  Gave the model a nudge on one weather pattern and watched how it responded over time. It got the directions and timing of cause-and-effect right for the fast patterns, but only recovered
  about half the true relationships, and it couldn't hold onto the slow patterns' long timescales.

  Block C — Better scorecard for the feature-finder (SAE)
  Re-scored an earlier tool using a stricter, literature-standard metric. The old results held up, and the sharper finding is that each "feature" can't cleanly tell one weather pattern from
  another — they all track shared background content.

  Block C2 — Fancier feature-finder bake-off
  Swapped in a more flexible version of that tool (KAN) to see if it did better. It tied — no real improvement — so they kept the simpler original.

  Block E — Can we tell which way the arrows point?
  Tested whether existing statistical methods could figure out the direction of certain cause-effect links. They performed at coin-flip level (~50%) — no better than guessing. That problem
  stays unsolved.

  Block D — Where does pattern-identity go?
  Investigated why the features can't separate patterns. Found that all 8 patterns crowd into one shared region of the model instead of getting their own space — identity gets diluted and
  smeared together.

  Block F — Do the features actually steer the model?
  Pushed on individual features to see if they act like clean control knobs for one pattern. Pushing did produce smooth, predictable effects — but it moved all patterns at once, so each feature
  is a global volume knob, not a per-pattern handle.

  Block G — Is the blurring step to blame?
  Checked whether the step that pools pixels into patterns loses the causal signal. It doesn't — pooling was perfectly clean. The signal loss happens inside the model's representation, which
  pinpoints the real culprit.

  Block H — Independent timescale check
  Used a simple, model-free math tool on the raw data to verify the patterns' speed ordering. It nailed the ordering (0.98 correlation), confirming the slow/fast structure is genuinely there in
  the data — even though absolute speeds get squished.


---

# Follow-up session — 2026-07-03 (afternoon)

Executes the closing section's Top-3 follow-ups, in order.

## Follow-up 1 — Explain the Block G collapse (F1 0.853 → 0.020)

**Setup.** `pcmci/explain_activation_collapse.py`,
`results/activation_collapse_explained.npy`. First finding is a *bug-level*
confound: `sae_data_hetdynamics_eqvar/activations_full.npy` is (100, 8, **480**,
256) against T=2400 — the pooled GNN activations were extracted at **stride 5**
(verified: `Z_full` aligns with `latent_states[:, t+K]` only for stride 5).
Block G's series (c) therefore ran PCMCI+ at 5-fine-step cadence while being
scored against fine-unit ground-truth lags (1..6) — every true edge was
unmatchable by construction. Series (a)/(b) ran at stride 1. Fix: re-extract
stride-1 pooled activations for reals 0–23 (PCMCI) + 80–99 (ridge fit), cached
at `sae_data_hetdynamics_eqvar/activations_stride1_sel.npy` (44, 8, 2397, 256).

**(i) Per-lag ridge readouts** (fit on tail-20 reals, |r| held-out on reals
0–23; δ = target offset Z_j(e+δ) from window-end frame e = t+K−1):

| mode | δ=−4 | −3 | **−2** | **−1** | **0** | +1 (fcst tgt) | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| X0 | 0.05 | 0.18 | 0.995 | 0.995 | 0.999 | 0.13 | 0.01 | 0.01 | 0.03 |
| X1 | 0.14 | 0.27 | 0.995 | 0.995 | 0.999 | 0.36 | 0.10 | 0.03 | 0.03 |
| X2 | 0.13 | 0.36 | 0.996 | 0.996 | 0.999 | 0.45 | 0.20 | 0.08 | 0.03 |
| X3 | 0.25 | 0.47 | 0.996 | 0.996 | 0.999 | 0.49 | 0.22 | 0.08 | 0.02 |
| X4 | 0.33 | 0.55 | 0.997 | 0.997 | 1.000 | 0.55 | 0.30 | 0.18 | 0.10 |
| X5 | 0.41 | 0.65 | 0.997 | 0.997 | 1.000 | 0.68 | 0.50 | 0.37 | 0.27 |
| X6 | 0.51 | 0.72 | 0.998 | 0.998 | 0.999 | 0.74 | 0.58 | 0.44 | 0.32 |
| X7 | 0.54 | 0.73 | 0.998 | 0.998 | 1.000 | 0.74 | 0.55 | 0.42 | 0.34 |

All three frames INSIDE the K=3 window are near-losslessly decodable
(|r| 0.995–1.000, **every** mode incl. fastest X0), and the within-window
*change* Z_j(e)−Z_j(e−2) reads out at |r| = 0.997 for all modes. The K=3 conv
does NOT smear lags — the activation keeps the three input frames linearly
separable. Outside the window, |r| decays and is monotone in φ (that decay is
forecast skill / intrinsic memory, not lag structure).

**(ii) PCMCI+ at matched (stride-1) cadence** (ParCorr, τ_max=6, α=0.05,
24 reals — identical protocol to Block G):

| series | F1 vs GT | P | R |
|---|---|---|---|
| (a) true Z (Block G) | 0.853 | 0.75 | 1.00 |
| (c) pooled acts, stride 5, lumped readout (Block G) | 0.020 | 0.02 | 0.02 |
| (c′) stride 1, readout δ=0 (last input frame) | **0.855** | 0.75 | 1.00 |
| (c′) stride 1, readout δ=−2 (first input frame) | 0.847 | 0.74 | 1.00 |
| (c′) stride 1, readout δ=+1 (forecast target, Block G's object) | 0.595 | 0.47 | 0.81 |

**(iii) DMD on GNN activation streams** (rank 20, 20 reals, stride-1 streams;
stacked 8×256-channel matrix, mode matched by block norm; secondary: per-block
DMD, eigenvalue matched by corr with Z_j):

| mode | X0 | X1 | X2 | X3 | X4 | X5 | X6 | X7 |
|---|---|---|---|---|---|---|---|---|
| τ design | 0.53 | 0.83 | 1.15 | 1.67 | 2.59 | 4.03 | 6.63 | 11.99 |
| τ act (stacked) | 0.51 | 0.70 | 0.79 | 0.81 | 1.00 | 2.26 | 1.79 | 3.74 |
| τ act (per-block) | 0.76 | 0.77 | 1.01 | 0.92 | 1.36 | 2.18 | 2.57 | 3.19 |

**Spearman(τ_design, τ_DMD-acts) = 0.976 (both variants) — identical to the
pixel-DMD 0.976.** Slow-mode compression (X7 12.0 → 3.2–3.7, ~3.2–3.8×) matches
the pixel-DMD noise shrinkage, i.e. the GNN's internal representation adds NO
extra timescale compression. Block B's rollout damping (uniform decay beyond
~10 autoregressive steps) is a property of the *generative loop*, not of the
representation.

**Verdict on the key question: activations do NOT lack lag structure.** The
Block G 0.853 → 0.020 drop decomposes as ~all cadence artifact + readout-target
choice: at stride 1 the nowcast readout (δ=0) recovers the graph at
**F1 0.855 = true-Z (0.853)**, edge-for-edge precision/recall. Even Block G's
exact object (the forecast-target readout) reaches 0.595 once the cadence is
fixed — the residual gap vs 0.853 is prediction noise in the readout (|r| 0.13
on X0), not missing lag structure. Consequences: (1) the Block G "all
distortion is representation-level" verdict is RETRACTED; representation-level
loss is ~zero for causal-graph purposes. (2) Downstream SAE/VPD pipelines are
NOT doomed by missing lag structure — the activation pathway carries the full
graph; what they fail at (uniqueness, per-mode surgery) is mode *identity*, not
lag content. (3) All future activation-side causal work must extract at
stride 1 (or match τ units to the extraction stride).

## Follow-up 2 — VPD objective surgery: explicit anti-redundancy term

**Setup.** New loss metric `GateDecorrelationLoss` added to param-decomp
(`param_decomp/metrics/gate_decorrelation.py` + registration; full diff in
`vpd/param_decomp_gate_decorrelation.patch` — param-decomp is its own repo,
left uncommitted there): per decomposed module, flatten the upper-leaky CI
gates over (batch, node) → (N, C), penalize mean squared off-diagonal Pearson
correlation between components; summed over the 8 modules. Config
`vpd/config_gnn_vpd_m4_eqvar_decor.yaml` = Block A's M4 eqvar config + the new
term. Runs: coeff 1.0 → `vpd_out/runs/p-50fb4988`; coeff 10.0 →
`vpd_out/runs/p-87fe946d` (5000 steps each). Results:
`results/vpd_eqvar_redundancy_decor{,_c10}.npy`.

**Coeff 1.0** (same columns as Block A; baseline p-6b9ba3ba in parentheses):

| module | PR (/64) | med pairwise \|r\| | med blob CV | #pref (/64) | sp(φ,ent) |
|---|---|---|---|---|---|
| layers_0_mlp_0 | 1.08 (1.08) | **0.274 (0.81)** | 0.000 (0.038) | 0 (0) | −0.06 |
| layers_0_mlp_2 | 1.15 (1.18) | 0.188 (0.54) | 0.032 | 6 (1) | −0.08 |
| layers_1_mlp_0 | 1.17 (1.15) | 0.092 (0.03) | 0.000 | 0 (6) | −0.33 |
| layers_1_mlp_2 | 1.26 (1.27) | 0.065 (0.24) | 0.034 | 3 (7) | −0.17 |
| layers_2_mlp_0 | 1.32 (1.31) | 0.042 (0.24) | 0.001 | 6 (4) | −0.04 |
| layers_2_mlp_2 | 1.45 (1.54) | 0.040 (0.35) | 0.091 | 11 (12) | +0.05 |
| layers_3_mlp_0 | 1.31 (1.27) | 0.039 (0.54) | 0.102 | 6 (6) | +0.13 |
| layers_3_mlp_2 | 1.39 (1.29) | 0.022 (0.36) | 0.307 (0.229) | 21 (14) | +0.26 |

Training cost: final StochasticReconSubset 4.6e-3 (baseline 1.9e-3), PGD recon
5.9e-2 (4.1e-2), Faithfulness identical (9.8e-4) — the decorrelated
decomposition still reconstructs the frozen GNN. GateDecor 0.476 → 0.007.

**The PR/pattern decomposition (key to reading the table).** Raw PR is
amplitude-weighted; it confounds *pattern sharing* with *amplitude
concentration*. Splitting them (row-normalized gate maps, and restricted to
substantive components with centered-map norm > 1):

| module | PR_pattern base → decor | n_subst base → decor | #pref(subst) base → decor |
|---|---|---|---|
| layers_0_mlp_0 | 1.71 → 5.16 | 59 → 28 | 0 → 0 |
| layers_1_mlp_2 | 3.18 → 15.59 | 64 → 33 | 7 → 1 |
| layers_2_mlp_2 | 4.05 → 24.47 | 64 → 34 | 12 → 4 |
| layers_3_mlp_0 | 3.86 → 26.38 | 64 → 36 | 6 → 5 |
| layers_3_mlp_2 | 4.96 → 34.31 | 64 → 42 | 14 → 8 |

**Coeff 10.0** pushes further in the same direction and over-suppresses:
med|r| 0.002–0.058, but substantive components collapse to 10–17/64,
pattern-PR among them 3.9–13.9, and recon degrades (stochastic 1.4e-2 = 7×
baseline, PGD 1.3e-1 = 3×). The full-64 #pref counts (up to 32/64, layer-3
blob CV 0.946) are carried by near-dead low-amplitude components. Coeff 1.0 is
the operating point.

**Outcome vs the success bar (raw PR > 3/64 with mode-preferential
components): NOT met — but the failure is now cleanly factored.**
1. The anti-redundancy term does exactly its job: the single shared gate
   pattern (median pairwise map |r| 0.81) splits into tens of *distinct*
   spatial patterns — pattern-level PR jumps from 1.7–5.0 to 5.2–34.3 among
   substantive components. The Block-A "one pattern copied 64×" symptom is
   method-fixable.
2. Raw PR stays ≈ 1.1–1.45 because it is dominated by the amplitude hierarchy
   that ImportanceMinimality *by design* imposes (few high-usage components);
   decorrelation acts on patterns, not magnitudes — the two objectives are
   orthogonal, and no decorrelation coefficient trades one into the other
   (coeff 10 kills components instead).
3. The decisive negative: the newly distinct patterns are NOT mode-mechanisms.
   Mode-preferential counts among substantive components do not rise (0–8 vs
   baseline 0–14); dominant-mode counts stay spread; sp(φ, entropy) ≈ 0.
   Given Follow-up 3's finding that pooled activations carry ~no mode-unique
   linear structure (readout uniqueness ≤ 0.024), the parsimonious reading is
   that the GNN's computation genuinely does not factor into per-mode
   mechanisms — it implements one mode-agnostic amplitude-dynamics operator,
   and VPD (now de-redundified) correctly reports that. The Block-A verdict
   should be amended: the *pattern collapse* was method-level; the *absence of
   mode-level mechanisms* looks model-level, not method-level.

## Follow-up 3 — Break the shared subspace before the SAE

**Diagnostics first** (`sae/preprocess_shared_subspace.py`,
`results/shared_subspace_diag.npy`; eqvar pooled activations, train split =
first 85 reals, held-out = last 15):

1. **One direction ≈ the whole readout.** A single ridge direction fit on the
   MIXED stream (target = own-stream Z, no mode label) reads out every mode:
   held-out |r| = 0.12 / 0.35 / 0.43 / 0.48 / 0.55 / 0.67 / 0.71 / 0.73
   (X0..X7) — matching the full 256-dim per-mode readouts.
2. **But deflation is flat.** Project that direction out and refit: mean
   cross-mode |r| = 0.506 → 0.506 → 0.506 → 0.506 over 4 deflations, and
   per-mode ridge ceilings are unchanged after k=1 or k=2 projection. The
   shared signal is HIGH-RANK — every removed direction is instantly replaced.
   There is no small "global amplitude direction" to project out; the C/D/F
   picture needs rewording from "one shared direction" to "one shared
   *signal*, redundantly distributed across the activation space".
3. **No mode-unique linear structure.** Cross-application matrix: mode j's
   readout applied to stream j′ reads Z_j′ as well as j′'s own readout does
   (readout uniqueness = own − best-other: −0.59, −0.35, −0.25, −0.24, −0.18,
   −0.03, +0.004, +0.024 for X0..X7). The pooled activation is a **mode-
   agnostic amplitude encoder**: which blob a vector was pooled from leaves
   ~no linearly usable trace in how Z is encoded. This is the
   representation-level ceiling that made SAE uniqueness ≈ 0.

**Interventions** (each a full datadir; `train_sae_mixed.py` /
`eval_sae_metrics.py` unchanged; 3 seeds, FINAL ckpts;
`results/sae_shared_subspace_break.npy`):
`sae_data_hetdynamics_eqvar_proj/` (k=1 shared-direction projection — the
literal follow-up ask, predicted null by diag 2) and
`sae_data_hetdynamics_eqvar_whiten/` (ZCA whitening, eigenvalue floor
1e-6·λ_max — rebalances the low-variance per-blob identity fingerprints that
per-channel normalization + TopK reconstruction deprioritize; per-mode ridge
ceilings verified unchanged under whitening).

| variant | Hungarian MCC | mean uniqueness | mean matched F1 |
|---|---|---|---|
| baseline (raw acts, 3 seeds) | 0.4064 ± 0.0030 | −0.0000 ± 0.0253 | 0.5117 ± 0.0101 |
| proj (k=1) | 0.4110 ± 0.0068 | −0.0322 ± 0.0284 | 0.5168 ± 0.0040 |
| **whiten (ZCA)** | 0.3201 ± 0.0089 | **+0.0680 ± 0.0158** | 0.3267 ± 0.0115 |
| (Block C reference: eqvar mixed best-val ckpt) | 0.416 | +0.043 | 0.493 |

**Steering leakage rerun** (`sae/steering_shared_break.py`, Block-F protocol,
seed0-final ckpts, variant-matched Hungarian targets;
`results/steering_shared_break_{whiten,baseline_seed0}.npy`):

| variant | X5 leak | X6 leak | X7 leak | mean |
|---|---|---|---|---|
| Block F (old best-val ckpt) | 0.812 | 0.944 | 0.841 | 0.866 |
| baseline seed0-final | 0.816 | 0.916 | 0.775 | 0.836 |
| whiten seed0-final | 0.808 | 0.964 | 0.873 | 0.882 |

**Outcome vs the success bar (move uniqueness ↑ and leakage ↓): half-met.**
1. Projection is a confirmed NULL (MCC/uniq/F1 all within seed noise of
   baseline) — exactly as the deflation diagnostic predicted. The prescribed
   intervention was impossible in principle, and the diagnostic showing *why*
   (high-rank shared signal) is the more valuable artifact.
2. Whitening is the only intervention that moves uniqueness: ~0 → +0.068 ±
   0.016 (>2σ above baseline; 1.6× the previous best). Notably +0.068 exceeds
   the LINEAR mode-uniqueness ceiling (0.024): TopK features are nonlinear and
   can gate on the identity fingerprints whitening amplifies. The price is
   steep — matched |r| drops ~21% (MCC 0.406 → 0.320) and matched F1 ~36% —
   whitening trades tracker strength for specificity.
3. Leakage does not move (0.84 → 0.88 mean, within run noise) and cannot move
   from the SAE side: under the uniform per-node shift protocol, the pooling
   identity (L1-normalized W rows) sends the same α·d to every mode's pooled
   stream, so specificity is bounded by the frozen decoder's Jacobian
   homogeneity — a property of the GNN, not the dictionary.

**Synthesis across the three follow-ups.** The eqvar GNN carries the full
fine-lag content of every mode (FU1: within-window frames decodable at
|r| ≈ 0.999, PCMCI+ from activations = true-Z), but implements it as ONE
mode-agnostic amplitude-dynamics operator: no mode-unique linear structure in
the pooled representation (FU3, uniqueness ceiling 0.024), no mode-factored
mechanisms in the weights even after the decomposition objective is fixed
(FU2, pattern-PR 34/64 but zero mode-preferential gain). "Where does the mode
structure live?" now has a sharp answer: in the *data locations* (W pooling
restores everything), not in *channels* or *mechanisms*. For GraphCast-side
work, the actionable transfers are: extract at native cadence (the stride bug
class), expect global operators rather than per-pattern circuits, and use
whitening if feature-identity specificity matters more than tracker strength.

## Follow-up session closing — status

| follow-up | bar | outcome |
|---|---|---|
| 1 — explain G collapse | localize the 0.853→0.020 drop | **DONE, decisive**: stride-5 cadence artifact; stride-1 activations give F1 0.855 = true-Z; lag structure fully present; DMD-on-acts Spearman 0.976 = pixels; Block G verdict retracted |
| 2 — VPD objective surgery | raw PR > 3/64 + mode-preferential comps | **bar not met, failure factored**: pattern-PR 1.7–5.0 → 5.2–34.3 (collapse fixed), raw PR ≈ 1 is ImpMin's amplitude hierarchy; no mode-preferential gain → mode-mechanism absence is model-level |
| 3 — break shared subspace | uniqueness ↑ from ~0, leakage ↓ from 0.81–0.94 | **half-met**: projection = predicted null (shared signal high-rank); ZCA whitening moves uniqueness to +0.068 ± 0.016 (only mover, at MCC/F1 cost); leakage protocol-bounded, immovable from SAE side |

Corrections to earlier verdicts: Block G's "all distortion is
representation-level" is retracted (FU1); Block A's "collapse is
method/objective-level" is amended — the *pattern* collapse was method-level
(fixed by decorrelation), the absence of *mode-level* mechanisms is
model-level (FU2 + FU3 triangulate a single mode-agnostic operator).

---

# Moving-mechanism sub-resolution session — 2026-07-03 (evening)

Spec: `notes/moving_mechanism_subres_spec.md` (supersedes the position-locked
framing; machinery from `notes/subres_spatial_dealiasing_plan.md`). Sequencing:
T0 gate → movmech generator + train → P1/P2 → T1–T5 → T6.

## T0 — spatial de-aliasing identifiability floor (clean Fourier, no GNN)

**Setup.** `pcmci/subres_identifiability.py`,
`results/subres_t0_identifiability.npy`. 1-D fine grid Lf=64, T=4000,
per-pixel noise σ=√0.05 (SAVAR DY_SCALE), skew-normal AR(1) amplitudes at unit
stationary variance (no amplitude cue). Colliding pairs: (s=4, k 3↔19) and
(s=2, k 3↔29 — the movmech operator's stride). φ_low=0.90 fixed, φ_high swept.
Oracle = Kalman+RTS on the exact collided measurement m(t)=c_lo·a_lo+c_hi·a_hi
(known φ); linear = supervised ridge on a ±12-frame window of all coarse
pixels (fit half 1, corr half 2); baseline = |corr(m, a_high)| = the raw
mixture (0.703 at equal variance — recovery must EXCEED this to count as
de-aliasing).

corr(â_high, a_high), 8 seeds (both configs give identical D_sub numbers —
same collided measurement model):

| Δφ | D_sub oracle | D_sub ridge | D_avg(s2) oracle | D_avg(s2) ridge | D_avg(s4) oracle |
|---|---|---|---|---|---|
| 0.00 (twin) | 0.705 | 0.620–0.655 | 0.011 | 0.034 | 0.138 |
| 0.10 | 0.728 | 0.645–0.683 | 0.008 | 0.053 | 0.149 |
| 0.30 | 0.786 | 0.714–0.749 | 0.003 | 0.091 | 0.192 |
| 0.45 | 0.819 | 0.754–0.786 | 0.001 | 0.120 | 0.225 |
| 0.60 | 0.844 | 0.787–0.815 | −0.001 | 0.152 | 0.256 |
| 0.75 | 0.866 | 0.815–0.839 | −0.003 | 0.188 | 0.287 |

Mixture baseline is flat at 0.703–0.705 for D_sub at every Δφ.

**Reading.**
1. The phenomenon exists at our T/noise/φ-spread: D_sub recovery rises
   monotonically from the mixture floor (0.705 at Δφ=0, the identical-twin
   anchor — exactly no separation) to 0.866 at Δφ=0.75; excess over baseline
   0 → +0.16. A non-oracle windowed ridge tracks the oracle within 0.03–0.09,
   so recovery does not require known dynamics.
2. Operator fork behaves as designed: under D_avg the s2 collided channel is
   destroyed (oracle ≈ 0; c_hi = 0 exactly). The small D_avg ridge climb
   (≤0.19 s2, ≤0.29 s4) is residual attenuated-but-not-annihilated high-k
   leakage (boxcar transfer ≠ 0 off-Nyquist, e.g. 0.346 at s=4/k=19) that the
   full-pixel ridge exploits — for the movmech generator the checkerboard subs
   sit at EXACT Nyquist where the 2×2 block average is identically zero, so
   the β-testbed D_avg control is exact.
3. Δφ dose-response (T5 anchor): identifiability curve is smooth and monotone;
   Δφ ≥ ~0.3 gives a clearly separated (>0.08 excess) recovery.

**Verdict: GATE PASS** — proceed with the full de-aliasing battery (T1–T5) on
the moving-mechanism testbed. Twin-null and dose-response anchors established.

## Generator + plain-net train (MOVE=place)

`data_gen/generate_movmech.py` (fork of generate_hetdynamics): mechanism
identity = (φ_k, parents) IDENTICAL every realisation (eqvar innovation
scaling); blob CENTRE drawn fresh per realisation, rejection-sampled disjoint
(max pairwise footprint Jaccard 0.0). Field generated on a 100×100 FINE grid,
observed through D_sub (decimate → 50×50, aliasing, primary) and D_avg (2×2
mean → 50×50, low-pass control). Two fine sub-sources per parent blob,
CO-LOCATED (checker vs x-stripe patterns at exact Nyquist → both alias to the
same +envelope under D_sub, an exact spatial collision; both annihilated
exactly under D_avg). Sub dynamics: fast φ=0.30 (parent j, lag1) / slow φ=0.90
(parent (j+3)%8, lag2); twin blob 7 = both subs φ=0.60 same parent (T4 null).
`Z_fine` (16, T) = injected sub amplitude saved per realisation, corr 1.000 to
the fine-grid measured readout. 100 realisations, T=2400. Split 70/15/15
(`split_movmech.py`; D_sub + D_avg splits both built).

Plain MeshGNN (gcn, emb0, N_MP=4, hidden=256), 15 epochs, batch 64, on D_sub
coarse obs → `checkpoints_movmech_place/`. Final val corr 0.594, RMSE 0.642.
Per-mode ridge |r| ceiling of the pooled activations (φ-graded, forecast
skill): X0 0.107 → X7 0.730.

## P1/P2/P3 — the left-vs-right test on the PLAIN net

`sae/movmech_position_invariance.py`, `results/movmech_posinv_place_plain.npy`.
Pooled activation feat[r,k]=W[r,k]·H (W follows the moving blob); readout
Z_k ← feat fit on realisations where blob k is LEFT (fine-x<50), tested on
held-out LEFT (in-region) and RIGHT (out-of-region); symmetrised.

| mech | φ-ceil | P1 in R² | P1 out R² | P1 gap | P2 shuffle gap |
|---|---|---|---|---|---|
| X0 | 0.11 | −0.006 | −0.011 | +0.005 | −0.000 |
| X1 | 0.24 | 0.026 | 0.036 | −0.010 | +0.016 |
| X2 | 0.34 | 0.098 | 0.091 | +0.007 | −0.006 |
| X3 | 0.44 | 0.176 | 0.163 | +0.013 | +0.005 |
| X4 | 0.54 | 0.270 | 0.278 | −0.008 | +0.006 |
| X5 | 0.64 | 0.389 | 0.405 | −0.016 | +0.004 |
| X6 | 0.72 | 0.504 | 0.515 | −0.012 | +0.006 |
| X7 | 0.73 | 0.515 | 0.522 | −0.008 | +0.007 |
| **mean** | | **0.246** | **0.250** | **−0.004** | **+0.005** |

**P3 shift-equivariance** (roll input δ px, re-pool at shifted W;
corr(feat_shift, feat_orig)): **1.00 / 0.999 / 0.999 / 0.998** at δ=1/2/4/8,
all 8 mechanisms.

**Reading — the v2 prediction is INVERTED, cleanly.**
1. **No left-vs-right gap** (mean P1 gap −0.004, identical to the P2 shuffle
   gap +0.005). A location code would collapse out-of-region; it does not. The
   absolute recovery is φ-graded and equals the forecast-skill ceiling
   (out-R² tracks the ridge |r|² per mode), i.e. the ONLY limiter is intrinsic
   predictability, not a where-vs-what confound.
2. **P3 = 1.00**: the representation is (near-)perfectly translation-equivariant
   — a shifted blob yields a cleanly shifted internal map, so there is **no
   internal aliasing** for A0/blur-pool to fix (Zhang's premise does not apply
   to this GCN backbone; unlike the strided CNNs the vision papers studied, the
   symmetric-normalised message passing + blob-tracking W-pool is equivariant by
   construction, broken only weakly at the stride-5 hub lattice — invisible here).
3. Therefore the pooled per-mechanism code is **already location-invariant**:
   moving the mechanisms + an equivariant backbone factor out "where" for free.
   The abstraction the hypothesis wanted is present at the pooled level — but
   achieved by equivariance, not by a learned slot/what-code, and the pooling is
   still handed the ground-truth moving footprint (it selects WHERE to read; the
   CONTENT read is position-invariant).

**Verdict vs v2 scorecard.** "plain network fails left-vs-right" — NOT observed
(gap≈0). "A0 smoothing passes / plain fails" — pre-empted: P3=1.0 shows no
aliasing bug exists. Prediction for the sweep: A0/A1/A2 should NOT move P1,
because the plain net already has no position gap. Running the sweep to confirm
(the main result), per spec.
