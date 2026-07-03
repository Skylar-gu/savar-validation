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

*(further blocks appended as they complete)*
