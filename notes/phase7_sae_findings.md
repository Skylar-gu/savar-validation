# Phase 7 SAE Findings

## Result

7/8 modes pass alignment (|r| ≥ 0.35). 0/8 modes are monosemantic.

The CNN's res3 features encode **global system dynamics**, not mode-specific states.

---

## Setup

- **Extraction:** Mode-weighted pooling — `feat_j[t, c] = W[j, :] @ act[t, c, :]` — yields (100, 8, 497, 256)
- **SAE:** TopK, input=256, features=512, K=25
- **Training:** 8 per-mode SAEs, each on one mode's 49,700 samples (100 × 497)
- **Evaluation threshold:** 0.35 (derived from theoretical ceilings below)

---

## Theoretical ceilings

Maximum achievable Pearson |r| between any linear combination of the 256-dim
res3 activations and Z_j (from Ridge regression, out-of-fold):

| Mode | Ceiling | Notes |
|------|---------|-------|
| X0   | 0.49    | border |
| X1   | 0.57    | can exceed 0.5 |
| X2   | 0.48    | border |
| X3   | 0.59    | can exceed 0.5 |
| X4   | 0.45    | below 0.5 |
| X5   | 0.36    | fundamentally limited |
| X6   | 0.58    | can exceed 0.5 |
| X7   | 0.47    | border |

The PCA structure of mode-weighted activations explains the limits:
- PC0: 86% variance (shared global activity across all modes)
- PC1: 12% variance
- PC2+: <1% total

The Z_j signal lives almost entirely in PC0+PC1. Modes with stronger causal
connectivity (X1, X3, X6) have higher ceilings because global activity tracks
hub-node states more reliably.

---

## Per-mode SAE results

| Mode | Best feat | max\|r\| | Ceiling | Frac | Specificity | Status |
|------|-----------|---------|---------|------|-------------|--------|
| X0   | f65       | 0.377   | 0.489   | 0.77 | −0.085      | ALIGN / polysemantic |
| X1   | f306      | 0.446   | 0.575   | 0.78 | −0.017      | ALIGN / polysemantic |
| X2   | f307      | 0.367   | 0.479   | 0.77 | −0.091      | ALIGN / polysemantic |
| X3   | f135      | 0.465   | 0.589   | 0.79 | +0.006      | ALIGN / polysemantic |
| X4   | f478      | 0.353   | 0.450   | 0.79 | −0.118      | ALIGN / polysemantic |
| X5   | f163      | 0.284   | 0.358   | 0.79 | −0.186      | FAIL / polysemantic |
| X6   | f444      | 0.449   | 0.584   | 0.77 | −0.008      | ALIGN / polysemantic |
| X7   | f26       | 0.355   | 0.465   | 0.76 | −0.110      | ALIGN / polysemantic |

**Specificity** = r(feature, Z_j) − max_k r(feature, Z_k): positive means the
feature is more correlated with mode j than with any other mode.

---

## Why features are polysemantic

Every mode's best SAE feature correlates nearly equally with all 8 modes' Z
values. The root cause is that the dominant direction in res3 activations (86%
of variance, PC0) is a **global activity** signal that tracks the overall
excitation level of the system.

This global activity correlates with all modes' Z values because:
1. Any active mode increases the overall observation Y(t), raising global CNN activation
2. Hub modes (X1, X3, X6) have higher correlations (~0.52) because global activity is
   more directly driven by their incoming/outgoing connections
3. Leaf modes (X5, X7) have lower correlations (~0.33) because they're more isolated

Differential analysis (feature maximizing r_j − max_k r_k) confirms this: the
most mode-specific feature for any mode has differential < 0.03 and r < 0.16 —
negligible mode-specific signal.

---

## Interpretation

The CNN's res3 layer has learned to track **global system state** (how active is
the system overall) rather than **individual mode states**. This is still
effective for forecasting — val RMSE = 1.0723 vs oracle floor = 1.061, gap of
0.011 — because future observations depend on the overall dynamics.

This finding motivates Phase 8: modes with higher causal centrality (more
connections in the PCMCI graph) should have their Z values better captured by
the global activity feature. If so, causal centrality and forecast-feature
importance (gradient or ablation) should be correlated.

---

## Pass/fail summary

| Criterion | Result |
|-----------|--------|
| Alignment (|r| ≥ 0.35) | 7/8 PASS |
| Strong alignment (|r| ≥ 0.5) | 0/8 FAIL |
| Monosemanticity | 0/8 FAIL |
| Fraction of ceiling | 77–79% (consistent) |

Phase 7 reports: CNN has mode information but not mode-specific representations.
Global-activity encoding is the dominant strategy.

---

## GNN vs CNN

The same per-mode SAE pipeline, re-run on the **MeshGNN** forecaster instead of the
CNN. This is not one SAE over pooled activations — it is 8 independent per-mode
TopK SAEs (`train_sae_per_mode.py --gnn`), exactly as for the CNN.

- **Target activation:** the node hidden state `H` after the **last** message-passing
  layer, shape `(B, L=2500, 256)` — the direct analog of the CNN's res3 `(B, 256,
  50, 50)`. Same mode-weighted pool `feat[b,j,c] = W[j,:] @ H[b,:,c]`.
- **Model:** `checkpoints_finecadence/best.pt` (val RMSE 0.4295), finecadence data.
- **Extraction:** `sae/extract_activations_gnn.py`, `--stride 5` → 480
  windows/realisation (≈ the CNN's 497, for parity). It also measures the per-mode
  ridge ceilings (`ceilings.npy`), since finecadence+GNN ceilings were unmeasured.

### Per-mode results (GNN)

| Mode | Best feat | max\|r\| | Ceiling | Frac | Specificity | Status |
|------|-----------|---------|---------|------|-------------|--------|
| X0   | f186      | 0.273   | 0.366   | 0.74 | −0.138      | FAIL / polysemantic |
| X1   | f312      | 0.372   | 0.485   | 0.77 | +0.026      | ALIGN / polysemantic |
| X2   | f497      | 0.355   | 0.437   | 0.81 | −0.031      | ALIGN / polysemantic |
| X3   | f116      | 0.345   | 0.495   | 0.70 | −0.039      | FAIL / polysemantic |
| X4   | f376      | 0.277   | 0.358   | 0.77 | −0.071      | FAIL / polysemantic |
| X5   | f46       | 0.250   | 0.361   | 0.69 | −0.135      | FAIL / polysemantic |
| X6   | f312      | 0.307   | 0.465   | 0.66 | −0.093      | FAIL / polysemantic |
| X7   | f213      | 0.263   | 0.363   | 0.72 | +0.066      | FAIL / polysemantic |

### Head-to-head

| | GNN (last MP layer) | CNN (res3) |
|---|---|---|
| Aligned (\|r\| ≥ 0.35) | **2/8** (X1, X2) | 7/8 |
| Strong (\|r\| ≥ 0.5) | 0/8 | 0/8 |
| Monosemantic | 0/8 | 0/8 |
| PC0 variance share | **88–92%** | 86% |
| Ridge ceilings | 0.36–0.50 | 0.36–0.59 |
| Frac of ceiling | 66–81% (noisier) | 77–79% (tight) |

### Interpretation

Same qualitative story as the CNN, but **sharper**. The GNN's per-node state is even
more dominated by a single global-activity direction (PC0 = 88–92% vs 86%), so:

1. **Lower ceilings.** For every mode there is *less* linearly-decodable mode-specific
   signal than in the CNN — a property of the representation, not the SAE (the SAEs
   still recover 66–81% of ceiling).
2. **Worse alignment, zero specificity.** Only X1/X2 clear 0.35; no feature is
   mode-specific (all spec ≤ +0.07). Modes **X1 and X6 share the same best feature
   (f312)** — a literal global-activity feature firing across modes.

Message passing over the heterogeneous mesh collapses per-node state onto global
system activity *even more* than the CNN's convolutions do. Grid-lock / mode-specific
structure does not surface in the node representation — consistent with the VPD
finding that hub-locked mechanisms are epiphenomenal.

Artifacts: `sae_data_gnn/{activations_full,Z_full,ceilings}.npy`,
`sae_mode_{0..7}.pt`, `alignment_per_mode.npy`.

---

## Timescale heterogeneity fixes alignment, not monosemanticity (hetdynamics, 2026-07)

The finecadence collapse above has a cause: the 8 modes are **dynamically
homogeneous** (φ ∈ [0.30, 0.55], variance ratio 1.1×), so the optimal forecaster
is one shared shrink-to-mean gain — nothing rewards per-mode circuitry.
`data_gen/generate_hetdynamics.py` spreads the self-loops to φ = 0.15…0.92
(22.8× timescale spread; **eqvar** variant equalizes variances to isolate
timescale from amplitude). Same MeshGNN recipe; GNN reaches 96–98% of the
data's forecast ceiling (corr ≈ 0.48; latent innovation is the bottleneck).

**Alignment now works, and scales with timescale.** Per-mode SAE best |r| rises
monotonically X0→X7: 0.11 → 0.47 (as-is) / 0.53–0.54 (eqvar) vs 0.25–0.37
finecadence best. Frac-of-ceiling ≈ constant (0.66–0.88): the network encodes
each mode ∝ its forecast usefulness. eqvar ≥ as-is ⇒ **timescale, not
amplitude, drives specialization**. Slow modes' best features sit off the PC0
axis along the ridge Z-readout direction (the clean slow-state copy lives in
low-variance channels ⊥ PC0; ridge ⊥ PC0 at cos ≤ 0.012).

**The mixed-SAE claim above is overturned on heterogeneous data.** One SAE on
all modes pooled (`train_sae_mixed.py`, global norm) *beats* the per-mode SAEs:
X7 |r| = 0.594 vs 0.468 (as-is), X6/X7 ≈ 0.628 vs 0.54 (eqvar); 3/8 strong in
both; 84–86% of ceiling. "Mixed fails because shared PC0" was a property of
homogeneous dynamics + per-mode normalization, not of mixed training.
Dictionary size is irrelevant to this (`sweep_sae_size.py`): alignment flat
from N=32 to N=1024 (even 32 features hit 91% of ceiling); only reconstruction
degrades at small N. Slow-mode content is low-rank and high-salience.

**Monosemanticity still 0/8, and the probes say why**
(`probe_mode_identity.py`). Mode identity IS decodable from the pooled
activations — linear probe 71% (chance 12.5%), MLP no better — but collapses to
~16% after removing per-mode means. So identity is a **static per-mode mean
offset** (an "address" the mesh geometry stamps on each mode's footprint), not
a dynamic signature. The true Z's are nearly independent (|corr| ≤ 0.09), so
correlated latents are NOT the cause. The shared-weight GCN MLPs (node_emb=0)
write "current local state" into the *same channels at every node*; the SAE
feature that decodes Z_7 is a generic "local slow value" detector that decodes
Z_5 equally well when evaluated on mode 5's pooled stream ⇒ specificity ≈ 0 at
every dictionary size. SAE codes preserve identity fully (probe on codes: 71%)
— distributed, never carved into one-feature-per-mode, because a constant
offset has near-zero within-mode variance and TopK features earn their keep by
explaining variance.

**Why timescale differences don't make modes distinguishable per-sample:** an
AR(1) marginal is Gaussian regardless of φ — a single snapshot of a slow and a
fast mode (variance-equalized) is statistically identical; timescale lives in
the *autocorrelation*, which a per-timestep SAE/probe never sees. The ~16%
demeaned-probe residue is the whisper of within-window increment size (3-frame
input). A probe on a temporal *window* of pooled activations should recover
identity from content alone — untested.

Artifacts: `sae_data_hetdynamics[_eqvar]/{alignment_per_mode,alignment_mixed,probe_mode_identity,sweep_sae_size}.npy`,
`figures/sae_features_3d_hetdynamics*[_mixed].png/.gif`.
NOTE: `train_sae_per_mode.py`'s docstring still states the pre-hetdynamics
"mixed SAE fails" claim.

---

goodfire research with weight matrix SAEs
training on 3 months x 2 
more modes? gaussian something? Grid size?
not disjoint? 
SPD?? for stationary -- weights are stationary
models impose dynamical systems into stationary models -- learning stationarity from chaotic system

SAEs forced to do reconstruction where grids are 0-1, SAEs one per mode? 
graphcast weight 