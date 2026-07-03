# MeshGNN Forecaster — Consolidated Results

The current forecaster: `train/gnn_forecaster.py` (MeshGNN, ~0.89M params).
Heterogeneous multi-scale mesh over the 50×50 grid (local 8-neighbour edges +
long-range hub–hub edges, node degree 3/8/32), 4 message-passing layers with
**shared** MP MLPs `H ← LayerNorm(H + MLP([H ; ÂH]))`, learned per-node embedding
**off** (`GNN_EMB_DIM=0` — it is a confound for grid-lock; GraphCast has no learned
per-cell table). A GraphCast-faithful directional edge-MLP mode exists
(`GNN_MP_MODE=graphcast`) but all results below are the `gcn` mode.
This is the architecture the GraphCast work will build on.

Sources: `notes/phase7_sae_findings.md` (GNN section), `notes/vpd_results.md`,
memory `project_savar_mode_specialization`, `sae_data_*/`,
`vpd_out/runs/p-0625f7dd/`, `results/pcmci_modes.npy`,
`results/pcmci_subsample_sweep.npy`.

---

## 1. Datasets (all 100 realisations, T_fine = 2400, 70/15/15 chronological split)

| Dataset | Generator | What it adds |
|---|---|---|
| `realisations_finecadence` | `generate_finecadence.py` | same 8-mode edge SET as baseline, but heterogeneous fine lags ℓ∈{1,2,3,4,6}; **nonlinear** dynamics (saturating AR + bilinear coupling); **non-Gaussian** innovations (skew-normal) |
| `realisations_hetdynamics` | `generate_hetdynamics.py` | + widened per-mode self-loop φ ∈ [0.15, 0.92] → 22.8× timescale spread (X0 fastest → X7 slowest) |
| `realisations_hetdynamics_eqvar` | same, `HD_INNOV_SCALE` | + per-mode innovation rescaling → equal mode variances (isolates timescale from amplitude) |

## 2. Forecast skill (1-step-ahead, K=3 input frames)

| Checkpoint | val RMSE | val corr | Context |
|---|---|---|---|
| `checkpoints_finecadence/best.pt` | **0.4295** | ~0.47 | data std ≈ 0.459 → **R² over mean only 0.125**; behaves as damped persistence (corr(pred, last frame)=0.78, gain ≈ 0.27) |
| `checkpoints_hetdynamics/best.pt` | 0.4319 | 0.468 | ≈ 98% of the achievable corr ceiling 0.477 |
| `checkpoints_hetdynamics_eqvar/best.pt` | 0.4352 | 0.456 | — |

Ceiling analysis (hetdynamics): obs-based linear ceiling 0.422 RMSE / 0.477 corr ≈
true-Z dynamics oracle (0.421/0.483) → **latent innovation, not denoising, is the
bottleneck**; the GNN sits at ~96% of achievable corr. Weak aggregate skill is the
*data's* ceiling, not undertraining. Crucially, per-mode 1-step R² of Z spans
0.06→0.54 (9×) on hetdynamics — per-mode heterogeneity exists even though aggregate
skill is modest.

## 3. Causal discovery on the fine-cadence data

- **Full cadence, PCMCI+ on Z** (`run_pcmci_modes.py`): F1 **0.823** (R=1.0,
  P=0.70), sign acc 100% — at dy=0.01 and dy=0.05 alike. Discovery on latents is
  healthy on this data; the hard problems are downstream (features/mechanisms).
- **Subsampling/aliasing sweep** (`run_pcmci_subsample.py`, stride s: fine lags
  ℓ<s alias to τ=0): PCMCI+ recovers aliased contemporaneous edges (CON-F1 0.63 /
  0.49 / 0.38 at s=2/3/4) while plain PCMCI's contemporaneous recall is 0 by
  construction (overall F1 collapses 0.59→0.19). **Justifies PCMCI+ as the
  default.**
- **Orientation gap:** PCMCI+ leaves τ=0 edges unoriented (`o-o`; orient rate
  0.08–0.30) because ParCorr ignores the non-Gaussian innovations that make τ=0
  direction identifiable (LiNGAM) — open follow-up.

## 4. SAE on GNN activations

Target activation: node hidden state H after the **last** MP layer, (B, 2500, 256)
— direct analog of CNN res3; same mode-weighted pooling; same per-mode TopK SAE
pipeline (`extract_activations_gnn.py`, `train_sae_per_mode.py --gnn`).

### 4.1 Finecadence (homogeneous modes) — worse than the CNN, same story sharper

- **2/8 aligned** (X1 0.37, X2 0.35), 0/8 strong, 0/8 monosemantic (CNN: 7/8
  aligned). X1 and X6 share the same best feature f312 — a literal global-activity
  feature.
- PC0 = **88–92%** of variance (CNN 86%); ridge ceilings **lower** (0.36–0.50 vs
  0.36–0.59). SAEs still reach 66–81% of ceiling → the limit is the
  representation, not the SAE.
- Diagnosis: the forecast task barely rewards mode specialization — modes are
  dynamically homogeneous (variance ratio 1.1×, similar φ), so the optimal
  forecaster is one shared scalar shrink applied everywhere.

### 4.2 Hetdynamics / eqvar — the data lever works (for alignment, not identity)

- **Alignment rises monotonically with mode timescale in both datasets**: per-mode
  SAEs X0 r≈0.11 → X5–X7 r≈0.47 (as-is) / 0.53–0.54 (eqvar). Frac-of-ceiling stays
  0.66–0.88 → the network encodes each mode ∝ its forecast usefulness.
- **eqvar ≥ as-is ⇒ timescale, not amplitude, drives specialization** (the
  designed control worked). eqvar: 3/8 strong; as-is: 0 strong.
- **Mixed-mode SAE now beats per-mode** (it failed outright on homogeneous data):
  eqvar X6 r=0.628, X7 r=0.629; as-is X7 r=0.594; some positive specificity
  (as-is X4 +0.37). The Phase-7 "mixed SAE fails" result was a property of
  homogeneous dynamics, not intrinsic.
- **Dictionary-size sweep (N=32…1024): alignment flat** (X7 |r|≈0.63–0.67
  everywhere; N=32 already at 91% of ceiling). Slow-mode content is
  low-rank/high-salience; feature splitting doesn't dilute it. Specificity ≈ 0 at
  every size — the monosemanticity failure is structural, not capacity.
- **But monosemanticity stays 0/8** (specificity ≤ ~0.1). Verified NOT due to
  correlated modes (true Z cross-correlations ≤ 0.09).

### 4.3 Why no mode-specific features — the identity probes

`probe_mode_identity.py` (8-way "which mode is this pooled stream?", chance 12.5%):

| probe | hetdynamics | eqvar |
|---|---|---|
| linear on raw pooled acts | **71.2%** | 70.0% |
| MLP on raw | 72.2% | 71.0% |
| linear after per-mode demeaning | **16.6%** | 16.3% |
| linear on mixed-SAE codes | 71.2% | 70.1% |

- Mode **identity IS decodable** — but demeaning collapses it → identity is a
  **static per-mode mean offset** ("address" inherited from mesh geometry), not a
  dynamic signature. X0/X4 are 100% identifiable (distinctive pooled profiles).
- The SAE **preserves** identity fully (71% from codes) — distributed across
  features, never carved into one-feature-per-mode. TopK allocates features by
  reconstruction variance; a constant offset has near-zero within-mode variance, so
  no feature is spent on it.
- The **content code is shared across modes by construction**: shared-weight,
  equivariant MP MLPs (node_emb=0) put "my local slow value" in the *same channels*
  at every node; mode identity lives only in the pooling weights W[j,:]. Cross-mode
  r's confirm it (e.g. eqvar X5's best feature decodes X7 at r=0.54).
  **Identity-monosemantic features are structurally impossible in this
  architecture–data pair** — equivariance predicts exactly this.
- Geometry (eqvar): corr(PC0-projection, Z_j) rises with timescale 0.13→0.72, yet
  the ridge readout direction is ⊥ PC0 (cos ≤ 0.012) — the clean copy of slow-mode
  state lives in low-variance channels; best SAE features align with that clean
  off-PC0 direction. Figures: `sae_features_3d_{gnn,hetdynamics,hetdynamics_eqvar}[_mixed].png/_hero.gif`.

## 5. VPD (adVersarial Parameter Decomposition) on the frozen finecadence GNN

Run M4 `vpd_out/runs/p-0625f7dd`: all four MP layers (`layers.{0..3}.mlp.{0,2}`),
C=64 each, layerwise `vector_mlp` CI gates, full VPD losses (stochastic + PGD
adversarial recon, frequency-minimality, faithfulness), 5000 steps.

- **Reconstruction converged**: stochastic recon ≈ 2.0e-3, adversarial PGD recon ≈
  3.5e-2, faithfulness ≈ 7.6e-4. The decomposition faithfully reproduces the frozen
  forecaster under worst-case node masks.
- **Depth gradient (real structure):** early layers' gates lock to **mesh hubs**
  (hubC > blobC, 19–26/64 components mode-dominant); layers 2–3 flip to **mode
  blobs** (48–50/64). Early = structural/mesh, late = content.
- **Grid-lock epiphenomenal in one object:** corr(hub-contrast, ablation drop) =
  **−0.48** (layer 0), −0.37 (layer 1) — the more hub-locked a component's gate,
  the less its ablation matters. Reproduces the CNN SAE grid-lock finding without a
  second ablation experiment.
- **Component redundancy (the problem):** every component fires on all 8 mode blobs
  near-equally (blob-usage CV ≈ 0.04) and all 64 stay active. Quantified 2026-07-03:
  the (C,50,50) gate maps have **participation ratio ≈ 1.1 of 64** at every layer —
  effectively ONE spatial gate pattern shared by all components (median pairwise
  gate-map correlation 0.79 at layer 0, with a sign-flipped complement family).
  Components are shards of a single mechanism, differing in scale/sign only.
- Loss-balance note: the *total* loss is dominated by FaithfulnessLoss
  (1e7 × 7.6e-4 ≈ 7600); ImportanceMinimality contributes ~0.15 after its 1e-4
  coefficient and recon ~0.02. Minimality pressure was weak in absolute terms —
  a candidate config amplifier of redundancy, but not the root cause (see
  interpretation).
- Ablation magnitudes are tiny throughout (means 1–3e-3, max 3.7e-2) — expected for
  a ~7%-over-climatology target; read gate *structure*, not magnitudes.

## 6. Interpretation — one story for SAE and VPD

Both tools returned the same verdict because both are reading the same object: **the
frozen finecadence GNN contains essentially one computation** — a damped local
smoothing/shrink-to-mean applied identically at every node — because the data's
homogeneous mode dynamics make that the optimal forecaster (§2). SAEs see its
activation shadow (one dominant PC0 direction, content code shared across modes);
VPD sees its weight shadow (64 rank-1 shards of one mechanism, one gate pattern).
Reconstruction being excellent while components look identical is *consistent*, not
contradictory: recon/faithfulness are satisfiable by any redundant shredding of W,
and minimality only forces differentiation if separable input-dependent mechanisms
exist to find. On this checkpoint they don't.

The hetdynamics experiments prove the lever is the **data**: widening per-mode
timescales creates per-mode forecast value, and SAE alignment follows it
mode-by-mode. The residual failure — no identity-monosemantic features, zero
specificity — is **architectural in a way that is faithful to GraphCast**
(weight-shared processor ⇒ identity lives in position, which enters only as a
static offset), and GraphCast handles it by feeding static geographic inputs
(lat/lon, orography) so identity is available *as content*. That is the next rung.

**VPD has not yet been re-run on the hetdynamics/eqvar checkpoints** — that is the
decisive pending experiment (see [next_steps_plan.md](next_steps_plan.md)).
