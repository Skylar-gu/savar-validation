# CNN Forecaster — Consolidated Results

The first-generation forecaster: `train/cnn_forecaster.py` (SpatioTemporalCNN, ~3.2M
params — 3D conv over a k=3 frame window → 3× 2-D ResBlocks → 1×1 head → next frame).
Superseded by the MeshGNN ([results_gnn.md](results_gnn.md)) because the CNN cannot
host grid-locked structure and its architecture is far from GraphCast; the CNN
results remain the reference for the *content* and *cycle* experiments.

Sources: `notes/REPO_SUMMARY_AND_AUDIT.md`, `notes/phase7_sae_findings.md`,
`notes/importance_tables.md`, `notes/rmse_baselines.md`, `results/*.npy`.

---

## 1. Datasets the CNN was trained on

| Dataset | D_y | Cadence | Dynamics | Purpose |
|---|---|---|---|---|
| `realisations/` | 1.0·I | abstract | linear VAR(2) | high-noise baseline |
| `realisations_dy005/` | 0.05·I | abstract | linear | low-noise (reanalysis-like) |
| `realisations_diurnal/` | 0.05·I | 6 h, 2 yr | linear + diurnal/annual forcing | GraphCast-styled variant |
| `realisations_nonlinear/` | 0.05·I | 6 h | saturating AR + bilinear coupling | nonlinearity stress (generated; CNN retrain never completed — effort moved to the GNN) |

All variants share the same 8 modes, W, and edge set (VAR(2), τ_max=2).

## 2. Forecast skill

| Variant | val RMSE | Notes |
|---|---|---|
| baseline (D_y=I) | **1.0723** | oracle floor 1.061, persistence 1.48 → gap to floor 0.011 |
| diurnal | **0.596** (corr 0.64) | lower-variance data; oracle floor for D_y=0.05 is 0.419 |

Oracle floor includes the mode-noise term: `sqrt((1/L)‖W⁺‖²_F + σ²_y)` — 1.061 for
D_y=I, 0.419 for D_y=0.05 (`notes/rmse_baselines.md`).

## 3. Causal discovery (Phase 6)

On privileged latents Z (method calibration):
- PCMCI (ParCorr, τ_max=2): P 0.715 / R 0.984 / **F1 0.825**, sign acc 100%.
- DYNOTEARS: **F1 0.908** (model exactly matched to linear VAR) — but this advantage
  is an artifact of privileged latents (see below).
- TSCI: chance (AUROC 0.500) — correct behavior; no deterministic manifold in a
  stochastic VAR.

On **discovered SAE features** (the realistic case, one Z-aligned feature per mode):
- PCMCI: **F1 0.695** (graceful degradation from 0.825), sign acc 100%.
- DYNOTEARS collapses to F1 0.376 — the linear-model match breaks on noisy nonlinear
  feature proxies. Realistic ranking: **PCMCI > DYNOTEARS > TSCI**.

Diurnal cycle as confounder: raw PCMCI F1 0.293 (FP 58/realisation, same-phase mode
pairs) → **ensemble-mean deseasonalization exactly restores F1 0.825**. On feature
series: raw 0.351 → deseason 0.570 (PCMCI), 0.343 → 0.576 (DYNOTEARS).

## 4. SAE on activations (Phase 7)

Setup: mode-weighted pooling of res3 (`feat[t,j,c] = W[j,:] @ act[t,c,:]`), 8
per-mode TopK SAEs (256→512, K=25). Mixed-mode SAE fails outright — activations
collapse onto a single global-activity direction (**PC0 = 86%** variance, PC1 12%).

Baseline (D_y=I) per-mode results: **7/8 aligned (|r|≥0.35), 0/8 strong (≥0.5),
0/8 monosemantic**; best features reach 77–79% of the per-mode ridge ceilings
(0.36–0.59). Every best feature correlates nearly equally with all modes'
Z — the res3 code tracks **global system activity**, not per-mode states. Hub modes
(X1/X3/X6) have higher ceilings because global activity tracks them better.

Diurnal variant: raw alignment looks better (8/8 aligned, 5/8 strong) but
R²(PC0~cycle)=0.32 — the "strong" features track the shared clock; after
deseasonalization alignment collapses to 2/8. The polysemantic global-activity
encoding is intrinsic to linear-Gaussian generation, not caused by the cycle.

## 5. Grid-locked vs content features (spatial SAE, diurnal CNN)

- The CNN invents position structure **only from zero-padding**: on white-noise
  input, fixed-pattern strength 19× the noise floor, border energy 2× chance;
  159/256 channels grid-locked (`probe_gridlock.py`).
- Spatial TopK SAE: **30 grid-locked features (position_R² ≈ 0.98)** vs **6 content
  features (content_R² ≈ 0.38)** — architecture features more numerous and crisper.
- Coordinate-injection sweep: grid-locked count dose-responds (0→33→52) for ~zero
  forecast payoff (RMSE flat at 0.59).
- **Causal ablation:** zeroing the 30 grid-locked directions (23% of activation
  norm) costs **−0.02% RMSE** — exactly inert. Ablating the 6 content features
  (1.8% of norm) costs **+10.1%**. Grid-locked structure is *decodable but causally
  unused*.
- The "93 effective features" participation-ratio number is an SAE-config artifact
  (ranges 52–213 across dict-size/sparsity); the position-vs-content conclusion is
  hyperparameter-robust.

## 6. Forecast importance & causal centrality (Phases 5 + 8)

- Per-feature ablation, held-out test: grid-locked mean importance ≈ 0; top-3
  important features are all content (f8 +0.41%, f380, f85). content_R² does *not*
  quantitatively predict importance out-of-sample (in-sample +0.43 was inflation).
- Per-mode input ablation vs graph centrality: **out-driving centrality predicts
  forecast importance** (out_degree r=+0.70, out_strength +0.76); in-flavoured
  measures go negative (in_degree −0.59). Confound: mode amplitude (corr +0.68/+0.81
  with importance); after partialling out variance, total_degree (+0.56) and a
  weakened out_degree (+0.36) survive. **Claim with the variance control, or you
  overstate it.**

## 7. What the CNN chapter established (and why it ended)

1. The pipeline machinery works: PCMCI recovery, per-mode SAE + ceiling scoring,
   ensemble-mean deseasonalization (fixes discovery F1 *and* unmasks the
   representation limit), ablation-based causal tests.
2. The representation limit found is a property of **linear-Gaussian data**: the
   optimal one-step map is linear → near-low-rank activations → polysemantic SAE
   features. Not a property of forecasters in general; do not transfer to GraphCast.
3. The CNN **cannot** host GraphCast-style grid-lock: it is translation-equivariant
   except padding, so its only "position" feature is a border artifact. Testing the
   grid-lock-vs-content distinction requires mesh heterogeneity → the MeshGNN.
4. Hence the switch: fine-cadence nonlinear non-Gaussian data + heterogeneous-mesh
   GNN ([results_gnn.md](results_gnn.md)).
