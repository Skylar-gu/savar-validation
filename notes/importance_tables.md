# Forecast-importance result tables

Generated from `results/forecast_importance_sweep_test.npy` (PDF Phase 5) and
`results/phase8_centrality_vs_importance.npy` (PDF Phase 8). Both on the
held-out **test** split, diurnal CNN. See [[project-savar-pipeline]].

---

## Phase 5 — per-SAE-feature forecast importance (ablation)

Ablate each alive spatial-SAE feature from res3, measure RMSE rise on test.
`importance = RMSE_ablated − RMSE_base`. Base RMSE = **0.6046**, 509 alive
features, 2200 windows.

### Importance by feature type

| Feature type | Mean importance | Interpretation |
|---|---|---|
| Grid-locked (position) | **−4.0e-06** (≈ 0) | Causally inert / epiphenomenal |
| Content | **+5.84e-04** | Carries all forecast-relevant signal |

### Top-10 features by importance

| Rank | Feature | Importance | % of base RMSE | content_R2 | position_R2 | Type |
|---|---|---|---|---|---|---|
| 1 | f8 | 0.00245 | +0.41% | 0.374 | 0.264 | content |
| 2 | f380 | 0.00157 | +0.26% | 0.374 | 0.286 | content |
| 3 | f85 | 0.00107 | +0.18% | 0.321 | 0.263 | content |
| 4 | f396 | 0.00027 | +0.04% | 0.024 | 0.307 | other |
| 5 | f35 | 0.00024 | +0.04% | 0.161 | 0.225 | other |
| 6 | f118 | 0.00019 | +0.03% | 0.008 | 0.543 | other |
| 7 | f80 | 0.00011 | +0.02% | 0.025 | 0.141 | other |
| 8 | f121 | 0.00007 | +0.01% | 0.013 | 0.233 | other |
| 9 | f274 | 0.00006 | +0.01% | 0.041 | 0.065 | other |
| 10 | f368 | 0.00006 | +0.01% | 0.008 | 0.090 | other |

### Importance vs feature R² (Pearson, OOS)

| Predictor | corr with importance |
|---|---|
| content_R2 | **−0.141** |
| position_R2 | −0.027 |

**Takeaway:** grid-locked features are exactly inert and the top-3 are all
content features, but content_R2 does **not** quantitatively predict importance
out-of-sample (the +0.43 seen in-sample was inflation). position_R2 predicts
nothing.

---

## Phase 8 — node forecast importance vs causal centrality

Per-mode input-space ablation on test (base RMSE = **0.6138**, 2200 windows).
Centrality on the ground-truth summary graph (PCMCI-recovered ≥50% has
identical topology).

### Per-node importance and centrality (sorted by importance)

| Node | Importance | var(Z) | out_deg | out_str | in_deg | desc | anc | pagerank |
|---|---|---|---|---|---|---|---|---|
| X1 | 0.03031 | 5.12 | 2 | 0.65 | 1 | 7 | 2 | 0.073 |
| X0 | 0.02585 | 3.68 | 3 | 0.85 | 1 | 7 | 2 | 0.076 |
| X3 | 0.01858 | 3.43 | 2 | 0.45 | 2 | 2 | 3 | 0.110 |
| X6 | 0.01637 | 3.31 | 1 | 0.20 | 2 | 1 | 6 | 0.211 |
| X2 | 0.01625 | 5.07 | 2 | 0.52 | 1 | 7 | 2 | 0.084 |
| X4 | 0.01335 | 3.39 | 1 | 0.35 | 1 | 3 | 3 | 0.070 |
| X7 | 0.00980 | 3.03 | 0 | 0.00 | 2 | 0 | 7 | 0.256 |
| X5 | 0.00120 | 2.77 | 1 | 0.25 | 2 | 2 | 4 | 0.120 |

### Centrality ↔ importance correlations

| Centrality | Pearson | Spearman | Partial (controlling var(Z)) |
|---|---|---|---|
| out_strength | **+0.758** | +0.738 | +0.146 |
| out_degree | **+0.703** | +0.776 | +0.363 |
| descendants | +0.678 | +0.552 | −0.382 |
| total_degree | +0.445 | +0.617 | **+0.563** |
| betweenness | +0.308 | +0.415 | ≈ 0 |
| pagerank | −0.415 | −0.452 | +0.200 |
| ancestors | −0.536 | −0.651 | +0.287 |
| in_degree | −0.589 | −0.436 | +0.480 |
| **var(Z)** (confound) | **+0.682** | +0.810 | — |

**Takeaway:** forecast importance tracks **driving** (out-degree/out-strength,
descendants) and is *negatively* related to being-driven (in-degree, ancestors,
pagerank). But var(Z) alone correlates +0.68/+0.81; after partialling out
variance only total_degree (+0.56) and a weakened out_degree (+0.36) survive —
centrality predicts importance but the amplitude confound must be controlled.
