# VPD Results — SAVAR MeshGNN

Results of running adVersarial Parameter Decomposition (VPD) on the frozen
fine-cadence `MeshGNN` forecaster. Method and design in [vpd_on_savar_gnn.md](vpd_on_savar_gnn.md).

## Headline

VPD decomposes the shared message-passing MLPs into per-node causal-importance
(CI) gate maps over the 50×50 mesh. Two findings:

1. **Depth gradient: mesh-locked early → content-locked late.** Early MP layers'
   gates concentrate on the **mesh hubs** (structural/grid-locked); late layers'
   gates track the **SAVAR mode blobs** (content).
2. **Grid-lock is epiphenomenal — in a single object.** The more hub-locked a
   component's gate is, the **less** its ablation changes the forecast
   (corr(hub-contrast, ablation) ≈ **−0.48** at layer 0). Because `g` is itself
   the ablation-defined importance, "position-locked **and** causally inert" falls
   out of one map — no separate ablation experiment, unlike the SAE route
   ([phase7_sae_findings.md](phase7_sae_findings.md), [[project_savar_gridlock]]).

---

## Runs

- **M4 — all four MP layers** (`vpd_out/runs/p-0625f7dd`, 5000 steps, ~13 min L40S).
  Decomposes `layers.{0..3}.mlp.{0,2}`, C=64 each, layerwise `vector_mlp` CI.
  Full VPD losses on: stochastic + PGD adversarial recon, frequency-minimality,
  faithfulness. This is the run all numbers below come from.
- M2/M3 — single layer `layers.0.mlp.{0,2}` (`vpd_out/runs/p-f77f0dca`, 4000 steps).

**Reconstruction converged** (recon is MSE to the frozen GNN's own output, not to
ground truth): final StochasticReconSubset ≈ 2.0e-3, adversarial PGD recon ≈
3.5e-2, faithfulness ≈ 7.6e-4. The decomposition faithfully reproduces the frozen
forecaster under worst-case node masks.

---

## Per-layer gate structure (M4)

`hubC` = mean gate on hubs − background; `blobC` = mean gate on the 8 mode blobs −
background; `>=diag` = # of 64 components at least as mode- as hub-locked;
`abl` = forecast-recon MSE rise when a component is fully masked (×10⁻³).

| layer         | active | hubC  | blobC | abl mean | abl max | ≥diag |
|---------------|:------:|:-----:|:-----:|:--------:|:-------:|:-----:|
| layers.0.mlp.0 | 64 | **0.479** | 0.369 | 1.86e-3 | 17.3e-3 | 19/64 |
| layers.0.mlp.2 | 64 | **0.258** | 0.113 | 0.62e-3 |  2.1e-3 | 18/64 |
| layers.1.mlp.0 | 64 | **0.441** | 0.327 | 3.08e-3 | 37.1e-3 | 21/64 |
| layers.1.mlp.2 | 64 | **0.222** | 0.149 | 0.20e-3 |  0.6e-3 | 26/64 |
| layers.2.mlp.0 | 64 | 0.151 | **0.340** | 2.40e-3 | 25.1e-3 | 50/64 |
| layers.2.mlp.2 | 64 | 0.006 | **0.227** | 0.10e-3 |  1.2e-3 | 48/64 |
| layers.3.mlp.0 | 64 | 0.118 | **0.365** | 1.24e-3 |  8.2e-3 | 48/64 |
| layers.3.mlp.2 | 64 | 0.041 | 0.049 | 0.03e-3 |  0.9e-3 | 41/64 |

- **Depth flip.** `layers.0/1` have hubC > blobC and only ~19–26/64 components more
  mode- than hub-locked. `layers.2/3` reverse: blobC > hubC, ~48–50/64 mode-locked.
  Early mechanisms lock to mesh structure (hubs); late ones lock to content (modes).
  Matches the design prediction (note §M4: "early local/mesh vs later mode/global").
- The `mlp.0` (input) sublayers carry the strong structure; the `mlp.2` (output)
  sublayers are weaker on both axes — the gating lives where the gate sees
  `concat([H, agg])` and can read hub-vs-cell from the aggregate.

---

## The payoff cross (grid-lock ≠ used)

At `layers.0.mlp.0` and `layers.1.mlp.0`:

```
corr(hub_contrast, ablation_drop) = -0.48   (layer 0)
                                  = -0.37   (layer 1)
```

Hub-locked components are the **least** forecast-relevant. So the mesh
heterogeneity *is* legible in the gate (position decodable), but the computation
barely depends on it (position causally unused). This is the SAE grid-lock finding
([[project_savar_gridlock]]) reproduced without a second experiment: the gate map
is the causal-importance map, so lock-and-inert is one readout.

## Magnitudes — read structure, not size

Ablation drops are tiny throughout (max 3.7e-2, means ~1–3e-3), as expected for a
weak-signal target (~7% over climatology; note §1.1/§10). No single component is
individually forecast-critical; the honest reading is the **spatial structure** of
the gates (hub vs mode, and the depth gradient), not absolute effect sizes.

Also note (per `make_param_decomposition_figure.py`): every component fires on all
8 mode blobs near-equally (blob-gate CV ~0.05) and on hubs simultaneously — there
is **no** clean per-component "tracks mode j" dichotomy. Components are structured
but redundant/distributed, consistent with the rank-1 splitting risk (note §10).

---

## Artifacts

- Gate maps etc. per decomposed matrix: `vpd_out/runs/p-0625f7dd/diagnostics/`
  (`mean_g_map__*.npy` (C,50,50), `hub_usage__*`, `usage_by_mode__*`, `ablation_drop__*`).
- Figures: `figures/param_decomposition.png`, `figures/param_decomposition_faithful.png`
  (single fixed gate scale, honest hub/blob-contrast scatter coloured by ablation).
- Diagnostics regenerator: `vpd/diagnostics.py`; figure: `vpd/make_param_decomposition_figure.py`.
</content>
</invoke>
