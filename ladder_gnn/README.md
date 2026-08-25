# SAVAR SAE → PCMCI+ oracle-ablation ladder — **GNN port** (run 2026-08-24/25)

Same protocol as `../savar_sae_pcmci/` (PREREG.md there), re-run on the **MeshGNN** forecaster
instead of the CNN. Question: is the CNN's end-to-end collapse (R3b F1 0.128, R4 ≤ 0.079,
indistinguishable from random features) a property of that architecture, or of what a
predictive representation does to the causal variables?

**Answer: the GNN collapses harder.** GraphCast-matched rung R3b: **F1 0.019** (ladder
protocol; own true-Z ceiling 0.616) / **0.003** (PCMCI+ protocol; ceiling 0.859), 0–1 true
edges recovered against ~100 (ladder) / 8.5 (PCMCI+) false ones. Not distinguishable from
randomly drawn features (p = 0.31). Un-pooled per-node SAE (R4, the closest analogue of a
GraphCast mesh-node SAE): F1 0.004–0.010, p = 0.33–0.93 vs random draws.

## What differs from the CNN ladder

| | CNN ladder (`../savar_sae_pcmci`) | GNN ladder (this dir) |
|---|---|---|
| forecaster | SpatioTemporalCNN, `checkpoints/base/best.pt`, res3 | MeshGNN, `checkpoints/hetdynamics_eqvar/best.pt`, last MP layer H |
| dataset | `data/realisations` (base): T=500, VAR(2), 12 cross edges at lags 1–2 | `data/realisations_hetdynamics_eqvar`: T=2400, 12 cross edges at lags {1,2,3,4,6} |
| τ_max | 2 | 6 |
| activations | (100, 8, 497, 256), pooled by true W | (100, 8, 2397, 256), pooled by true W, **stride 1** (`extract_stride1.py`; stride-5 made every lag unmatchable — Follow-up 1 in savar-project) |
| SAEs | `sae_data/base/sae_mode_*.pt`, `sae_best.pt` | `sae_data/hetdynamics_eqvar/sae_mode_*.pt`, `sae_mixed.pt` (256→512, K=25) |
| PCMCI | `run_pcmci`, τ_min 1, pc_alpha 0.2, α 0.05 ("ladder") | same ("ladder"); **plus** a pass with `run_pcmciplus`, τ_min 0, pc_alpha 0.05 ("plus" = Block G / E1 bake-off protocol, `LADDER_PROTOCOL=plus`) |

Everything else (SEL-VAR, MAP-ID / MAP-R / MAP-FOOT, Hungarian-strict scoring, nulls) is
the CNN code verbatim — `common.py`, `nulls_gnn.py`, `rung_r4_gnn.py` are sed/line ports.

Gate: stride-1 extraction reproduces the Follow-up-1 cache (`activations_stride1_sel.npy`,
44 realisations) to max|diff| 1.2e-5.

## Results — ladder protocol (100 realisations, mean per-realisation F1)

| rung | oracle removed | P | R | **F1** | TP/FP/FN | notes |
|---|---|---|---|---|---|---|
| trueZ | — (ceiling) | 0.448 | 1.000 | **0.616** | 12.0/15.4/0 | α=0.05 over 324 cross-lag slots at T=2400 ⇒ ~16 chance FP; hence the low ceiling |
| R0 | none (replication) | 0.296 | 0.906 | **0.445** | 10.9/26.4/1.1 | 72% of ceiling (CNN: 84%) |
| R1 | sign | 0.296 | 0.906 | **0.445** | | = R0 to every digit (ParCorr is sign-invariant) |
| R2 | feature selection | 0.109 | 0.260 | **0.153** | 3.1/26.3/8.9 | freq 0.108 / pc1 0.445 |
| R3a | per-mode dictionary | 0.114 | 0.329 | **0.168** | 4.0/31.0/8.1 | only **4 distinct** features across 8 streams (f216 wins 3) |
| **R3b** | **N + partition — GRAPHCAST-MATCHED** | **0.011** | 0.089 | **0.019** | 1.1/**99.6**/10.9 | freq 0.037 / pc1 0.040; 7/8 modes matched by MAP-R but max \|r\| only 0.08–0.35 |
| R4 MAP-FOOT | pooling (per-node SAE, 15 reals) | 0.002 | 0.022 | **0.004** | 0.3/116.7/11.7 | footprint cos to true W ≤ 0.52 |
| R4 MAP-R | pooling | 0.006 | 0.050 | **0.010** | 0.6/116.4/11.4 | freq 0.021 / pc1 0.025 |

## Results — PCMCI+ protocol (`--quick`: no R1, no freq/pc1 sensitivities)

| rung | P | R | **F1** | TP/FP/FN |
|---|---|---|---|---|
| trueZ (ceiling) | 0.758 | 0.998 | **0.859** | 12.0/4.1/0 — reproduces the recorded 0.853 anchor |
| R0 | 0.504 | 0.865 | **0.634** | 10.4/10.6/1.6 |
| R2 | 0.197 | 0.211 | **0.202** | 2.5/10.6/9.5 |
| R3a | 0.170 | 0.238 | **0.197** | 2.9/14.2/9.1 |
| **R3b** | 0.003 | 0.003 | **0.003** | **0.0**/8.5/12.0 |

## Nulls (R_null = 10 realisations, 100 draws each; `nulls_gnn_eqvar.npy`, `rung_r4_gnn.npy`)

| rung | null | obs F1 | mean | sd | p95 | max | p |
|---|---|---|---|---|---|---|---|
| R3b | N-RAND | 0.0211 | 0.0133 | 0.0126 | 0.0383 | 0.0440 | **0.31** |
| R3b | N-PHASE / N-SHIFT | 0.0211 | 0.000 | 0.000 | 0.000 | 0.000 | void (point mass at 0 — PX_geo failure mode, as for the CNN) |
| R4 MAP-FOOT | N-RAND | 0.0040 | 0.0041 | 0.0063 | 0.0179 | 0.0283 | **0.33** |
| R4 MAP-R | N-RAND | 0.0100 | 0.0223 | 0.0095 | 0.0371 | 0.0529 | **0.93** (random draws score *higher*) |
| R4 MAP-FOOT | N-PHASE | 0.0040 | 0.0021 | 0.0016 | 0.0050 | 0.0064 | 0.14 |

**Verdicts (prereg rule: positive only if p < 0.05 under N-RAND *and* N-PHASE):**
R3b **NEGATIVE**; R4 **NEGATIVE**.

## CNN vs GNN, side by side (fraction of each ladder's own ceiling)

| rung | CNN F1 (ceiling 0.825) | frac | GNN F1, ladder (ceiling 0.616) | frac | GNN F1, PCMCI+ (ceiling 0.859) | frac |
|---|---|---|---|---|---|---|
| R0 | 0.695 | 0.84 | 0.445 | 0.72 | 0.634 | 0.74 |
| R2 | 0.236 | 0.29 | 0.153 | 0.25 | 0.202 | 0.24 |
| R3a | 0.653 | 0.79 | 0.168 | 0.27 | 0.197 | 0.23 |
| **R3b** | **0.128** | 0.16 | **0.019** | 0.03 | **0.003** | 0.00 |
| R4 (MAP-FOOT / MAP-R) | 0.004 / 0.079 | ≤0.10 | 0.004 / 0.010 | ≤0.02 | — | — |

The one qualitative difference is R3a: the CNN's mixed SAE still separates the 8 oracle
streams (0.653), the GNN's does not (0.168; 4 distinct features for 8 streams). This is the
per-mode-alignment finding (GNN PC0 share 88–92% vs CNN 86%; 2/8 vs 7/8 aligned) showing up
one rung earlier in the ladder.

## Files

`common.py` (helpers, τ_max 6, PROTOCOL switch) · `extract_stride1.py` → `activations_stride1_all.npy` (1.96 GB), `Z_stride1_all.npy` ·
`ladder_gnn.py` → `ladder_gnn_eqvar.npy`, `ladder_gnn_eqvar_plus.npy`, `r3b_series_*.npy` ·
`nulls_gnn.py` → `nulls_gnn_eqvar.npy` · `rung_r4_gnn.py` → `rung_r4_gnn.npy`, `spatial_sae_gnn_eqvar.pt`, `r4_series_mean_gnn.npy`, `r4_footprints_gnn.npy` ·
logs `log_*.txt`. ~/savar-project untouched (read-only).

Ops note: with >1 BLAS thread per worker the 24-way null stage ran at load 114 and made no
progress in an hour; it was restarted with `OMP_NUM_THREADS=1` exported (`run_rest.sh`) and
finished in 19 min. `_init_worker` in common.py sets the env too late (after numpy import).
