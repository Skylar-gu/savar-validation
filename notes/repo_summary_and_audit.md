# SAVAR Project — Full Repo Summary & Climate-Dynamics / GraphCast Audit

_Written 2026-06-13. Sources: `data_gen/`, `train/`, `pcmci/`, `sae/`, `notes/`, `logs_diurnal/`, memory files._

---

## PART I — What Has Been Built (chronological summary)

### Purpose of the project
- A **controlled synthetic testbed** to validate an interpretability + causal-discovery pipeline before applying it to a real ML weather model (GraphCast).
- Core claim under test: _a CNN trained to forecast a spatial field learns internal structure that (a) reflects a known causal graph and (b) can be recovered with SAEs and linked to causal centrality._
- The system is **SAVAR** (Spatially Aggregated VAR): a linear vector-autoregression on N latent "modes," each painted onto a 2-D grid by a fixed spatial weight, plus per-grid-point noise.

### The generative model (`data_gen/instantiate_model.py`)
- **Spatial domain:** 50×50 grid (L=2500), N=8 modes as **non-overlapping Gaussian blobs** on a 3×3 layout (8 of 9 slots used). W is L1-normalised per mode; `W_plus = pinv(W)`.
- **Dynamics:** linear **VAR(2)** — `links_coeffs` defines Φ(τ) with τ_max=2. Hand-designed graph with: hub (X0), converging node (X3), pure sink (X7), mixed lag-1/lag-2 edges, 3 negative edges, diverging + converging paths, one weak feedback (X2→X0). Spectral radius < 1 (stable).
- **Noise:** `D_x = I_N` (independent latent innovations — needed for PCMCI causal sufficiency), `D_y = λ·I_L` per-grid-point noise.
- **Ground truth stored** in every `.npz`: observations, latent_states Z, G, W, W_plus — Z retained so SAE alignment `corr(feature, Z_j)` is computable.

### Three dataset variants
| Dataset | Noise D_y | Cadence | T | Dynamics | Purpose |
|---|---|---|---|---|---|
| `realisations/` (baseline) | 1.0·I_L | abstract | 500 | linear | High-noise paper baseline |
| `realisations_dy005/` | 0.05·I_L | abstract | 500 | linear | Low-noise (reanalysis-like) |
| `realisations_diurnal/` | 0.05·I_L | **6 h** | **2920 (2 yr)** | linear | **GraphCast-comparison variant** |
| `realisations_nonlinear/` | 0.05·I_L | **6 h** | **2920 (2 yr)** | **nonlinear** | **Tests whether activation collapse survives nonlinearity** (`data_gen/generate_nonlinear.py`) |

- All variants share **identical G, W, and edge set** → PCMCI/SAE results are directly comparable across variants. The nonlinear variant changes only the _functional form_ of each edge (same parents → same edges).
- 100 realisations each (same Earth/graph/weights, different noise seed).

### The diurnal/annual variant (`data_gen/generate_diurnal.py`) — the GraphCast-facing dataset
- **6 h cadence** (matches GraphCast), 2-year series, **D_y = 0.05·I_L** (low-noise).
- Adds **deterministic, physically-motivated forcing** injected in latent space _before_ the VAR recurrence (so it propagates and is amplified through Φ):
  - **Diurnal (24 h = 4 steps):** per-mode phase from "longitude" (blob x-center): `φ_d = −2π·x/nx`, eastern modes lead → westward solar sweep.
  - **Annual (1 yr = 1461 steps):** per-mode phase flips by "hemisphere" (blob y-center).
  - **Afternoon heteroskedasticity:** per-mode innovation variance peaks ~3 h past local noon (convective turbulence), longitude-phased.
  - **12 h semidiurnal tide deliberately dropped** — sits at the 6 h Nyquist, aliases away (as in ERA5/GraphCast).
- **Climatology fixed across realisations** (like W/G); only noise varies per seed.
- After amplitude rebalancing: mean variance share **dynamics+noise 55.6% / annual 32.9% / diurnal 11.5%** — dynamics dominate so PCMCI still works on raw data; realistic ordering annual > diurnal preserved.

### The forecaster (`train/cnn_forecaster.py`)
- **SpatioTemporalCNN**, ~3.2M params: 3D conv collapsing a k=3 frame window → 3× 2-D ResBlocks → 1×1 head → next frame. Single scalar channel, **single-step-ahead** forecast, MSE loss, AdamW + cosine LR, 50 epochs.

### Pipeline phases & status
- **Phase 1a — CNN training** ✅ baseline val RMSE 1.072 vs oracle floor 1.061, persistence 1.48 (gap 0.011, excellent).
- **Phase 5 — RMSE baselines** ✅ (`baselines/rmse_baselines.py`, `notes/rmse_baselines.md`). Oracle floor formula includes mode-noise term: `sqrt((1/L)‖W⁺‖²_F + σ²_y)`; for D_y=0.05, floor = 0.419 (not 0.224).
- **Phase 6 — PCMCI** ✅ baseline P=71.5% R=98.4% F1=82.5%, sign acc 100%. tigramite convention verified: `p_matrix[cause, eff, tau]`.
- **Phase 7 — SAE on CNN activations** ✅ (per-mode TopK SAEs; mixed-mode SAE failed because activations collapse onto a global-activity PC0). Result: modes are _aligned_ but _not monosemantic_; CNN encodes **global system state**, not per-mode states.
- **Phase 8 — causal centrality vs forecast importance** ✅ DONE 2026-06-16 (`sae/phase8_centrality_vs_importance.py`, `results/phase8_centrality_vs_importance.npy`). Per-mode forecast_importance via INPUT-space mode ablation (subtract `W⁺[:,j]·Z_j` from input frames, predict original next frame) on the held-out **test** split (2200 windows, base RMSE 0.614). Centrality on the ground-truth AND PCMCI-recovered summary graphs (the recovered graph at ≥50% recovery has the SAME topology as ground truth → degree correlations identical). **Hypothesis supported at raw level:** out-driving centralities predict forecast importance — out_degree r=+0.70 (Spearman +0.78), out_strength +0.76, descendants +0.68; while IN-flavoured ones go negative — pagerank −0.42, in_degree −0.59, ancestors −0.54. So forecast importance tracks how much a mode *drives* the system, not how much it's driven (hub X0/chain-driver X1 most important; pure sink X7 / near-sink X5 least). **Confound:** input-space ablation scales with mode amplitude — corr(var(Z), importance) = +0.68/+0.81. After partialling out variance, the surviving positive signals are total_degree (+0.56) and a weakened out_degree (+0.36); out_strength (+0.15) and descendants (−0.38) largely wash out. **Verdict: causal centrality predicts forecast importance, but you must control for the amplitude/variance confound or you overstate it** — key caveat for transferring to GraphCast.
- **Stress tests 1b–1f** ❌ not started.

### Diurnal-pipeline results (the GraphCast comparison, 2026-06-13)
- **CNN:** val RMSE 0.596, corr 0.64 (lower-variance data; not comparable to 1.07).
- **PCMCI:** raw F1=0.293 (FP 58 — cycle is a shared confounder) → **deseasonalized** (Z − ensemble mean) **F1=0.825, exactly restoring baseline**. Cycle is a clean, removable confounder.
- **SAE raw:** 8/8 aligned, 5/8 strong, only 2/8 monosemantic; PC0 var 86→88.7%; R²(PC0~cycle)=0.32 — the "strong" features mostly track the **shared clock**, inflating alignment while hurting specificity.
- **SAE deseasonalized:** R²(PC0~cycle)→0.00 (cycle removed) but alignment **collapses** to 2/8 aligned, 0/8 strong. **Punchline: the raw "strong" features were entirely cycle-tracking artifacts**; the residual low-noise dynamics carry even less mode signal. The CNN's polysemantic global-activity encoding is intrinsic, not caused by the cycle.

### Grid-locked vs physics features — architecture-induced position structure (2026-06-15)
Goal: reproduce, *inside SAVAR*, the paper's split between **grid-locked** (architecture/position) features and **physical-abstraction** (content) features, and test whether it's causal. All on the diurnal CNN. Scripts: `sae/probe_gridlock.py`, `extract_spatial.py`, `train_spatial_sae.py`, `physics_vs_position_tradeoff_sweep.py`, `probe_variance_diag.py`, `refit_content_r2.py`, `probe_gridlock_ablation.py`, `probe_feature_splitting.py`, `probe_feature_splitting_coordcnn.py`.

- **Step 1 — Does the CNN invent position structure on symmetric data? Yes** (`probe_gridlock.py`, fig `gridlock_step1.png`). The CNN is translation-equivariant except for zero-padding, so any spatial structure surviving on *structureless white-noise input* is architecture-induced. Time-mean res3 map: **fixed-pattern strength 19× the sampling-noise floor**, **border energy 2.0× chance**, **corner 3.3× chance**, **159/256 channels grid-locked**. On real input the 8 blobs light up and the border artifact drops below chance (content masks it). Unambiguously padding-induced.
- **Step 2a/b — Does a spatial SAE isolate it? Yes, crisply** (`extract_spatial.py` → per-pixel res3 acts tagged with position+content+mode; `train_spatial_sae.py`, fig `gridlock_step2_features.png`). The per-mode pipeline can't show this — it W-pools space away before the SAE. The spatial TopK SAE yields **30 grid-locked features at position_R²≈0.98** (single-edge detectors: one each for top/left/right/bottom) vs **6 content features topping out at content_R²≈0.38**. Reproduces the paper's bucket ordering: architecture features more numerous and far crisper than physics features.
- **Step 2c — Physics-vs-position trade-off sweep** (`physics_vs_position_tradeoff_sweep.py`; mean ± sd over 3 seeds). Inject scaled x/y coordinate channels (dose = `strength`), retrain CNN per dose, re-probe:

  | strength | val RMSE | #grid-locked | median pos_R² | max content_R² |
  |---|---|---|---|---|
  | 0 | 0.600 ± 0.000 | 0.0 ± 0.0 | — | 0.440 ± 0.020 |
  | 1 | 0.590 ± 0.000 | 32.7 ± 12.7 | 0.911 ± 0.031 | 0.558 ± 0.038 |
  | 3 | 0.590 ± 0.000 | 52.0 ± 7.8 | 0.946 ± 0.009 | 0.532 ± 0.101 |

  Reads: (1) **position buys ~1.6% skill and saturates** (RMSE flat, sd 0.000) — position is a near-useless shortcut for this task. (2) **Grid-locked count dose-responds 0→33→52** (noisy ±, direction solid). (3) **median pos_R² robustly high and rising** (0.91→0.95). (4) **max content_R² is increase-then-noise**: the 0→1 rise is real, the 1→3 "dip" is inside the ±0.10 spread. **Headline: injecting position makes the SAE spend growing, crisp capacity on position features for ≈zero forecast payoff.** *(Caveat: this table's `max content_R²` column was computed with the old one-hot metric and predates the cont_R2 fix below; the re-fit shows the one-hot moves max content_R² by only ~0.007, so the **earlier "the 0→1 rise is the one-hot confound" reading is superseded** — the rise is not driven by the one-hot. The table has not been regenerated with the corrected metric; the grid-locked / pos_R² columns are unaffected.)*
- **Why `n_content`=0 in every condition (incl. control) — a rigged metric, not a finding.** The content bucket requires `cont_R2 > pos_R2`. On SAVAR the 8 modes sit at *fixed locations* (physics ≈ position), and `pos_R2` is scored at per-pixel (2500-group) resolution while `cont_R2`'s spatial term is only a 9-level mode one-hot — so per-pixel position always explains ≥ as much variance and wins the tie. `n_content`≈0 by construction. Treat `#content`/`max content_R²` as a noise floor; trust `#grid-locked` and `median pos_R²`.
- **cont_R2 confound — fixed, and it was NOT suppressing content (`refit_content_r2.py`, 2026-06-15).** Dropped the mode one-hot from the content design matrix entirely; `cont_R2` is now OLS R² on `[1, target, target²]` only (patched in `train_spatial_sae.py` and `physics_vs_position_tradeoff_sweep.py`). Pure OLS re-fit on the *cached* diurnal SAE (no retrain): **max content_R² 0.375 → 0.374**, taxonomy unchanged (30 grid-locked, 6 content), **0 features reclassified**, mean |Δcont_R²| = 0.007. So the low content score is **genuine, not an artifact of the position-proxy one-hot**. (The one-hot was harmless because position is already captured by the per-pixel `pos_R2` it was being compared against; the physics signal in res3 is just weak.)
- **Variance concentration & feature-splitting — the "93 effective features" is largely an SAE-config artifact (`probe_variance_diag.py`; `probe_feature_splitting.py` diurnal; `probe_feature_splitting_coordcnn.py` CoordCNN, 2026-06-15).** The participation ratio is *not* a stable property: on the CoordCNN (strength 1) it ranges **52→213 effective features** across a dict-size × sparsity grid (smaller dict and sparser/lower-K codes both shrink it; the same `(512,25)` config reads 93 *without* dead-feature resampling vs 213 *with* it — proving the count tracks training choices, not the model). Sparser codes concentrate variance into fewer, **more redundant top directions** (top-10 decoder cosine redMax 0.29→0.47 as K drops), confirming the clone/feature-splitting signature. The grid-locked features still dominate in *count and selectivity, not energy* (≈8.8% of total variance at the diurnal default).
  - **But the position-vs-content picture is hyperparameter-robust.** Across all 9 CoordCNN configs **max pos_R² stays 0.98–0.996, max content_R² stays 0.29–0.43, #content = 0**; the diurnal grid is the same story (maxPos ~0.95+, maxCont ~0.4, #content tiny). Re-running the SAE with different dictionary size / sparsity does **not** change the content/position conclusion — position is always crisply decodable, content always weak.
- **Causal ablation — grid-locked directions are epiphenomenal (`probe_gridlock_ablation.py`, 2026-06-15).** Zeroed the grid-locked SAE directions out of res3 (subtract their decoder contribution per pixel) and re-ran `model.head` on 3,504 val windows. Despite the 30 grid-locked features carrying **23% of the per-pixel activation norm, ΔRMSE = −0.02% (0.5779→0.5778)** — no forecast cost. Controls on the same machinery: ablating the **6 content features (only 1.8% of norm) costs +10.1%**; a **random alive set of the same count (16.7% of norm) costs +3.0%**; the **full SAE reconstruction residual costs +34.8%**. So the grid-locked structure is **decodable but causally unused** — confirming the flat RMSE sweep and the 8.8%-variance reframe. This is the direct answer to "is position used or just decodable": **just decodable.**
- Outputs: `figures/gridlock_step1.png`, `gridlock_step2_features.png`, `physics_vs_position_tradeoff_sweep.png`; `results/physics_vs_position_tradeoff_sweep.npy`, `probe_variance_diag.npy`, `gridlock_ablation.npy`, `feature_splitting_sweep.npy`, `feature_splitting_coordcnn.npy`; `sae_data_diurnal/content_r2_refit.npz`.

### Phase 6 (PDF) — Causal discovery on SAE **feature** time series (2026-06-15)
`pcmci/run_pcmci_features.py`. The existing `run_pcmci.py` runs PCMCI on the ground-truth latent modes Z (a method check, F1=0.825). The PDF's Phase 6 runs discovery on the **discovered SAE features** — the realistic case. Built an 8-variable feature time series by encoding each mode's activations through its per-mode SAE and taking the Z-aligned feature (`alignment_per_mode` `best_feat`, sign-flipped to +align with its latent), then ran the **same** PCMCI (ParCorr, tau_max=2, α=0.05) against the **same** ground-truth G. Original stationary dataset (`sae_data/`, `data/realisations`, 100 realisations).

| metric | latent Z (baseline) | **SAE feature** |
|---|---|---|
| Precision | 0.715 | **0.629** |
| Recall | 0.984 | **0.789** |
| F1 | 0.825 | **0.695** |
| Sign accuracy (TP) | 100% | **100%** |

Reads: **discovery on imperfect, discovered features recovers the causal graph with graceful degradation** (F1 0.825→0.695) and perfect sign recovery. Recovery tracks edge strength AND feature-alignment quality: strong lag-1 edges into well-aligned modes recover ~90–100% (X0→X1 92%, X1→X2 100%, X3→X6 98%, X4→X5 100%), while the hardest cases collapse — `X3(t-2)→X7` (coeff −0.15) 33%, `X6(t-1)→X7` 63%, `X2(t-2)→X0` 54% (lag-2, small coeff, and/or X7/X5 weak alignment |r|≈0.36/0.28). Dominant false positive `X3(t-2)→X6` (34%) is a lag-2 echo of the true `X3(t-1)→X6`. MI pre-screen (Step 1) keeps 8/12 true pairs; PCMCI's PC step does the real screening. **Not yet done:** diurnal variant (needs deseasonalized feature series — non-stationary), and orientation of any contemporaneous links (none at τ≥1 here). Output: `results/pcmci_features.npy`.

### Phase 6 — alternative causal discoverers: DYNOTEARS & TSCI (2026-06-15)
Implemented two CausalDynamics-leaderboard baselines (kausable.github.io/CausalDynamics) as drop-in alternatives to PCMCI, scored on the **same latent-Z modes, same ground-truth G** as `run_pcmci.py` (100 realisations, `data/realisations`). Runner: `pcmci/run_baselines.py --method {pcmci,dynotears,tsci}`. Shared scoring extracted to `pcmci/eval_common.py` (adds summary-graph collapse + AUROC/AUPRC alongside the lag-resolved P/R/F1). Routing PCMCI through the new runner **reproduces the baseline exactly (P 0.715 / R 0.984 / F1 0.825 / sign 100%)** — regression guard on the refactor.

| Method | Precision | Recall | F1 (lag-resolved) | Sign acc | AUROC (summary) | AUPRC |
|---|---|---|---|---|---|---|
| PCMCI (ParCorr) | 0.715 | 0.984 | **0.825** | 100% | 0.995 | 0.990 |
| **DYNOTEARS** | 0.839 | 0.995 | **0.908** | 100% | 1.000 | 0.998 |
| **TSCI** | — | — | — (summary F1 0.30) | — | **0.500** | 0.277 |

- **DYNOTEARS *beats* PCMCI here (F1 0.908 vs 0.825)** — its linear structural-VAR model is *exactly* matched to SAVAR's linear VAR(2) generative process, so it recovers every edge (all 12 at ≥94%, the −0.15 lag-2 edge at 94% vs PCMCI's 86%) with fewer false positives (2.5 vs 5.0/realisation). Dominant FP is the lag-2 echo of a lag-1 edge (`X1(t-2)→X2`, 14%). Perfect sign recovery.
- **TSCI degenerates to chance (AUROC 0.500)** — and this is the honest, expected result, not a bug: TSCI assumes a *deterministic dynamical system* with a smooth attractor/vector field (it pushes forward finite-difference velocities through a tangent-space Jacobian). SAVAR is a *stochastic* VAR with near-white innovations, so the velocity field is noise-dominated and there is no manifold to exploit. Auto lag/dim selection (`--tsci_auto`) only nudges it to AUROC 0.545 — still uninformative. Matches the leaderboard's weak TSCI on simple/stochastic settings. Verified faithful to upstream by reproducing the paper's ~0.99 separation on the coupled Rössler–Lorenz system.
- **Takeaway for method choice on SAVAR-like (linear, stochastic) data:** DYNOTEARS ≥ PCMCI (model match); CCM-family methods like TSCI are the wrong tool (no deterministic manifold).

**Implementation notes.** DYNOTEARS uses `causalnex`, which pins `numpy<1.24 / pandas<2.0` — incompatible with the main env (numpy 2.0 for tigramite/torch). So it runs in an **isolated venv** (`.venv_causalnex`, gitignored) via a shell-out worker: `pcmci/methods/dynotears.py` (main env) → `pcmci/methods/dynotears_worker.py` (venv). TSCI (`KurtButler/tangentspaces`, NeurIPS 2024, MIT) is **vendored & trimmed** into `pcmci/methods/tsci/` (numpy-based ACF; jaxtyping/tqdm/statsmodels removed; optional `bmi` MI path lazy-imported) so it runs in the main env with no extra installs. Tests: `tests/test_baselines.py` (edge/lag/sign conventions on a planted 3-node VAR; DYNOTEARS skipped if venv absent). Outputs: `results/baseline_{dynotears,tsci}.npy` (PCMCI baseline still at `results/pcmci_results.npy`, untouched).

### Phase 6 — three methods on SAE **feature** series, across dataset variants (2026-06-16)
Ran all three discoverers on the **discovered SAE feature** time series (not the privileged latents) across four data variants, via `pcmci/run_baselines.py --features [--dy005|--diurnal [--deseason]]`. Feature construction mirrors `run_pcmci_features.py` exactly (per-mode SAE, Z-aligned `best_feat`, sign-flipped); routing PCMCI-features through the runner reproduces the canonical `pcmci_features.npy` (F1 0.695) exactly. 100 realisations each. Figure: `figures/baseline_methods_comparison.png`. Results: `results/baseline_{method}_features{_dy005|_diurnal|_diurnal_deseason}.npy`.

| method | variant | AUROC | AUPRC | F1 | sign | mean FP |
|---|---|---|---|---|---|---|
| PCMCI | regular | **0.923** | 0.863 | **0.695** | 100% | 5.9 |
| PCMCI | dy005 | 0.930 | 0.864 | 0.680 | 100% | 6.5 |
| PCMCI | diurnal (raw) | 0.796 | 0.633 | 0.351 | 100% | **38.1** |
| PCMCI | diurnal deseason | 0.904 | 0.844 | 0.570 | 100% | 13.3 |
| DYNOTEARS | regular | 0.725 | 0.623 | 0.376 | 100% | 22.9 |
| DYNOTEARS | dy005 | 0.857 | 0.716 | 0.545 | 100% | 13.6 |
| DYNOTEARS | diurnal (raw) | 0.697 | 0.432 | 0.343 | 96% | 27.4 |
| DYNOTEARS | diurnal deseason | 0.915 | 0.789 | 0.576 | 100% | 9.9 |
| TSCI | regular | 0.502 | 0.262 | 0.301 | — | — |
| TSCI | dy005 | 0.507 | 0.280 | 0.309 | — | — |
| TSCI | diurnal (raw) | 0.480 | 0.258 | 0.284 | — | — |
| TSCI | diurnal deseason | 0.500 | 0.262 | 0.295 | — | — |

- **The latent-Z ranking *inverts* on discovered features.** DYNOTEARS's privileged-latent dominance (F1 0.908 on Z) collapses to **F1 0.376** on features: the SAE features are noisy nonlinear proxies (|r|≈0.28–0.47 regular), breaking the exact linear-VAR model match. **PCMCI is the strongest feature-space method** (F1 0.695, AUROC 0.923) — its nonparametric CI test tolerates the feature distortion better than NOTEARS's global linear fit. So "DYNOTEARS > PCMCI" was an artifact of feeding privileged latents; realistic ranking is **PCMCI > DYNOTEARS > TSCI**.
- **The diurnal cycle is a strong confounder; deseasoning fixes it (both lag-resolved methods).** raw→deseason: PCMCI F1 0.351→0.570 (FP 38→13), DYNOTEARS 0.343→0.576 (FP 27→10). The shared diurnal/annual forcing makes all modes co-oscillate → spurious lagged edges; subtracting the ensemble-mean climatology roughly halves FPs and restores ~baseline.
- **Correction to an earlier prediction:** ParCorr conditioning did **not** weather the cycle better than DYNOTEARS — under the raw cycle PCMCI has the *most* FPs (38.1 vs 27.4). The cycle is an unobserved common forcing (never in the conditioning set), so ParCorr's significance tests flag all cycle-induced lagged correlations; DYNOTEARS's L1 sparsity is more conservative. **Sparsity beat significance-testing on cycle resistance here.**
- **TSCI stays at chance on diurnal too (no synchrony edge inflation).** Diurnal-SAVAR adds a deterministic *additive forcing* but the *intrinsic* dynamics remain stochastic VAR, so TSCI still has no deterministic manifold. ⇒ **Diurnal-SAVAR is NOT a GraphCast proxy**; the GraphCast question (intrinsically deterministic emulator) is the real TSCI test and remains open.
- dy005 ≈ regular (same regime, didn't degrade; DYNOTEARS even higher at 0.857 — read as estimator noise, not a real effect). **Caveat:** feature-space deseasoning is approximate — the TopK encoder is nonlinear, so ensemble-mean subtraction isn't the exact forced/noise split that latent-Z deseasoning gives.

**Nonlinear CI tests wired for the GraphCast path.** `run_baselines.py --cond_ind_test {parcorr,gpdc,cmiknn}` selects the PCMCI conditional-independence test (same PCMCI+ algorithm / `p_matrix` format). ParCorr=linear/Gaussian (fast); GPDC=GP+distance-corr (nonlinear, slow, needs `dcor`); CMIknn=kNN CMI (any dependence, slowest, needs `numba`). `dcor`/`numba` installed without disturbing pinned numpy 2.0.2. gpdc/cmiknn give **unsigned** scores (sign accuracy auto-disabled); non-parcorr runs save to `..._{gpdc,cmiknn}.npy`. Intended for GraphCast, where ParCorr would miss nonlinear links.

### Repo layout (reorganised 2026-06-13; now under git as of 2026-06-15 — `mv`-reversible moves, history via git)
- `data_gen/`, `train/`, `pcmci/`, `sae/`, `baselines/`, `visualization/`; outputs in `results/`, `logs/`, `logs_diurnal/`, `checkpoints*/`, `sae_data*/`. **All scripts must run from project root.** Orchestrator `run_diurnal_pipeline.sh` (+ `run_sae_deseason.sh`).

---

## PART II — Audit: Fidelity to Real Climate Dynamics

**Bottom line: this is a synthetic causal-graph testbed, not a climate emulator. Physical fidelity is intentionally low; the diurnal/annual layer adds _surface_ realism but the engine underneath is linear stochastic and non-physical. That is a defensible design choice for the stated goal, but several claims that lean on "climate-like" or "GraphCast-like" realism are weaker than they read.**

### Where it is climate-faithful (genuine strengths)
- ✅ **6 h cadence** matches GraphCast/ERA5 exactly; lag structure (6 h, 12 h) is the natural sampling.
- ✅ **Diurnal cycle phased by longitude** = a real solar sweep; eastern-leads-western is physically correct.
- ✅ **Annual cycle with hemispheric phase flip** is physically correct (N/S seasons opposed).
- ✅ **Afternoon-peaked heteroskedasticity** is a real signature of convective turbulence; verified per-mode peak at the correct local-afternoon slot.
- ✅ **Semidiurnal tide dropped at Nyquist** — correct and honest (a real 6 h product can't resolve it either).
- ✅ **Low-noise regime** is a fair stand-in for reanalysis being "observation-clean."
- ✅ **Fixed climatology / per-seed weather** mirrors "same Earth, different realisation."
- ✅ Stationary, stable VAR with a known graph — exactly what's needed to make causal-discovery results _attributable_.

### Where it is NOT climate-faithful (the gaps)
- ❌ **Linear dynamics.** The atmosphere is governed by nonlinear primitive equations (advection, Coriolis, pressure gradients, moist thermodynamics, baroclinic instability). SAVAR is linear VAR(2). No nonlinearity, no conservation laws, no fluid mechanics, no chaos/sensitive dependence.
- ❌ **No spatial transport.** Modes are **fixed Gaussian blobs**; nothing advects, propagates, or moves across the grid. Real fields have travelling waves, fronts, and storm tracks. The grid's spatial structure is essentially static per-mode footprints — the only spatial "physics" is the fixed projection W.
- ❌ **No spherical geometry / rotation.** A flat 50×50 grid; "latitude/longitude" are just axis labels used to phase the cycles. No Coriolis, no convergence of meridians, no latitude-dependent dynamics.
- ❌ **Single scalar field.** GraphCast couples dozens of variables (geopotential, T, u/v/w wind, humidity) across ~37 levels. SAVAR has one channel — no multivariate or vertical coupling, which is where most real atmospheric causality lives.
- ❌ **Teleconnections are instantaneous lagged scalar links**, not emergent wave dynamics. They're a reasonable _abstraction_ of teleconnection patterns (ENSO-like), but they don't arise from a transport mechanism.
- ❌ **Gaussian, near-white innovations.** Real climate noise is non-Gaussian, heavy-tailed, and spatiotemporally correlated.
- ❌ **Very short memory (τ_max=2 = 12 h).** Real atmospheric predictability and persistence span days.
- ❌ **Additive, hand-tuned forcing amplitudes.** The diurnal/annual amplitudes were calibrated by hand to hit target variance shares (annual ~33%, diurnal ~11%) rather than derived from physics; the annual band rides a 3.5× low-frequency AR resonance `1/(1−a)`.

---

## PART III — Audit: Comparability to GraphCast (structure & dynamics)

GraphCast = a ~37M-param **GNN** (encoder–processor–decoder on an icosahedral multi-mesh, 16 message-passing layers) trained on **ERA5** to autoregressively forecast the global atmosphere at 0.25° (~1M grid points × dozens of variables × 37 levels), 6 h steps, with multi-step rollout fine-tuning.

| Dimension | GraphCast | This project | Match? |
|---|---|---|---|
| Temporal cadence | 6 h | 6 h | ✅ strong |
| Diurnal + annual cycles | present (ERA5) | added explicitly | ✅ good |
| Noise regime | reanalysis-clean | low-noise (D_y=0.05) | ✅ fair |
| Forecast task | spatial field → next state | spatial field → next frame | ✅ conceptual |
| Governing dynamics | nonlinear PDE, chaotic | **linear VAR(2)** | ❌ major gap |
| Spatial transport/advection | yes (waves, fronts) | **none (fixed blobs)** | ❌ major gap |
| Geometry | rotating sphere | flat 50×50 | ❌ gap |
| Dimensionality | ~10⁶ pts × dozens of vars × 37 lvls | 2500 pts × 1 var | ❌ huge gap |
| Variable coupling | multivariate + vertical | single scalar | ❌ gap |
| Model architecture | GNN on multi-mesh | small CNN on regular grid | ⚠️ partial (both local message-passing-like) |
| Training regime | autoregressive rollout fine-tuning | **single-step** MSE | ⚠️ differs |
| Model scale | ~37M params | ~3.2M params | ⚠️ smaller |

### The central caveat (most important finding to flag)
The headline result — _"the GraphCast-like low-noise regime is inherently hard for mode-level interpretability"_ — is **over-stated**. **The activation collapse is a property of the linear-Gaussian data-generation process, not of GraphCast.** Because the generating process is linear and Gaussian, the optimal one-step forecaster is essentially linear, so a nonlinear CNN's optimal representation is nearly low-rank: activations are dominated by a single global-activity direction (PC0 ≈ 86–88%) and the per-mode SAE features come out polysemantic. None of this requires anything specific to a weather model — it is the signature of fitting a nonlinear network to data that a linear map already explains.

A real GraphCast is trained on genuinely nonlinear, multivariate, advective dynamics; there is no reason its internal representations must collapse the same way, and good evidence (rich learned weather features) that they don't. So:
- The **methodological** machinery validated here (PCMCI recovery; per-mode SAE alignment scoring; ensemble-mean deseasonalization removing both a causal and a representational confound) **does transfer** and is the real contribution.
- The **substantive interpretability conclusion** ("hard to find monosemantic mode features") is **conditional on the linear engine** and should _not_ be transferred to GraphCast without a nonlinear stress test.

---

## PART IV — Risks, Over-claims & Recommendations

### Over-claims to soften in any writeup
1. "GraphCast-like" should be qualified: the variant matches GraphCast's _sampling and seasonal surface structure_, not its _dynamics_.
2. The SAE "intrinsic limit" conclusion is really a **linear-dynamics** limit. State this explicitly.
3. "Accurately simulating climate dynamics" is not what SAVAR does — it simulates a _causal graph with climate-like cycles_. Recommend reframing as "climate-_styled_ causal testbed."

### Highest-value next steps to close the GraphCast gap (ordered)
1. ✅ **Add nonlinearity** to the latent dynamics — **IMPLEMENTED** in `data_gen/generate_nonlinear.py` (saturating autoregression `g(m)=(1−α)m+α·tanh(m)` + bounded bilinear/advective coupling along lag-1 cross edges; α,β knobs; α=β=0 recovers the linear diurnal variant). Verified bounded (max|Z|≈7, spectral radius of linear skeleton 0.78) and genuinely nonlinear: nonlinear features add only +0.010 R² to a linear mode predictor on the linear dataset vs **+0.066 R² on the nonlinear dataset**, so a linear forecaster is provably insufficient. **Still to run:** full 100-realisation generation → `split_nonlinear.py` → retrain CNN → re-extract activations → SAE, to test whether the activation collapse / polysemanticity survives.
2. **Add spatial transport** (advect mode blobs, or a propagating-wave forcing) so the CNN's spatial convolutions must learn real motion kernels — closer to GraphCast's job.
3. **Autoregressive multi-step training/eval** for the CNN to mirror GraphCast's rollout regime.
4. **Multivariate fields** (≥2 coupled channels) to introduce cross-variable causality, the dominant real-world case.
5. **Finish Phase 8** (causal centrality vs forecast importance) — it's the scientific payload and is independent of the realism gaps.
6. Consider a **spherical or latitude-weighted** grid if geometry effects matter for the centrality story.

### Smaller technical notes
- Diurnal at 4 samples/cycle is right at the resolvable limit — fine and realistic, but amplitude/phase recovery is sensitive; the FFT check in the generator is the right guard.
- Forcing amplitudes are hand-tuned to variance targets; document this as a calibration choice, not a derived quantity.
- ✅ Now under git (as of 2026-06-15; initial commit + pytest suite + diagnostics). Dataset/script lineage is recoverable via history; the spatial/position scripts above are still untracked working-tree additions.


on redundancy of features: for phase 6v
What was actually done: for each mode, pick the single best-aligned SAE feature (highest |corr(feature, Z_j)|), sign-flip it to match the latent's direction, and use that one feature per mode as the 8-variable time series for PCMCI. This sidesteps redundancy by construction — you're handpicking one representative per mode rather than feeding PCMCI the full SAE dictionary.

next steps:
- ✅ Phase 5's full per-feature forecast_importance[i] sweep — DONE 2026-06-16 (`sae/forecast_importance_sweep.py`). Per-feature ablation of each of the 509 alive spatial-SAE directions out of res3 on the diurnal CNN. Run on two splits via `--split`:
  - **raw full timeline** (`results/forecast_importance_sweep.npy`, 1168 windows, base RMSE 0.585): grid-locked (n=30)=−0.00000, content (n=6)=+0.00534, other (n=473)=+0.00004; top feat f380 (content) +4.35%; corr(imp, content_R2)=+0.426, corr(imp, position_R2)=+0.025. **NOTE: ~70% of these windows overlap the CNN's training timesteps** (the split is chronological 70/15/15; raw spans all of it), so these are partly in-sample.
  - **held-out test** (`results/forecast_importance_sweep_test.npy`, 2200 windows from 50 reals' final-15% segment, base RMSE 0.605 — genuinely out-of-sample, CNN trained only on steps 0:2043): grid-locked=−0.00000, content=+0.00058, other=−0.00003; top feat f8 (content) +0.41%; corr(imp, content_R2)=−0.141, corr(imp, position_R2)=−0.027.
  - **Verdict:** the QUALITATIVE taxonomy claim is out-of-sample robust — grid-locked features are exactly causally inert (ΔRMSE≈0) and the top forecast-important features are the content ones (f8, f380, f85) in BOTH splits; position decodability predicts forecast influence in neither. But the QUANTITATIVE content_R2↔importance correlation (+0.43) was in-sample inflation: out-of-sample the content-bucket importance shrinks ~9× and the dictionary-wide correlation collapses to ≈0. Do not claim content_R2 linearly predicts forecast importance; do claim grid-locked = epiphenomenal and content = the (small) seat of forecast influence. (Caveat: the SAE dictionary/taxonomy itself was extracted on the full timeline incl. test — affects feature definitions, not the CNN forecast, and the SAE is unsupervised wrt the target; the held-out numbers came out weaker not stronger.)
- ✅ Phase 7 (ground-truth feature↔latent mapping via per-mode SAEs) — DONE 2026-06-16 (`sae/eval_sae_per_mode.py`). Diurnal CNN: 8/8 aligned (|r|≥0.35), 5/8 strong (|r|≥0.5), 2/8 mode-specific/monosemantic (X1,X4); most best-features hit or exceed the linear ceiling. Default D_y=I CNN: 7/8 aligned, 0/8 strong, 0/8 monosemantic (X5 fails at r=0.284). → diurnal per-mode SAEs are substantially more aligned/monosemantic than D_y=I; polysemanticity (best feature also fires for other modes) remains the rule in both. Saved to `sae_data_diurnal/alignment_per_mode.npy` and `sae_data/alignment_per_mode.npy`.
- ✅ **Phase 8** (causal centrality vs forecast importance) — DONE 2026-06-16, see the Phase 8 line under "Pipeline phases & status" above. Per-mode input-space ablation on held-out test; out-driving centrality predicts forecast importance (out_degree r=+0.70), modulo an amplitude/variance confound (controlled via partial correlation). Next open: PDF Phase 9 (edge interventions) and Phase 10 (event subspaces).

Note for PCMCI for graphcast, should swap ParCorr for GPDC or CMIknn to detect more than linear (and roughly gaussian) dependencies
GPDC (Gaussian Process Distance Correlation) and CMIknn (conditional mutual information via k-nearest-neighbors) are drop-in replacements for ParCorr within tigramite — same PCMCI+ algorithm (same screening/orientation logic, same p_matrix output format), just a different independence test that can detect nonlinear dependencies.