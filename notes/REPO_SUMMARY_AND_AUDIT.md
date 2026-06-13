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
- **Phase 8 — causal centrality vs forecast importance** ❌ not started.
- **Stress tests 1b–1f** ❌ not started.

### Diurnal-pipeline results (the GraphCast comparison, 2026-06-13)
- **CNN:** val RMSE 0.596, corr 0.64 (lower-variance data; not comparable to 1.07).
- **PCMCI:** raw F1=0.293 (FP 58 — cycle is a shared confounder) → **deseasonalized** (Z − ensemble mean) **F1=0.825, exactly restoring baseline**. Cycle is a clean, removable confounder.
- **SAE raw:** 8/8 aligned, 5/8 strong, only 2/8 monosemantic; PC0 var 86→88.7%; R²(PC0~cycle)=0.32 — the "strong" features mostly track the **shared clock**, inflating alignment while hurting specificity.
- **SAE deseasonalized:** R²(PC0~cycle)→0.00 (cycle removed) but alignment **collapses** to 2/8 aligned, 0/8 strong. **Punchline: the raw "strong" features were entirely cycle-tracking artifacts**; the residual low-noise dynamics carry even less mode signal. The CNN's polysemantic global-activity encoding is intrinsic, not caused by the cycle.

### Repo layout (reorganised 2026-06-13; **not a git repo** — moves reversible by `mv`)
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
- Not a git repo — there is **no version history**; the only provenance is these notes + memory files. Recommend `git init` before further changes so the dataset/script lineage is recoverable.
