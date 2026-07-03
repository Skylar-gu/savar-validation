# Single-Session Agent Plan — literature-integrated experiments (2026-07-03)

One autonomous session on the local box (data + checkpoints + L40S GPU present).
Work through the blocks **in order** — they are sorted by value-per-hour. The user
picks up results tomorrow; an unfinished tail of blocks is fine, a half-done block
with no logged state is not.

Background: `notes/results_gnn.md` (current state), `notes/next_steps_plan.md`
(diagnosis + updated recommendations), `notes/literature_review.md` (sources cited
below per block).

## Protocol

- Branch: `agent/session-2026-07-03` off the current branch (`phase5-7-8-analysis`)
  so all uncommitted work-in-progress files remain visible.
- Progress log: append to `notes/agent_progress.md` when a block starts, finishes,
  or blocks (timestamp, status, key numbers, blocker if any).
- Results: numeric artifacts to `results/` as `.npy` (dict pattern used elsewhere in
  repo), narrative + tables to `notes/session_results.md` (one section per block,
  numbers first, interpretation second).
- Commit after each completed block with a message naming the block. **Local commits
  only — never push.**
- If a block errors twice on the same obstacle, log the blocker with the exact error
  and move to the next block. Never silently skip.
- Run everything from repo root with `python3` (the main environment used for
  train/, sae/, pcmci/ scripts). Exception: Block A uses `param-decomp/.venv`
  (py3.13, torch 2.8 cu128) — see `notes/vpd_repo_guide.md` and
  `notes/vpd_on_savar_gnn.md`.
- Read the existing scripts for dataset paths and conventions before writing new
  code (e.g. `sae/extract_activations_gnn.py`, `pcmci/run_pcmci_subsample.py`,
  `data_gen/generate_hetdynamics.py` for where W, W_plus, Φ ground truth live).
- `pip install` into the main env is allowed for: `lingam`, `pydmd`,
  `python-louvain` or `leidenalg`+`igraph`. Nothing else without logging why.

## Guard rails (standing, do not violate)

- **Never add learned per-position parameters** to any model (node embeddings,
  per-pixel biases) — they are a confound for the grid-lock question. Static
  coordinate *inputs* are allowed.
- Do not modify or overwrite frozen checkpoints or existing datasets; do not
  regenerate data (use `realisations_finecadence`, `realisations_hetdynamics`,
  `realisations_hetdynamics_eqvar` as they are).
- Match repo style: flat scripts with env-var config at top, results as dict `.npy`.

---

## Block A — VPD on the hetdynamics_eqvar checkpoint (THE decisive experiment)

*Why (lit):* our diagnosis says VPD components collapsed to one mechanism because
the finecadence GNN computes one mechanism; SPD/APD papers say minimality only
differentiates when separable mechanisms exist. hetdynamics_eqvar has a 22.8×
per-mode timescale spread → separable mechanisms should now exist.

1. Re-run VPD with the M4 configuration (all four MP layers `layers.{0..3}.mlp.{0,2}`,
   C=64, layerwise `vector_mlp` CI gates, full losses, 5000 steps) but with
   `checkpoints_hetdynamics_eqvar/best.pt` and the eqvar dataset. Reuse the exact
   invocation recorded in `notes/vpd_results.md` / `notes/vpd_on_savar_gnn.md`;
   previous run dir: `vpd_out/runs/p-0625f7dd` (copy its config as the template).
2. Recompute the redundancy diagnostics (same code as the 2026-07-03 analysis):
   gate-map participation ratio per layer, median pairwise gate-map correlation,
   blob-usage CV. **New:** per-component blob-usage profile (C×8) — test whether
   components now segregate by mode, and whether component→mode assignment orders
   by timescale φ (Spearman of dominant-mode φ vs component index clustering).
3. Score against `notes/results_gnn.md` §5 baselines (PR≈1.1/64, blob CV≈0.04,
   median corr 0.79).

**Success:** PR meaningfully above ~3/64 AND blob-usage CV well above 0.04 with
mode-preferential components → data-problem verdict CONFIRMED. **Failure** (still
PR≈1) → method/config problem; then run the secondary experiment below before
concluding.

*Secondary (run only if A finishes early or fails):* same checkpoint, C=16 and
ImportanceMinimality coefficient ×30 (the loss-balance amplifier identified in
results_gnn §5) — tests whether config, not data, was suppressing differentiation.

## Block B — Impulse-response dynamical test of the frozen GNN (Hakim–Masanam arm)

*Why (lit):* dynamical-testing line (Hakim & Masanam): probe the trained emulator
with structured perturbations, no internals needed. This is the internals-free
control arm for every interp claim we make.

1. Load frozen `checkpoints_hetdynamics_eqvar/best.pt`. From held-out eqvar data,
   take K=3-frame input windows; add an impulse δ·pattern_j to the last frame where
   pattern_j is mode j's spatial footprint (column of W_plus / row of W — check the
   generator for which maps latent→pixel), δ scaled to ~1σ of pixel std.
2. Autoregressive rollout 12 steps for perturbed vs unperturbed; response
   ΔX_t. Project ΔX_t onto every mode's pattern → latent response matrix
   R[i, j, τ] (response of mode i at lag τ to impulse in mode j), averaged over ≥200
   windows.
3. From R, build a directed graph (edge j→i if max_τ |R[i,j,τ]| exceeds a
   permutation-null threshold, excluding i=j) and score **F1 vs the ground-truth Φ
   edge set** (same scoring as `pcmci/run_pcmci_modes.py`). Also compare each mode's
   self-response e-folding time against its designed φ.

**Deliverables:** `results/impulse_response.npy`, F1 + per-mode timescale table.
**Success:** F1 in the ballpark of PCMCI-on-Z (~0.8) → the GNN internalized the
causal graph and we have a cheap graph-recovery channel that never opens the model.
CPU-feasible if GPU is busy with Block A — run this concurrently only if memory
allows; otherwise sequentially.

## Block C — SAE metric-suite upgrade (SynthSAEBench)

*Why (lit):* SynthSAEBench (arXiv 2602.14687) + "Are SAE benchmarks reliable?"
(2605.18229): best-|r| scalars overstate; use Hungarian-matched global metrics,
report seed variance, never select by recon MSE.

1. New script `sae/eval_sae_metrics.py`: given SAE codes + true Z for a dataset,
   compute (a) **Hungarian-matched MCC** — `scipy.optimize.linear_sum_assignment`
   on the feature×mode |Pearson| matrix, report mean matched |r|; (b) **per-mode
   matched F1** — binarize feature activity (in-TopK) and mode activity
   (|Z| above per-mode median), F1 of the matched pairs; (c) **feature uniqueness**
   — for each matched feature, margin between its matched-mode |r| and its best
   other-mode |r| (this generalizes our specificity).
2. Run on all existing SAE artifacts: `sae_data_gnn/` (finecadence),
   `sae_data_hetdynamics/`, `sae_data_hetdynamics_eqvar/` — per-mode and mixed.
3. Retrain the **mixed eqvar SAE with 3 seeds** (cheap; reuse
   `sae/train_sae_mixed.py`) → error bars on every eqvar metric.

**Deliverables:** `results/sae_metrics_suite.npy`, table in session_results with the
old best-|r| numbers side-by-side. This re-scores the whole SAE story with
literature-standard metrics.

## Block C2 — KAN-SAE bake-off (runs right after C; gated downstream propagation)

*Why (lit):* KAN-SAE (arXiv 2605.17493) puts per-feature learnable 1-D spline gates
in the encoder — the designated backup architecture because our saturating-AR
dynamics plausibly make the activations→Z map nonlinear, which a linear-encoder
TopK SAE can only capture linearized. Block C's frac-of-ceiling numbers say whether
there is headroom for this to matter; the bake-off measures it directly.

1. Implement as a **flag on the existing trainer** (`sae/train_sae_mixed.py`, e.g.
   `SAE_ARCH=kan`), not a parallel script. Spline (or small per-feature monotone
   MLP) gates go on the **encoder pre-activations only**; keep the TopK sparsity
   mechanism and the **linear decoder unchanged** — decoder linearity is
   load-bearing (Hungarian matching against Z directions in Block C, steering
   directions in Block F).
2. Train on the identical mixed eqvar activations with the identical 3 seeds as
   Block C's retrains. CPU-feasible if the GPU is still busy with Block A (pooled
   activations are small).
3. Score with the Block-C metric suite, paired against TopK seed-for-seed:
   Hungarian-matched MCC, per-mode matched F1, feature uniqueness. **Never select
   on reconstruction MSE.**
4. **Gate:** only if KAN-SAE wins on the matched metrics beyond the seed error bars
   do its codes propagate downstream — then rerun Block D (does identity move out
   of the dilution regime?) and Block F (spline gates give native per-feature
   dose–response curves) with KAN codes. If it ties or loses, TopK stays primary
   (keeps comparability with the MacMillan & Ouellette GraphCast SAEs). If
   wall-clock is short, run the bake-off but log the D/F reruns as follow-ups
   instead of executing them.

**Deliverables:** `results/kan_sae_bakeoff.npy`, paired table in session_results.
Roadmap note: whichever SAE wins this bake-off is the extractor that deploys on
GraphCast activations — that decision is the point of doing it on synthetic data.

## Block D — Dilution diagnosis via inverse-Ising + community structure

*Why (lit):* Goodfire concept-manifolds (2604.28119): concepts land in capture /
tiling / dilution regimes; diagnose with inverse-Ising couplings on feature
co-activation + Leiden communities. Prediction: mode *identity* is in the dilution
regime; slow-mode *content* approaches capture.

1. Binarize mixed eqvar SAE codes (active = selected by TopK). Fit pairwise Ising
   couplings by per-feature logistic regression (pseudo-likelihood; sklearn is fine;
   restrict to features active in >0.5% of samples).
2. Cluster the |J| coupling graph (Leiden via `leidenalg`, else Louvain). Map each
   community to modes by mean |corr| of member features with each Z_j.
3. Classify each mode: **capture** (one feature dominates), **tiling** (one
   community of co-active features, high within-community J), **dilution** (spread
   across communities, weak couplings). Repeat for the identity signal: same
   analysis on per-mode demeaned codes vs raw.

**Deliverables:** `results/ising_regimes.npy`, per-mode regime table.

## Block E — τ=0 orientation benchmark (Gong et al. ICML 2015)

*Why (lit):* Gong 2015 proves subsampled non-Gaussian linear systems are
identifiable — our skew-normal innovations make the aliased τ=0 edges orientable in
principle. Benchmark existing estimators; do not invent heuristics.

1. `pip install lingam`. On the subsampled fine-cadence Z (strides s=2,3,4, same
   setup as `pcmci/run_pcmci_subsample.py`), run **VAR-LiNGAM**.
2. Score: (a) orientation accuracy on exactly the aliased edge set (fine lags ℓ<s)
   that PCMCI+ left `o-o`; (b) full-graph F1 for comparison with the existing sweep
   (`results/pcmci_subsample_sweep.npy`: CON-F1 0.63/0.49/0.38).
3. If VAR-LiNGAM underperforms, try DirectLiNGAM on PCMCI+ residuals
   (hybrid: PCMCI+ for skeleton, LiNGAM for orientation) — one attempt only.

**Deliverables:** extend the subsample results file or add
`results/orientation_benchmark.npy`; orientation-rate + accuracy table vs the
PCMCI+ 0.08–0.30 orient rate.

## Block F — Steering dose–response (causal validation of SAE features)

*Why (lit):* CFD-GNN SAE paper (2604.04946) shows frozen-GNN SAE features work as
causal steering handles but static offsets fail on oscillatory latents — test both.

1. For each slow mode X5–X7 (eqvar), take its best Z-aligned mixed-SAE feature.
   Steer: H ← H + α·d_f (decoder direction) at the last MP layer for all nodes,
   α ∈ {−3σ_f, −1σ_f, +1σ_f, +3σ_f} where σ_f is the feature's activation std.
2. Measure Δprediction projected onto the steered mode's spatial pattern
   (dose–response slope + linearity R²) and onto the other 7 modes (specificity).
3. Time-aware variant: modulate α by the feature's own current activation sign
   (the paper's fix for oscillatory failure); compare.

**Deliverables:** `results/steering_dose_response.npy`. **Success:** monotone
on-target response with off-target leakage ≪ on-target → SAE features are causal
handles, closing the loop the MacMillan & Ouellette GraphCast paper left open.

## Block G — Aggregation-consistency check on W-pooling (Adag)

*Why (lit):* Adag (2505.10476): pixel→mode aggregation can create/destroy
conditional independencies unless consistency conditions hold. Our W-pooling is
exactly such an aggregation; this gates the future overlapping-modes rung.

1. Numerically check the overlap structure of W (row support overlap matrix — the
   current blobs are near-disjoint, quantify how near).
2. Empirically: PCMCI+ on (a) true Z, (b) W-pooled pixel series, (c) W-pooled GNN
   *activation* series (the pipeline's actual object) — same realisations, same
   settings. Report edge-set agreement (F1 of b and c against a's graph, and
   against ground truth Φ).

**Deliverables:** `results/aggregation_consistency.npy`. A drop from (a)→(c) that
is NOT present (a)→(b) localizes distortion to the representation, not the pooling.

## Block H — Koopman/DMD timescale cross-check (cheapest, do last)

*Why (lit):* SINDy-SHRED/Koopman line: spectra of a linear surrogate give an
architecture-free readout of system timescales.

1. `pip install pydmd`. DMD (rank ~20) on eqvar pixel data per realisation;
   eigenvalue e-folding times vs the designed φ spectrum (22.8× spread);
   check whether leading DMD modes' spatial supports match W blob footprints
   (cosine vs each pattern_j).

**Deliverables:** `results/dmd_timescales.npy`, one table.

## End-of-session

Write `notes/session_results.md` closing section: per-block status table
(done/partial/blocked + headline number), the Block-A verdict stated plainly
(data problem confirmed or not), and the 3 highest-value follow-ups for tomorrow.
Final commit.
