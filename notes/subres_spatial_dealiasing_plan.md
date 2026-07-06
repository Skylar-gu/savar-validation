# Spatial sub-resolution / de-aliasing test battery — design

**Question.** Can the forecaster+SAE recover a spatial mechanism *finer than the
observed grid* — i.e. "read between the lines" in space? This is the spatial
analog of the fine-cadence *temporal* aliasing result
([[project_savar_subsample_pcmciplus]]): there, subsampling in time aliased τ=0
edges and PCMCI+ + non-Gaussianity de-aliased them. Here we subsample in space.

**The decisive design choice — the coarsening operator.**
- `D_sub` (subsample / decimate: keep every s-th fine pixel) → **aliasing**
  regime. Fine high-wavenumber structure folds into observed low wavenumbers:
  scrambled but *preserved* → de-alias-able IF a dynamical trace distinguishes
  the colliding scales. This is the regime where recovery is *possible*.
- `D_avg` (block-average s×s) → **low-pass/destructive** regime. High spatial
  frequencies are attenuated/destroyed → *not* recoverable; any "recovery" is
  the model's prior hallucinating plausible detail. This is the control that
  should FAIL.

The whole battery is only meaningful if the fine structure carries a distinct
**temporal signature** (distinct AR timescale φ, distinct causal parent, or
advection/dispersion coupling scale↔frequency). Identical dynamics = permanently
fused (see [[project_savar_gridlock]] — position decodable but epiphenomenal).

---

## Generative testbeds

### Testbed α — clean Fourier de-aliasing (identifiability floor, no GNN/SAE)
1-D fine grid length `Lf`; coarse length `Lc = Lf/s`. Pick P spatial wavenumbers,
paired so `k_low` and `k_high = k_low + Lc` **collide** under `D_sub` (same
observed wavenumber). Each wavenumber amplitude `a_k(t)` is AR(1) with a
*distinct* `φ_k` within a colliding pair, non-Gaussian (skew-normal) innovations.
Fine field `f(x,t) = Σ_k Re[a_k(t) e^{i k x}]`. Observe `y = D_sub f` or
`D_avg f`. **Recovery target:** the high-k amplitude series `a_high(t)`.
Fully analyzable → gives the theoretical ceiling and confirms the phenomenon
exists at our `T`, noise, and φ-spread before anything downstream is built.

### Testbed β — SAVAR-embedded (the SAE question)
Off `generate_hetdynamics.py`. Keep the 8 modes, but add fine sub-structure that
aliases under `D_sub`: within each blob place 2 fine sub-sources at distinct
fine-grid positions with distinct φ (slow/fast) and distinct causal parents.
Generate the fine pixel field, then observe through `D_sub` (primary) or `D_avg`
(control). Save fine ground truth `Z_fine(t)` (per sub-source amplitude series)
alongside the coarse observation `Y`. Train the existing forecaster on `Y`;
extract activations; train SAE. Score against `Z_fine`, not the observed modes.

---

## Test battery

**T0 — identifiability floor (Testbed α, oracle/linear only).**
Can *any* method de-alias `D_sub` given distinct φ? Report corr(â_high, a_high).
Gate: if the oracle fails here, STOP — nothing downstream can succeed.

**T1 — operator fork (α and β).** Same recovery under `D_sub` vs `D_avg`.
Prediction: recovery under `D_sub`, ≈chance under `D_avg`. This *is* the core
conceptual result (aliasing recoverable, low-pass not).

**T2 — recoverability ladder (β).** Recover each `Z_fine` component at:
| level | source | tests |
|---|---|---|
| (0) | coarse `Y`, single frame | resolution floor (should be low for parts, ≈1 for their sum) |
| (1) | coarse `Y`, K-frame window, linear | temporal context alone |
| (2) | linear spatiotemporal surrogate (VAR/DMD on `Y`, reuse `dmd_timescales.py`) | does a *linear* dynamical model already de-alias? |
| (3) | GNN activations | trained nonlinear model |
| (4) | SAE latents | **the SAE question** |
Metric = corr to the TRUE realization `Z_fine(t)`.

**T3 — true-realization vs statistics control (hallucination detector).**
For each recovered component report BOTH (a) corr to the true realization and
(b) match of spectrum/marginal statistics only. Real de-aliasing = (a) high.
Prior hallucination = (a)≈0 but (b) high. Decides which of the user's two
intuitions holds.

**T4 — identical-twin / null-dynamics control.** Include a sub-source pair with
identical φ + identical parent. Must be unrecoverable at every level and every
operator. If "recovered" → position/grid-lock leakage, not dynamics.

**T5 — dose–response on the dynamical gap.** Sweep Δφ between aliasing partners
(0 → large). Recovery R² should rise from ≈0 (T4 point) to high. Yields the
identifiability curve: how different must the dynamics be? Connects to the 22.8×
[[project_savar_mode_specialization]] spread.

**T6 — SAE-specific: preserve vs destroy, dedicated vs diffuse.**
(i) level (3) activations vs (4) SAE latents — does the bottleneck keep the
de-aliased info? (ii) within the SAE, is the fine component a DEDICATED latent
(Hungarian-MCC + uniqueness vs `Z_fine`, reuse `sae/eval_sae_metrics.py` matched
to fine GT) or diffuse? This is the literal "can SAEs encode sub-spatial
mechanisms" answer.

---

## Metrics
- Primary: corr(recovered, true `Z_fine`) per fine component, per level.
- Operator contrast: R²(`D_sub`) − R²(`D_avg`) (T1).
- Identifiability curve: R² vs Δφ (T5).
- Hungarian-MCC / uniqueness vs `Z_fine` GT (T6).
- Nulls: identical-twin recovery ≈ chance (T4); spectrum-only match with
  realization corr ≈ 0 flags hallucination (T3).

## Falsification (what kills the claim)
- T0 oracle fails → phenomenon absent at these params.
- `D_sub` ≈ `D_avg` (T1) → not de-aliasing, just prior.
- realization corr ≈ 0 while spectrum matches (T3) → hallucination, not reading
  between the lines.
- identical-twin recovers (T4) → spatial-position leakage.

## New files (distinct from running work; no collision)
- `data_gen/generate_subres.py` (Testbed β), Fourier gen inline in T0 script.
- `pcmci/subres_identifiability.py` (T0/T1 oracle + linear).
- `sae/subres_ladder.py` (T2/T6 ladder + SAE scoring), extend
  `eval_sae_metrics.py` with a `--ground-truth Z_fine` path.
- results → `results/subres_*.npy`.

## Sequencing
Run T0 first (cheap, no training) as a go/no-go gate. Only build Testbed β +
train the forecaster if T0 clears. Queue after the current 3-followup agent
finishes (it is touching `data_gen/`, `sae/`, `results/`) or run in an isolated
worktree.
