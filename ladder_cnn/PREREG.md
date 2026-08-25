# PREREG — SAE -> PCMCI+ on SAVAR with known ground truth: an oracle-ablation ladder

**Status: FROZEN 2026-08-21, before any new number was computed.**
Written after reading only *already-published* repo numbers (`results/pcmci_results.npy`,
`results/pcmci_features.npy`, `results/litext_e1_discovery.npy`) and the source of
`pcmci/run_pcmci_features.py`, `pcmci/run_pcmci.py`, `sae/discover_modes.py`.
No new analysis had been run at the time of writing.

## 0. Question

Does SAE -> PCMCI+ recover a causal graph when the ground truth is KNOWN and **no oracle
information** is used — i.e. in the configuration that structurally matches what the
GraphCast SAE-graph work actually does?

## 1. Fixed scoring protocol (identical for every rung, every null, every control)

- Dataset: `data/realisations` (base), `sae_data/base`. 100 realisations, T_eff = 497.
- Ground truth: `ground_truth_graph` (8,8,2), index convention `G[eff, cause, tau-1]`.
  Cross-mode edges only. **Gate:** must equal the 12 edges stored in
  `results/pcmci_results.npy['ground_truth']`.
- Causal discovery: tigramite `PCMCI.run_pcmci(tau_min=1, tau_max=2, pc_alpha=0.2,
  alpha_level=0.05)`, ParCorr. Detection = `p_matrix[cause, eff, tau] < 0.05`, cross-only.
  (Byte-identical to `run_pcmci_features.py`, so R0 is directly comparable to the
  published 0.695.)
- Metric: per-realisation P/R/F1, then the **mean over realisations**. Reported with sd.
- Edge mapping when the variable set is not already the 8 true modes:
  **Hungarian-strict**, exactly as `sae/discover_modes.py::score_candidate`:
  a detected edge `(c,e,tau)` on discovered variables is mapped to `(m(c), m(e), tau)`
  if both endpoints are matched; **edges touching an unmatched variable count as FP**;
  **gt edges at unmatched true modes count as FN**. No more permissive scorer is allowed.

## 2. Variable->mode matching rules (declared in advance)

- **MAP-ID**: identity. Used when the variable set already carries the true mode index
  (rungs where the oracle W-pooling still supplies the stream label).
- **MAP-FOOT**: Hungarian on cosine between the discovered footprint and the true `W`,
  accepted only at cosine >= 0.30 (the `MATCH_COS_MIN` of `discover_modes.py`). Used for
  pixel-pooled controls and for R4.
- **MAP-R**: Hungarian on |Pearson r| between the discovered variable's series and the
  true latent `Z_j`, accepted only at |r| >= 0.10. This is an **evaluation-time oracle**
  and is deliberately generous. It is legitimate ONLY because every null and every
  control below is granted the identical freedom. A positive result under MAP-R that the
  null also reaches is a NEGATIVE result.

## 3. The ladder

| rung | oracle removed | variable set | matching |
|---|---|---|---|
| **R0** | none (replication of `run_pcmci_features.py`) | per-mode SAE_j, feature = `alignment_per_mode[j].best_feat` (max &#124;r&#124; vs Z_j), sign-flipped to +align Z_j | MAP-ID |
| **R1** | oracle **sign** | as R0, sign fixed by the unsupervised rule `sign = sign(skew(series))`, no Z | MAP-ID |
| **R2** | oracle **feature selection** | per-mode SAE_j, feature = **argmax variance** of the encoded series over all (r,t). No Z anywhere. | MAP-ID |
| **R3a** | oracle **per-mode dictionary** | one **mixed** SAE (`sae_best.pt`), candidate = (stream j, feature f); one feature per stream by argmax variance | MAP-ID |
| **R3b** | oracle **N and the mode partition** — **THE GRAPHCAST-MATCHED RUNG** | one mixed SAE; series_f[r,t] = mean over the 8 streams of a[r,j,t,f] (the analogue of pooling a GraphCast SAE feature over mesh nodes); variables selected by SEL-VAR below; N not given | MAP-R |
| **R4** | oracle **pooling** | spatial SAE trained on per-pixel res3 vectors; each feature's mean activation map is its footprint; pool pixels through the footprint to get the series; variables selected by SEL-VAR | MAP-FOOT (primary) + MAP-R (secondary) |

### SEL-VAR — the declared unsupervised selection rule (frozen)

1. Drop candidates whose series is constant or has nonzero-fraction < 0.02 in every
   realisation (dead features).
2. Rank surviving candidates by variance of the pooled series, averaged over realisations
   (variance computed on the z-scored-per-realisation series is NOT used; raw variance is).
3. Greedily accept candidates in rank order, rejecting any whose |Pearson r| with an
   already-accepted candidate exceeds **0.90** (redundancy prune).
4. Stop at N_hat = 12 accepted, or when the candidate pool is exhausted.
   N_hat is reported, not assumed. (12 = the `C0` used by `discover_modes.py`.)

**Sensitivity (reported, not headline):** the same ladder with the ranking in step 2
replaced by (a) activation frequency (fraction nonzero) and (b) |loading on PC1 of the
candidate x candidate covariance|.

## 4. Guardrail #9 — the bar, calibrated on BOTH sides

The bar for "a causal graph was recovered" at R3b is **NOT an F1 value**. It is:

> **Observed F1 must exceed the 95th percentile of the null F1 distribution built with
> the same selection-and-matching freedom (p < 0.05, one-sided).**

Three obligations, all mandatory:

- **(i) the null VARIES.** Report the full null F1 distribution (mean, sd, min, max,
  5/50/95th percentiles). If sd == 0 the null is a point mass and the rung is VOID
  (the PX_geo failure mode).
- **(ii) the bar is ATTAINABLE under the null.** Report where each rung sits in the null
  as an explicit p-value. If the null's 95th percentile already exceeds the observed
  ceiling, the bar is vacuous and must be reported as such.
- **(iii) a negative control FAILS.** `shift5` and `diag8` misplaced footprints, pooled
  from pixels through the corrupted W, scored with the identical protocol. They must
  score at or near 0 under MAP-FOOT. If they do not, the scoring protocol is broken and
  the whole ladder is reported as an instrument failure.

### Nulls (all with the identical selection + Hungarian matching freedom)

- **N-RAND**: N_hat variables drawn uniformly at random from the live candidate pool
  (bypassing SEL-VAR's ranking, keeping its dedup), then discovery + matching as normal.
- **N-PHASE**: the observed selected series, each independently phase-randomised
  (preserves autocorrelation + marginal spectrum, destroys cross-dependence).
- **N-SHIFT**: the observed selected series, each independently circularly shifted by a
  random lag in [50, T-50].

Draws: >= 200 per null where affordable; the realisation count per draw (R_null) is set by
a measured timing probe and reported. The observed value it is compared against is
recomputed at the SAME R_null.

## 5. Pre-declared outcome semantics

- If R0 does not reproduce 0.695 +/- 0.02, **STOP** and report an instrument failure.
  Nothing below R0 is interpretable.
- R3b is declared **POSITIVE** only if observed F1 > null 95th percentile with p < 0.05
  under N-RAND *and* N-PHASE, *and* the negative control fails.
- Otherwise R3b is declared **NEGATIVE**, stated plainly, and the F1 number is reported
  as uninformative regardless of its magnitude.
- Any rung that cannot be run for measured cost reasons is reported with the measured
  timing that killed it, and named explicitly as omitted. The ladder is never silently
  narrowed.

## 6. Also pre-declared (secondary, no bars attached)

- **Sparsity / n_eff**: per-feature zero fraction of the series entering PCMCI; per-pair
  co-firing count (timesteps where both are nonzero) vs the nominal T=497 the ParCorr
  p-values assume.
- **Deseasonalisation contrast**: `sae_data/diurnal` vs `sae_data/diurnal_deseason`,
  R0 / R2 / R3b re-derived on both. Question: does deseasonalisation rescue the
  unsupervised rungs or only the oracle ones? No bar; descriptive.
- **R3b false-positive structure**: logistic/point-biserial association between edge
  presence and (a) shared PC0 loading of the two features, (b) |corr| of the two series.

## 7. Known anchors, and one correction made before freezing

- `results/pcmci_results.npy` (PCMCI+ on true Z, base dataset) is **F1 0.825**
  (P 0.715, R 0.984), *not* 0.853. 0.853 is the `hetdynamics_eqvar` anchor recorded in
  `discover_modes.py`. **The true-Z ceiling for this ladder is 0.825.**
- `results/litext_e1_discovery.npy` (incl. shift5/diag8 = 0.000) was produced on
  `data/realisations_hetdynamics_eqvar` with tau_max=6 and `run_pcmciplus(tau_min=0)`.
  Those numbers are NOT directly comparable to this ladder. The shift5/diag8 control is
  therefore **re-run here** on the base dataset under this ladder's exact protocol.
- `sae_data/base/alignment.npy` (mixed SAE) has `mode_best_feat` = [383]*8: every one of
  the 8 modes' best mixed-SAE feature is the SAME feature 383. This is the recorded
  "mixed SAE collapses onto global activity" finding. It is a strong prior that R3 will
  be negative. It is NOT a result — R3 is run to measure it against a calibrated null.

## 8. Compute constraints

CPU only. `OMP_NUM_THREADS=2`, process pools <= 4 workers. `~/savar-project` is READ-ONLY;
every artefact written under this scratchpad. One PCMCI unit and one CNN-forward unit are
timed before any batch is launched.
