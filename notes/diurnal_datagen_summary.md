# Diurnal/Annual Data-Gen Summary (GraphCast comparison)

monosemantic: the best of the 512 SAE features tracks this mode (higher r value) more than any other with margin 0.05 

aligned: for the model, the top SAE feature reaches |r| > 0.35 with the mode's state; doesnt require exclusivity
- r is the Pearson correlation coefficient

## Changes to the data-gen functions

- **New generator `generate_diurnal.py`** (+ `split_diurnal.py`): forks the `generate_dy005.py` kernel but stamps a physical cadence of **dt = 6 h** (GraphCast), runs **T = 2920 steps = 2 years**, and keeps **D_y = 0.05·I_L** (low-noise, reanalysis-like). Ground-truth graph G and weights W are **unchanged**, so Phase 6/7 stay directly comparable.
- **Deterministic atmospheric cycles, injected in latent space *before* the VAR recurrence** (so they propagate through Φ): a **diurnal** cycle (24 h = 4 steps, phased by longitude), an **annual** cycle (1 yr = 1461 steps, phased by latitude/hemisphere), and **afternoon heteroskedasticity** (per-mode innovation variance peaks ~3 h past local noon). The 12 h semidiurnal tide is **dropped** — it sits exactly at the 6 h Nyquist frequency and aliases away (ERA5/GraphCast can't resolve it either).
- **Climatology is fixed across realisations** (same Earth, different weather) like W and G; only the stochastic noise varies per seed. Each `.npz` saves `forcing_latent`, `diurnal_amp/phase`, `annual_amp/phase`, and `cycle_meta = [dt, P_D, P_A, het]` so the cycles are exactly recoverable for deseasonalization.

## How the 8 modes tie to the grid (longitude & latitude)

- The grid has no real lat/lon, so **longitude ≡ x-axis** (full 50-col width = one 24 h solar sweep) and **latitude ≡ y-axis**. Diurnal phase depends only on longitude: `φ_d = −2π·x_center/nx` (east leads west); annual phase flips by hemisphere: `φ_a = π if y_center > ny/2 else 0`.
- The 8 modes occupy a **3×3 blob grid** (9th slot empty): 3 longitude bands (x = 8, 24, 40) × 3 latitude bands (y = 8, 24, 40). This yields **3 diurnal phase groups** and a **clean N/S split** for the annual phase.
- Mapping of mode → grid position → cycle phase:

| Mode | x (lon) | y (lat) | band | φ_d (diurnal) | φ_a (annual) |
|------|--------:|--------:|------|--------------:|-------------:|
| X0 | 8  | 8  | W / North | −1.01 | 0 |
| X1 | 24 | 8  | C / North | −3.02 | 0 |
| X2 | 40 | 8  | E / North | −5.03 | 0 |
| X3 | 8  | 24 | W / Mid   | −1.01 | 0 |
| X4 | 24 | 24 | C / Mid   | −3.02 | 0 |
| X5 | 40 | 24 | E / Mid   | −5.03 | 0 |
| X6 | 8  | 40 | W / South | −1.01 | π |
| X7 | 24 | 40 | C / South | −3.02 | π |

## Recent results (rebalanced, realisation 000)

- **First pass was annual-swamped** (annual ~88% of variance, diurnal ~1%): the annual forcing sits at ω≈0 and is amplified ~3.5× by the low-frequency AR gain `1/(1−a)`, while the diurnal cycle (ω=π/2) is slightly attenuated. Fixed by cutting `A_a` (→ U(0.35,0.70)) and raising `A_d` (→ U(0.75,1.40)).
- **Rebalanced shares:** dynamics dominate (causal signal recoverable by PCMCI on raw data), annual is a real seasonal signal, diurnal is a clear ~11% feature — realistic ordering annual > diurnal preserved.
- **Longitude-phasing confirmed:** after removing the cycles, residual variance peaks at a *different* diurnal slot per mode (X0→slot 2, X2→slot 0, X4→slot 3), i.e. each mode's local afternoon falls at a different UTC step.

| Mode | Diurnal % | Annual % | Dynamics+noise % | Z std |
|------|----------:|---------:|-----------------:|------:|
| X0 |  9.4 | 44.6 | 46.0 | 2.13 |
| X1 |  8.7 | 54.9 | 36.4 | 2.63 |
| X2 | 19.1 | 43.5 | 37.4 | 2.56 |
| X3 |  3.4 | 20.4 | 76.2 | 1.95 |
| X4 |  5.8 | 50.6 | 43.6 | 2.17 |
| X5 | 16.6 | 16.7 | 66.7 | 1.76 |
| X6 | 14.8 |  1.7 | 83.5 | 1.79 |
| X7 | 14.0 | 31.1 | 54.9 | 1.89 |
| **Mean** | **11.5** | **32.9** | **55.6** | — |

| Dataset | Cadence | T | D_y | Files | Split (train/val/test) | CNN k=3 windows |
|---------|---------|---|-----|-------|------------------------|-----------------|
| realisations_diurnal | 6 h | 2920 (2 yr) | 0.05·I_L | 100 (2.6 G) | 2043 / 438 / 439 | 204k / 43.5k / 43.6k |

---

# Pipeline results (CNN → PCMCI → SAE), 2026-06-13

Run unattended via `run_diurnal_pipeline.sh` (tmux); logs in `logs_diurnal/`.
Ground-truth graph G unchanged vs baseline, so all metrics are comparable.

## CNN forecaster

- Retrained on `splits_diurnal/` (`train/cnn_forecaster.py --diurnal` → `checkpoints_diurnal/best.pt`).
- **Val RMSE 0.596, val corr 0.64.** Not comparable to the 1.07 baseline — the diurnal data has lower total variance (obs std ≈ 0.79 vs 1.45).

## PCMCI — the cycle is a clean, removable confounder

Deseasonalization = subtract the ensemble mean over realisations (which, because
the dynamics are linear and the forcing is shared, equals the exact forced cycle).

| Run | F1 | Mean FP | Note |
|-----|----|---------|------|
| **Raw** (cycles present) | 0.293 | 58.1 | cycle confounds all modes → FP explosion |
| **Deseasonalized** (Z − ensemble mean) | **0.825** | 5.2 | **exactly restores baseline** |
| baseline (no cycles) | 0.825 | — | reference |

Top raw false positives are same-cycle-phase mode pairs (X0→X4, X4↔X6, X2→X6).

## SAE per-mode (raw diurnal activations)

| Metric | Diurnal | Baseline (D_y=1) | Meaning |
|--------|---------|------------------|---------|
| Aligned (\|r\|≥0.35) | 8/8 | 7/8 | a feature tracks the mode at all |
| Strong (\|r\|≥0.5) | 5/8 | 0/8 | strong tracking (mostly **cycle-tracking**) |
| Monosemantic | 2/8 | 0/8 | mode-specific (own > all others); fragile |
| Mean PC0 var% | 88.7% | 86% | representation collapse (worse) |
| Mean R²(PC0~cycle) | 0.32 | — | ~⅓ of PC0 *is* the deterministic cycle |

Per-mode best feature (max\|r\| / specificity): X0 .590/−.095, X1 .647/+.060,
X2 .594/−.067, X3 .463/−.152, X4 .590/+.089, X5 .385/−.276, X6 .438/−.197,
X7 .508/−.162. Six of eight have **negative** specificity (best feature correlates
more with another mode's Z than its own) → the shared cycle drives polysemanticity.

**Mechanism:** the shared diurnal+annual cycle is absorbed into PC0 (R²=0.32) →
PC0 grows more dominant (86→88.7%) → SAE features latch onto PC0 and track the
*shared clock*, inflating raw alignment (5/8 strong) while hurting genuine
mode-specificity (2/8 monosemantic). This is the internal-representation mirror
of the PCMCI confounding.

## Deseasonalized SAE (activation-space fix)

`sae/deseason_activations.py --diurnal` subtracts the ensemble-mean activation
climatology → `sae_data_diurnal_deseason/`; SAE chain re-run with `--deseason`.

Climatology (cycle) variance removed per mode: // fraction removed = 1 − var(anomaly) / var(original)

| Mode | X0 | X1 | X2 | X3 | X4 | X5 | X6 | X7 |
|------|----|----|----|----|----|----|----|----|
| act var removed | 56.3% | 65.0% | 62.8% | 38.2% | 57.6% | 39.0% | 28.1% | 47.6% |
| Z var removed | 54.6% | 64.6% | 64.1% | 26.2% | 56.2% | 33.8% | 17.2% | 44.4% |

SAE on deseasonalized activations (cycle removed):

| Metric | Raw diurnal | **Deseasonalized** | Baseline (D_y=1) |
|--------|-------------|--------------------|------------------|
| Aligned (\|r\|≥0.35) | 8/8 | **2/8** | 7/8 |
| Strong (\|r\|≥0.5) | 5/8 | **0/8** | 0/8 |
| Monosemantic | 2/8 | **1/8** | 0/8 |
| Mean PC0 var% | 88.7% | **87.2%** | 86% |
| Mean R²(PC0~cycle) | 0.32 | **≈0.00** | — |

Per-mode max\|r\| collapses to 0.24–0.44 (only X3 .39 and X6 .44 clear 0.35);
specificity still negative for 7/8 (X6 the lone +0.05).

**Interpretation — the deseason result is the punchline:**
1. Deseasonalization works as designed: **R²(PC0~cycle) → 0.00** confirms the
   cycle is fully removed from the dominant activation direction, and PC0 var%
   eases 88.7→87.2%.
2. But mode-specific dynamics features do **not** re-emerge — alignment
   *collapses* (5/8 strong → **0/8**, 8/8 aligned → 2/8). So the raw-diurnal
   "strong" features were **entirely cycle-tracking artifacts**, not dynamics.
3. The residual low-noise dynamics carry *less* recoverable mode signal than the
   D_y=1 baseline, PC0 stays ~87%, and specificity stays negative (7/8).
   **Conclusion:** the CNN's polysemantic global-activity encoding is a property
   of the **linear-Gaussian data-generation process**, not of GraphCast and not
   caused by the cycle. Because the generating process is linear+Gaussian, the
   optimal one-step forecaster is essentially linear, so a nonlinear CNN's
   representation is nearly low-rank (PC0≈87%) regardless of the cycle — the
   cycle was just a second shared confound layered on top. This says nothing
   specific about a weather model; a real GraphCast is trained on nonlinear,
   advective, multivariate dynamics and need not collapse this way.
   **Follow-up (in progress):** `data_gen/generate_nonlinear.py` adds bounded
   nonlinear latent dynamics (saturating AR + bilinear advective coupling) to
   test directly whether the collapse survives once the data is no longer
   linear-Gaussian. Verified material: nonlinear features add +0.066 R² to a
   linear mode predictor on the nonlinear data vs only +0.010 on the linear
   diurnal data. See `notes/REPO_SUMMARY_AND_AUDIT.md`.

Both confounds — causal-graph (PCMCI) and representational (SAE/PC0) — are cleanly
removed by the *same* ensemble-mean deseasonalization; the difference is that
PCMCI recovery is fully restored (F1 0.825) whereas SAE interpretability is
*unmasked* as intrinsically limited rather than improved.

### What the raw "strong" features actually tracked (`sae/decompose_feature_cycle.py`)

Each raw best-feature's \|r\| partitioned into diurnal / annual / dynamics
(harmonic regression of Z on absolute time):

| Mode | best \|r\| | r diurnal | r annual | r dynamics | dominated by |
|------|-----------|-----------|----------|------------|--------------|
| X0 | 0.590 | 0.143 | **0.578** | 0.239 | annual |
| X1 | 0.647 | 0.193 | **0.623** | 0.213 | annual |
| X2 | 0.594 | 0.184 | **0.568** | 0.227 | annual |
| X3 | 0.463 | 0.085 | 0.329 | 0.340 | annual ≈ dynamics |
| X4 | 0.590 | 0.050 | **0.649** | 0.179 | annual (diurnal≈0) |
| X5 | 0.385 | 0.190 | 0.317 | 0.216 | annual |
| X6 | 0.438 | 0.167 | 0.086 | **0.398** | **dynamics** |
| X7 | 0.508 | 0.171 | 0.445 | 0.269 | annual |

- The cycle-tracking is **annual**, not diurnal: r annual (0.32–0.65) ≫ r diurnal
  (0.05–0.19) everywhere. (Annual is 32.9% of variance, diurnal only 11.5%.)
- **Monosemantic set swaps, not shrinks.** Raw monosemantic X1, X4 are
  *annual-dominated* and only marginally specific (spec +0.06/+0.09) — they vanish
  on deseason. X6, the one mode with negligible annual (r annual 0.086, southern
  near-sink), has a feature that was **already dynamics-based** (r dynamics 0.398,
  highest) → it survives as the lone genuine monosemantic.
- **Consistency:** deseason best-r ≈ raw r-dynamics per mode (e.g. X6 0.435≈0.398,
  X3 0.392≈0.340), confirming the dynamics signal was always there and the annual
  cycle merely inflated the raw numbers.
