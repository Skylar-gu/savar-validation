# All-knobs per-mode SAE alignment (2026-08-23)

The Phase-7 feature–mode alignment experiment re-run with every stress knob on
simultaneously: diurnal + annual forcing (with afternoon heteroskedasticity),
nonlinear latent dynamics (saturating AR alpha 0.5, bilinear coupling beta
0.15), skew-normal innovations (alpha 4, unit variance), and per-realisation
mode placement (movmech MOVE=place, coarse-grid: sigma 2.5 blobs, disjoint by
rejection sampling, W per realisation; forcing phases follow the realised
centres). Same 8-mechanism VAR(2) edge set as every other variant. D_y =
0.05·I, T = 2920 at 6 h, 100 realisations.

Pipeline: `data_gen/generate_allknobs.py` → `data_gen/split_allknobs.py` →
`train/cnn_forecaster.py --allknobs --epochs 25` → `sae/extract_activations.py
--allknobs` → `sae/measure_ceilings.py --allknobs` →
`sae/train_sae_per_mode.py --allknobs` → `sae/eval_sae_per_mode.py --allknobs`
(driver: `run_allknobs_pipeline.sh`; logs in `logs_allknobs/`).

## Forecaster

Val RMSE **0.6741** (val corr 0.463) vs persistence 0.8721. No analytic oracle
floor exists for the nonlinear generator, so only the persistence bracket is
reported. 25 epochs, 247 s/epoch, converged (epoch-1 val RMSE was already
0.6846).

## Representation

PCA of the pooled mode-weighted res3 activations (all modes stacked):
**PC0 = 75.4%**, PC1 = 17.0%, PC2 = 2.5%. The global-activity direction is
softened relative to the linear-Gaussian baseline (86%) but still dominant.

Linear ceilings (5-fold out-of-fold ridge over realisations, alpha by inner
RidgeCV): X0 0.518, X1 0.562, X2 0.501, X3 0.595, X4 0.492, X5 0.421,
X6 0.552, X7 0.531 — slightly HIGHER than the baseline's 0.36–0.59, i.e. the
knobs did not destroy the linearly decodable mode signal.

## Per-mode TopK SAE alignment (256→512, K=25)

| Mode | Best feat | max\|r\| | Ceiling | Frac | Specificity | Status |
|------|-----------|---------|---------|------|-------------|--------|
| X0   | f251      | 0.423   | 0.518   | 0.82 | −0.072      | ALIGN / polysemantic |
| X1   | f137      | 0.402   | 0.562   | 0.72 | −0.034      | ALIGN / polysemantic |
| X2   | f241      | 0.338   | 0.501   | 0.68 | −0.083      | FAIL / polysemantic |
| X3   | f297      | 0.432   | 0.595   | 0.73 | +0.035      | ALIGN / polysemantic |
| X4   | f275      | 0.347   | 0.492   | 0.70 | −0.073      | FAIL / polysemantic |
| X5   | f113      | 0.250   | 0.421   | 0.59 | −0.188      | FAIL / polysemantic |
| X6   | f336      | 0.392   | 0.552   | 0.71 | −0.034      | ALIGN / polysemantic |
| X7   | f8        | 0.365   | 0.531   | 0.69 | −0.074      | ALIGN / polysemantic |

Aligned (|r| ≥ 0.35): **5/8**. Strong (≥ 0.5): **0/8**. Monosemantic
(specificity > 0.05): **0/8**. Best specificity +0.035 (X3); every other
mode's best feature correlates MORE with some other mode's state than with
its own.

## Reading

The linear-Gaussian escape hatch is closed and the result stands: with
nonlinear, non-Gaussian, seasonally forced dynamics and modes that move
between realisations, the CNN still encodes global system excitation rather
than mode-specific states. Alignment is slightly worse than baseline (5/8 vs
7/8 at the same 0.35 bar, 59–82% of ceiling vs 77–79%) while the ceilings
themselves are slightly higher — the representation, not the readout, remains
the bottleneck. Monosemanticity stays exactly where it was: 0/8.

Caveats: one CNN seed; 25 epochs (converged); the moving-mode blobs are
smaller (sigma 2.5, 11×11 support) than the fixed 16×16 slot blobs, a
necessary consequence of random disjoint placement on 50×50 — ceilings say
this did not cost decodable signal.
