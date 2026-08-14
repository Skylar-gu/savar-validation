# Project description — GraphCast causal-structure interpretability

Seed document for the eventual paper. Audience: advisor + collaborators.

## Structure
- `main.tex` — preamble + `\input` of sections + bibliography
- `sections/intro.tex` — motivation, the correlation-vs-causation question, the two probes (SAE + VPD), the gap vs. MacMillan & Ouellette (2025)
- `sections/background.tex` — GraphCast architecture, SAEs, PCMCI/PCMCI+/TSCI/DYNOTEARS, LiNGAM identifiability, parameter decomposition (SPD/VPD), SAVAR
- `sections/approach.tex` — the two routes (SAE→causal-discovery, VPD) + the SAE-temporal subtlety
- `sections/savar_results.tex` — completed SAVAR results on the default linear-Gaussian setting (the original validation)
- `sections/extensions.tex` — fine-cadence generator, subsampling/PCMCI+ sweep, MeshGNN forecaster, GNN SAEs, VPD results, mode-homogeneity diagnosis + the heterogeneous-dynamics specialization test (split result)
- `sections/selector.tex` — the answer-key-free selector: candidate pool, read/poke channels, why same-map agreement fails, pool-crossed agreement (PX), rungs R1/R3/R5/R4b, mechanism-robustness matrix, trust dial v2 + per-domain calibration
- `sections/open_questions.tex` — risks/limitations
- `sections/contributions.tex` — contributions + milestones
- `refs.bib` — bibliography (some entries marked TODO to verify)

## Build
```
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Open TODOs
- Verify bib entries marked TODO (savar, tsci, macmillan venue, spd/vpd arXiv ids).
- Author list; abstract (write last).
- `selector.tex` reports a selected subset of the rung results. The full record —
  R2 static-inputs, R4 (pre-fair-pool, +0.357), R6 atmosphere-regime, and the E2
  consistency-score program — is in `notes/literature_extension_results.md` if any
  of it is wanted in an appendix.
- Consider a figure from `results/plots/trust_dial_paper.pdf` in Sec. "trust dial".
- Optional: add the diurnal / dy005 / deseason variants to the method table
  (`results/baseline_*_{diurnal,dy005,...}.npy`) if a robustness column is wanted.
