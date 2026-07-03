# Project description — GraphCast causal-structure interpretability

Seed document for the eventual paper. Audience: advisor + collaborators.

## Structure
- `main.tex` — preamble + `\input` of sections + bibliography
- `sections/intro.tex` — motivation, the correlation-vs-causation question, the two probes (SAE + VPD), the gap vs. MacMillan & Ouellette (2025)
- `sections/background.tex` — GraphCast architecture, SAEs, PCMCI/PCMCI+/TSCI/DYNOTEARS, LiNGAM identifiability, parameter decomposition (SPD/VPD), SAVAR
- `sections/approach.tex` — the two routes (SAE→causal-discovery, VPD) + the SAE-temporal subtlety
- `sections/savar_results.tex` — completed SAVAR results on the default linear-Gaussian setting (the original validation)
- `sections/extensions.tex` — fine-cadence generator, subsampling/PCMCI+ sweep, MeshGNN forecaster, GNN SAEs, VPD results, mode-homogeneity diagnosis
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
- Add heterogeneous-dynamics (specialization) results to `extensions.tex` when the
  three-way runs complete (see `\todo` marker there).
- Optional: add the diurnal / dy005 / deseason variants to the method table
  (`results/baseline_*_{diurnal,dy005,...}.npy`) if a robustness column is wanted.
