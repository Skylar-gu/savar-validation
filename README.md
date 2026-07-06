# SAVAR Validation

Validation pipeline for **causal interpretability of a CNN forecaster** trained on
[SAVAR](https://github.com/xtibau/savar) (Spatially Aggregated VAR) data — a
spatiotemporal stochastic climate model for benchmarking causal-discovery methods
([Tibau et al., *Environmental Data Science*](https://www.cambridge.org/core/journals/environmental-data-science/article/spatiotemporal-stochastic-climate-model-for-benchmarking-causal-discovery-methods-for-teleconnections/0E066B8813BA2281D2B95279EF3272B4)).

The goal: train a CNN to forecast spatial observations, then test whether its
internal representations encode the **ground-truth causal structure** of the
underlying latent modes (via PCMCI on the modes and sparse autoencoders on the
CNN activations).

This repo also adds a **GraphCast-like extension**: a 6-hourly dataset with
deterministic atmospheric cycles (diurnal + annual, longitude/latitude-phased,
plus afternoon heteroskedasticity) to study how deterministic cycles confound
both causal discovery and learned representations — and how ensemble-mean
deseasonalization removes that confound. See `notes/diurnal_datagen_summary.md`.

## Layout

| Folder | Contents |
|--------|----------|
| `data_gen/` | model definition + dataset generators (`instantiate_model.py`, `generate_*.py`, `split_*.py`) |
| `train/cnn/` | CNN forecaster + training (`cnn_forecaster.py`, `train_dy005.py`, `resume_training.py`, `verify_gpu.py`) |
| `train/gnn/` | MeshGNN forecaster + architecture variants (`gnn_forecaster.py`, `mesh_gnn_variants.py`) |
| `pcmci/` | PCMCI / PCMCI+ causal discovery + DMD timescale checks |
| `sae/` | activation extraction, per-mode / mixed SAEs, evaluation, probes, steering |
| `vpd/` | adVersarial Parameter Decomposition on the GNN (uses vendored `param-decomp/`) |
| `baselines/` | RMSE baselines (oracle / persistence) |
| `visualization/` | model / mode visualizations |
| `notes/` | pipeline spec, requirements, and results write-ups (start with `summary_jul6.md`) |
| `checkpoints/<variant>/` | trained model checkpoints, one subfolder per experiment (gitignored) |
| `sae_data/<variant>/` | extracted activations + trained SAEs, one subfolder per experiment (gitignored) |
| `data/` | generated datasets, one subfolder per experiment (gitignored) |
| `results/` | small result arrays (`.npy`) |
| `figures/` | generated figures |
| `scratch/` | ad-hoc / historical exploration scripts and pipeline logs |

Orchestrators (run from repo root): `run_diurnal_pipeline.sh` (CNN → PCMCI → SAE)
and `run_sae_deseason.sh` (deseasonalized SAE chain).

## Setup

```bash
# 1. Clone the upstream SAVAR library into ./savar (gitignored here)
git clone https://github.com/xtibau/savar.git savar   # commit 532e5e5

# 2. Python deps (pinned)
pip install -r requirements.txt
```

See `requirements.txt` for pinned versions (incl. the PyTorch CUDA 12.4 build note).

GPU (CUDA) is used for CNN/SAE training; data generation and PCMCI are CPU-only.

## Running

**All scripts must be run from the repo root** (paths are root-relative):

```bash
python3 data_gen/generate_diurnal.py     # generate the diurnal dataset → data/
python3 data_gen/split_diurnal.py        # 70/15/15 split → data/splits_diurnal/
bash    run_diurnal_pipeline.sh          # CNN train → PCMCI (raw+deseason) → SAE
bash    run_sae_deseason.sh              # deseasonalized SAE chain
```

## Tests

Repeatable checks for the invariants the pipeline relies on (cycle math, the
tigramite `(cause, eff, tau)` convention, ground-truth graph structure, the
deseasonalization identity, and regression guards on the dynamic-`T` fix and
post-reorg import paths):

```bash
pip install -r requirements-dev.txt
pytest                       # ~3s; no GPU required
```

Tests that need the upstream `savar` library or generated data skip cleanly when
those aren't present. `diagnostics.py` provides run-context breadcrumbs
(`run_context(...)` / `run_failure(...)`) logging device/CUDA, git commit, and
dataset/checkpoint/output paths for diagnosing cluster runs.

## Key results (diurnal/annual run)

- **PCMCI**: raw F1 = 0.293 (cycle confounds → 58 false positives) → deseasonalized
  F1 = **0.825**, exactly restoring the no-cycle baseline.
- **SAE**: raw "strong" features are mostly **annual-cycle-tracking**, not dynamics;
  deseasonalizing collapses alignment (5/8 → 0/8 strong), unmasking the CNN's
  intrinsic polysemantic global-activity encoding in the low-noise regime.

Full write-up: `notes/diurnal_datagen_summary.md`.
