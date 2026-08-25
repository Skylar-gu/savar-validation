# SAVAR ground-truth calibration — minimal package

Everything behind the paper's SAVAR figure (`figures/paper_fig_savar.py`, panels (a) and (b))
in one self-contained tree: a linear VAR(2) climate-like benchmark with a **known** causal graph
([SAVAR, Tibau et al.](https://github.com/xtibau/savar)), a CNN and a MeshGNN forecaster trained
on it, and the SAE → PCMCI+ ladder that asks whether a forecaster's internal representation
exposes that graph once every oracle ingredient is removed.

This is the minimal subset of the public
[`savar-validation`](https://github.com/Skylar-gu/savar-validation) repository (branch
`refactor/minimal-package`); the many dataset variants, probes, steering and VPD experiments
that do not feed a paper number are left out.

## Paper number → script → artifact

| number (figure) | panel | produced by | shipped record |
|---|---|---|---|
| persistence RMSE 1.48, oracle floor 1.061 (D_y = I) | (a) | `data_gen/generate_dataset.py` → `data_gen/data_split.py`; `baselines/rmse_baselines.py` | `notes/REPO_SUMMARY_AND_AUDIT.md` (Phase 1a/5), `notes/rmse_baselines.md` (floor formula) |
| CNN val RMSE 1.072 | (a) | `train/cnn/cnn_forecaster.py` (+ `train/cnn/resume_training.py`, which finished the published run from epoch 30) | `checkpoints/base/best.pt` (not shipped); number in `notes/REPO_SUMMARY_AND_AUDIT.md` |
| PCMCI+ on true modes Z = **0.853** (bake-off ceiling); oracle-W pooling 0.855; varimax 0.819; k-means 0.280; DMD 0.177; misplaced footprints 0.000 | (b), PCMCI+ block | `data_gen/generate_hetdynamics.py` (eqvar knobs below) → `data_gen/split_finecadence.py` → `train/gnn/gnn_forecaster.py` → `bakeoff/discover_modes.py` | `results/litext_e1_discovery.npy` (`anchors.trueZ`, `anchors.oracle_acts_fu1`, `graph.{vmax_act,km_act,dmd_act,shift5}.F1`); `notes/literature_extension_results.md` §E1 |
| GNN end-to-end, PCMCI+ protocol: ceiling 0.859, **R3b 0.003** | (b), PCMCI+ block | `ladder_gnn/extract_stride1.py` → `LADDER_PROTOCOL=plus ladder_gnn/ladder_gnn.py --quick` | `results/ladder_gnn/ladder_gnn_eqvar_plus.npy` (`res.trueZ`, `res.R3b`) |
| GNN end-to-end, ladder protocol: ceiling 0.616, **R3b 0.019**, p = 0.31 vs random draws; R4 (per-node SAE) 0.004–0.010, p = 0.33–0.93 | (b), ladder block | `ladder_gnn/ladder_gnn.py`, `ladder_gnn/nulls_gnn.py`, `ladder_gnn/rung_r4_gnn.py` | `results/ladder_gnn/ladder_gnn_eqvar.npy`, `nulls_gnn_eqvar.npy`, `rung_r4_gnn.npy`; `ladder_gnn/README.md` (full tables) |
| CNN end-to-end: ceiling 0.825, **R3b 0.128** (P 0.080; 47 FP vs 12 true), p = 0.033; R4 0.004 (MAP-FOOT) – 0.079 (MAP-R), p = 0.287 | (b), ladder block | `sae/extract_activations.py` → `sae/train_sae_per_mode.py`, `sae/train_sae.py`, `sae/eval_sae_per_mode.py`; `ladder_cnn/ladder_r0_r3.py`, `nulls.py`, `rung_r4.py`, `r4_post.py` | `results/ladder_cnn/ladder_r0_r3_base.npy` (`trueZ_anchor`, `res.R3b`), `nulls_base.npy`, `r4_null.npy`; `ladder_cnn/log_*.txt`; protocol frozen in `ladder_cnn/PREREG.md` |
| R0 (all oracles kept) 0.695 — the published feature-graph number the ladder's R0 reproduces | caption | `pcmci/run_pcmci_features.py` | `results/pcmci_features.npy` |
| PCMCI baseline on Z, base data: F1 0.825 (the ladder gate) | — | `pcmci/run_pcmci.py` | `results/pcmci_results.npy` |
| 0/8 per-mode SAE features monosemantic; PC0 share 86 % (CNN) / 88–92 % (GNN) | caption | `sae/eval_sae_per_mode.py` (`--gnn --datadir sae_data/hetdynamics_eqvar` for the GNN) | `notes/phase7_sae_findings.md` |

Every value in the table was re-read from the shipped `.npy` files when this package was
assembled (e.g. `ladder_r0_r3_base.npy` → R3b 0.128, trueZ 0.8249; `nulls_base.npy` → p 0.033;
`ladder_gnn_eqvar_plus.npy` → trueZ 0.859, R3b 0.003).

## Layout

```
paths.py          SAVAR_ROOT resolution (env var, default = this directory)
data_gen/         SAVAR model (instantiate_model.py), base generator + split, het-dynamics generator + split
train/cnn/        SpatioTemporalCNN + training / resume / GPU check
train/gnn/        MeshGNN forecaster (heterogeneous multi-scale mesh) + training
baselines/        persistence / oracle-floor RMSE
pcmci/            PCMCI on true modes Z (ceiling) and on aligned SAE features (R0)
sae/              activation extraction (CNN / GNN), TopK SAEs (per-mode / mixed), alignment eval
bakeoff/          E1 unsupervised mode-discovery bake-off -> litext_e1_discovery.npy
ladder_cnn/       oracle-ablation ladder on the CNN (PREREG.md = frozen protocol; log_*.txt = published run)
ladder_gnn/       the same ladder ported to the MeshGNN (README.md = full result tables)
results/          small result arrays (everything the figure reads)
notes/            the write-ups the figure header cites
tests/            invariants (tigramite index convention, ground-truth graph, CNN shapes, no absolute paths)
```

## Setup

```bash
git clone https://github.com/xtibau/savar.git savar && git -C savar checkout 532e5e5   # upstream library (gitignored)
pip install -r requirements.txt            # see the torch CUDA note inside
pip install -r requirements-dev.txt && pytest   # ~3 s, no GPU; data-dependent tests skip
```

`SAVAR_ROOT` (default: this directory) is the one root for `data/`, `checkpoints/`, `sae_data/`
and `results/`. Run every script **from `SAVAR_ROOT`** — the training / extraction scripts use
root-relative paths (`data/realisations`, `checkpoints/base`, ...), the ladders and the bake-off
resolve the root through `paths.py`.

## Regenerate

All commands from `SAVAR_ROOT`. GPU for the three training steps and the extractions; the
PCMCI stages are CPU-only and want `OMP_NUM_THREADS=1` **exported before python starts** (a
24-worker pool with default BLAS threading ran at load 114 and made no progress).

```bash
# ── (a) skill on the base dataset (D_y = I) ─────────────────────────────────
python3 data_gen/generate_dataset.py                    # data/realisations/ (100 x T=500)
python3 data_gen/data_split.py                          # data/splits/{train,val,test}
python3 train/cnn/verify_gpu.py && python3 train/cnn/cnn_forecaster.py     # checkpoints/base/best.pt, 50 epochs
#   (python3 train/cnn/resume_training.py checkpoints/base/epoch_030.pt continues an interrupted run, as the published one was)
python3 baselines/rmse_baselines.py                     # persistence 1.48 / CNN 1.072 / oracle 1.061

# ── ceiling + R0 on the base dataset ────────────────────────────────────────
python3 pcmci/run_pcmci.py                              # results/pcmci_results.npy   (F1 0.825)
python3 sae/extract_activations.py                      # sae_data/base/{activations_full,Z_full}.npy
python3 sae/train_sae_per_mode.py && python3 sae/train_sae.py              # sae_mode_{j}.pt, sae_best.pt
python3 sae/eval_sae_per_mode.py                        # sae_data/base/alignment_per_mode.npy
python3 pcmci/run_pcmci_features.py                     # results/pcmci_features.npy  (F1 0.695)

# ── CNN ladder (results/ladder_cnn/) ────────────────────────────────────────
export OMP_NUM_THREADS=1
python3 ladder_cnn/ladder_r0_r3.py --tag base --workers 24    # ladder_r0_r3_base.npy, r3b_series_base.npy
python3 ladder_cnn/nulls.py --tag base --r_null 10 --draws 150 --workers 24   # nulls_base.npy
python3 ladder_cnn/rung_r4.py --n_real 15 --draws 100 --workers 24            # spatial_sae_base.pt, r4_series_mean.npy, r4_footprints.npy
python3 ladder_cnn/r4_post.py --workers 24                                    # r4_null.npy (the R4 null draws; see note below)

# ── (b) het-dynamics (equal-variance) dataset + MeshGNN — the bake-off substrate ──
HD_INNOV_SCALE="0.97,0.93,0.89,0.85,0.83,0.73,0.68,0.69" python3 data_gen/generate_hetdynamics.py   # data/realisations_hetdynamics_eqvar/
FC_REAL_DIR=data/realisations_hetdynamics_eqvar FC_SPLIT_DIR=data/splits_hetdynamics_eqvar python3 data_gen/split_finecadence.py
GNN_CKPT_DIR=checkpoints/hetdynamics_eqvar GNN_SPLIT_DIR=data/splits_hetdynamics_eqvar python3 train/gnn/gnn_forecaster.py
python3 sae/extract_activations_gnn.py --ckpt checkpoints/hetdynamics_eqvar/best.pt \
        --data data/realisations_hetdynamics_eqvar --out sae_data/hetdynamics_eqvar --stride 5
python3 sae/train_sae_per_mode.py --gnn --datadir sae_data/hetdynamics_eqvar      # sae_mode_{j}.pt
python3 sae/train_sae_mixed.py --datadir sae_data/hetdynamics_eqvar                # sae_mixed.pt
python3 sae/eval_sae_per_mode.py --gnn --datadir sae_data/hetdynamics_eqvar       # alignment (0/8 monosemantic)
python3 bakeoff/discover_modes.py                        # results/litext_e1_discovery.npy (anchors + candidate F1s)

# ── GNN ladder (results/ladder_gnn/) ────────────────────────────────────────
python3 ladder_gnn/extract_stride1.py                    # activations_stride1_all.npy (2 GB, stride 1 — stride 5 breaks every lag)
bash ladder_gnn/run_all.sh                               # ladder -> nulls -> R4 -> PCMCI+ ("plus") pass
```

Large intermediates that are regenerated by the commands above and deliberately **not
shipped** (gitignored): `results/ladder_gnn/activations_stride1_all.npy` (2 GB) and
`Z_stride1_all.npy` (7.7 MB) from `extract_stride1.py`; `r3b_series_*.npy` (11 MB each) from the
ladder scripts; `r4_series_mean*.npy` (74 / 15 MB), `r4_footprints*.npy` (10 MB) and
`spatial_sae_*.pt` from the R4 rungs; all datasets, checkpoints and `sae_data/`.

## Provenance notes

- **Which clone each file came from.** The public repo has two divergent lines: `main`
  (CNN pipeline, Phase 1a numbers) and `agent/session-2026-07-03` (GNN forecaster,
  het-dynamics data, bake-off, 271 extra files). Files present on both differ only in the
  checkpoint directory convention (`checkpoints_dy005/` vs `checkpoints/dy005/`) and in
  variant-specific CLI flags (`--allknobs` vs `--binary`); this package takes the
  `agent/session-2026-07-03` versions throughout so the ladders' `checkpoints/base/best.pt`
  convention is consistent. `data_gen/instantiate_model.py`, `generate_dataset.py`,
  `pcmci/run_pcmci.py`, `verify_gpu.py` and the tests are byte-identical on both lines.
  `bakeoff/discover_modes.py` is the committed `agent/session-2026-07-03` version that wrote
  the shipped `litext_e1_discovery.npy` (6 Jul), not the later uncommitted variant with the
  `E1_NMODES`/`E1_GRID` knobs.
- **Ladders.** `ladder_cnn/` and `ladder_gnn/` were developed as read-only scratch dirs
  outside the repo (`causal-graphcast/audit_pcmci_assumptions/savar_sae_pcmci{,_gnn}`) with
  absolute paths; the only edits here are `ROOT`/`OUT` now coming from `paths.py` and each
  script putting its own directory on `sys.path`. Result filenames and CLIs are unchanged.
  `ladder_cnn/log_r4.txt` ends in a `NameError` in the R4 null stage of `rung_r4.py`; the
  shipped `r4_null.npy` was produced by the follow-up `r4_post.py` (see `log_r4post.txt`),
  which is the R4 number in the figure.
- `sae/extract_activations_gnn.py` keeps its `--variant` flag; the non-plain variants
  (`mesh_gnn_variants.py`) are not part of this package and are only imported if you ask for one.
- The figure's bake-off rows are the **GNN** on `hetdynamics_eqvar` under the PCMCI+ protocol;
  the CNN end-to-end rows are the `base` dataset under the ladder protocol. Only the estimator
  is shared — keep the two blocks separate in any caption.
