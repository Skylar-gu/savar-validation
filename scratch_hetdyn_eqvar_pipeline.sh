#!/usr/bin/env bash
# Het-dynamics EQUAL-VARIANCE variant: isolates the timescale manipulation.
# generate (eqvar) -> split -> train GNN -> extract -> SAE -> eval -> 3D viz.
set -euo pipefail
cd /home/ec2-user/savar-project

echo "############ [0/6] GENERATE eqvar $(date) ############"
# Empirical calibration: innovation scales ∝ 1/std_asis (from the as-is run's
# per-mode stds [1.03,1.07,1.13,1.18,1.21,1.37,1.47,1.45]); generator renormalises
# to mean 1. Equalises variance while preserving the timescale spread.
HD_INNOV_SCALE="0.97,0.93,0.89,0.85,0.83,0.73,0.68,0.69" \
    python3 data_gen/generate_hetdynamics.py

echo "############ [1/6] SPLIT $(date) ############"
FC_REAL_DIR=data/realisations_hetdynamics_eqvar FC_SPLIT_DIR=data/splits_hetdynamics_eqvar \
    python3 data_gen/split_finecadence.py

echo "############ [2/6] TRAIN GNN $(date) ############"
GNN_CKPT_DIR=checkpoints_hetdynamics_eqvar GNN_SPLIT_DIR=data/splits_hetdynamics_eqvar \
    python3 train/gnn_forecaster.py

echo "############ [3/6] EXTRACT ACTIVATIONS $(date) ############"
python3 sae/extract_activations_gnn.py \
    --ckpt checkpoints_hetdynamics_eqvar/best.pt \
    --data data/realisations_hetdynamics_eqvar \
    --out  sae_data_hetdynamics_eqvar --stride 5

echo "############ [4/6] TRAIN PER-MODE SAE $(date) ############"
python3 sae/train_sae_per_mode.py --gnn --datadir sae_data_hetdynamics_eqvar

echo "############ [5/6] EVAL PER-MODE SAE $(date) ############"
python3 sae/eval_sae_per_mode.py --gnn --datadir sae_data_hetdynamics_eqvar

echo "############ [6/6] 3D VISUALIZATION $(date) ############"
python3 sae/visualize_features_3d.py \
    --data sae_data_hetdynamics_eqvar --out figures --tag hetdynamics_eqvar --hero 7

echo "############ EQVAR PIPELINE DONE $(date) ############"
