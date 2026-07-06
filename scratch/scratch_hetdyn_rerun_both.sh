#!/usr/bin/env bash
# Serialized rerun of BOTH het-dynamics pipelines (as-is, then eqvar).
# Data + splits already exist for both, so gen/split steps are skipped.
# Run inside tmux so it survives session exit.
set -euo pipefail
cd /home/ec2-user/savar-project

echo "################ AS-IS: TRAIN GNN $(date) ################"
GNN_CKPT_DIR=checkpoints/hetdynamics GNN_SPLIT_DIR=data/splits_hetdynamics \
    python3 train/gnn/gnn_forecaster.py

echo "################ AS-IS: EXTRACT ACTIVATIONS $(date) ################"
python3 sae/extract_activations_gnn.py \
    --ckpt checkpoints/hetdynamics/best.pt \
    --data data/realisations_hetdynamics \
    --out  sae_data/hetdynamics --stride 5

echo "################ AS-IS: TRAIN PER-MODE SAE $(date) ################"
python3 sae/train_sae_per_mode.py --gnn --datadir sae_data/hetdynamics

echo "################ AS-IS: EVAL PER-MODE SAE $(date) ################"
python3 sae/eval_sae_per_mode.py --gnn --datadir sae_data/hetdynamics

echo "################ AS-IS: 3D VISUALIZATION $(date) ################"
python3 sae/visualize_features_3d.py \
    --data sae_data/hetdynamics --out figures --tag hetdynamics --hero 7

echo "################ AS-IS PIPELINE DONE $(date) ################"

echo "################ EQVAR: TRAIN GNN $(date) ################"
GNN_CKPT_DIR=checkpoints/hetdynamics_eqvar GNN_SPLIT_DIR=data/splits_hetdynamics_eqvar \
    python3 train/gnn/gnn_forecaster.py

echo "################ EQVAR: EXTRACT ACTIVATIONS $(date) ################"
python3 sae/extract_activations_gnn.py \
    --ckpt checkpoints/hetdynamics_eqvar/best.pt \
    --data data/realisations_hetdynamics_eqvar \
    --out  sae_data/hetdynamics_eqvar --stride 5

echo "################ EQVAR: TRAIN PER-MODE SAE $(date) ################"
python3 sae/train_sae_per_mode.py --gnn --datadir sae_data/hetdynamics_eqvar

echo "################ EQVAR: EVAL PER-MODE SAE $(date) ################"
python3 sae/eval_sae_per_mode.py --gnn --datadir sae_data/hetdynamics_eqvar

echo "################ EQVAR: 3D VISUALIZATION $(date) ################"
python3 sae/visualize_features_3d.py \
    --data sae_data/hetdynamics_eqvar --out figures --tag hetdynamics_eqvar --hero 7

echo "################ EQVAR PIPELINE DONE $(date) ################"
echo "################ ALL DONE $(date) ################"
