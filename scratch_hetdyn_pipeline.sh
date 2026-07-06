#!/usr/bin/env bash
# Het-dynamics mode-specialization experiment: split -> train GNN -> extract -> SAE -> eval -> 3D viz.
set -euo pipefail
cd /home/ec2-user/savar-project

echo "############ [1/6] SPLIT $(date) ############"
FC_REAL_DIR=data/realisations_hetdynamics FC_SPLIT_DIR=data/splits_hetdynamics \
    python3 data_gen/split_finecadence.py

echo "############ [2/6] TRAIN GNN $(date) ############"
GNN_CKPT_DIR=checkpoints/hetdynamics GNN_SPLIT_DIR=data/splits_hetdynamics \
    python3 train/gnn_forecaster.py

echo "############ [3/6] EXTRACT ACTIVATIONS $(date) ############"
python3 sae/extract_activations_gnn.py \
    --ckpt checkpoints/hetdynamics/best.pt \
    --data data/realisations_hetdynamics \
    --out  sae_data_hetdynamics --stride 5

echo "############ [4/6] TRAIN PER-MODE SAE $(date) ############"
python3 sae/train_sae_per_mode.py --gnn --datadir sae_data_hetdynamics

echo "############ [5/6] EVAL PER-MODE SAE $(date) ############"
python3 sae/eval_sae_per_mode.py --gnn --datadir sae_data_hetdynamics

echo "############ [6/6] 3D VISUALIZATION $(date) ############"
python3 sae/visualize_features_3d.py \
    --data sae_data_hetdynamics --out figures --tag hetdynamics --hero 7

echo "############ PIPELINE DONE $(date) ############"
