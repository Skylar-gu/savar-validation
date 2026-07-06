#!/usr/bin/env bash
# SERIALIZED resume of the eqvar pipeline: wait until the as-is pipeline finishes
# (freeing the GPU), then run eqvar train -> extract -> SAE -> eval -> viz.
# Data + splits already generated/verified, so steps [0] and [1] are skipped.
set -euo pipefail
cd /home/ec2-user/savar-project

echo "############ WAIT for as-is pipeline to finish  $(date) ############"
while true; do
  if grep -q "PIPELINE DONE" scratch_hetdyn_pipeline.log 2>/dev/null; then
    echo "as-is pipeline reported DONE — starting eqvar $(date)"; break
  fi
  if ! pgrep -f "[s]cratch_hetdyn_pipeline.sh" >/dev/null 2>&1; then
    echo "as-is pipeline process gone (finished or failed) — starting eqvar $(date)"; break
  fi
  sleep 60
done

echo "############ [2/6] TRAIN GNN (eqvar) $(date) ############"
GNN_CKPT_DIR=checkpoints/hetdynamics_eqvar GNN_SPLIT_DIR=data/splits_hetdynamics_eqvar \
    python3 train/gnn/gnn_forecaster.py

echo "############ [3/6] EXTRACT ACTIVATIONS $(date) ############"
python3 sae/extract_activations_gnn.py \
    --ckpt checkpoints/hetdynamics_eqvar/best.pt \
    --data data/realisations_hetdynamics_eqvar \
    --out  sae_data/hetdynamics_eqvar --stride 5

echo "############ [4/6] TRAIN PER-MODE SAE $(date) ############"
python3 sae/train_sae_per_mode.py --gnn --datadir sae_data/hetdynamics_eqvar

echo "############ [5/6] EVAL PER-MODE SAE $(date) ############"
python3 sae/eval_sae_per_mode.py --gnn --datadir sae_data/hetdynamics_eqvar

echo "############ [6/6] 3D VISUALIZATION $(date) ############"
python3 sae/visualize_features_3d.py \
    --data sae_data/hetdynamics_eqvar --out figures --tag hetdynamics_eqvar --hero 7

echo "############ EQVAR PIPELINE DONE $(date) ############"
