#!/bin/bash
# All-knobs pipeline: split -> CNN -> extract -> ceilings -> per-mode SAEs -> eval
# Run from ~/savar-validation. Each stage logs to logs_allknobs/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs_allknobs
EPOCHS="${EPOCHS:-25}"

echo "[$(date)] stage 1: split"
python3 data_gen/split_allknobs.py > logs_allknobs/01_split.log 2>&1

echo "[$(date)] stage 2: CNN train ($EPOCHS epochs)"
python3 train/cnn_forecaster.py --allknobs --epochs "$EPOCHS" > logs_allknobs/02_cnn_train.log 2>&1

echo "[$(date)] stage 3: extract activations"
python3 sae/extract_activations.py --allknobs > logs_allknobs/03_extract.log 2>&1

echo "[$(date)] stage 4: measure ceilings"
python3 sae/measure_ceilings.py --allknobs > logs_allknobs/04_ceilings.log 2>&1

echo "[$(date)] stage 5: train per-mode SAEs"
python3 sae/train_sae_per_mode.py --allknobs > logs_allknobs/05_sae_train.log 2>&1

echo "[$(date)] stage 6: eval"
python3 sae/eval_sae_per_mode.py --allknobs > logs_allknobs/06_sae_eval.log 2>&1

echo "[$(date)] stage 7: persistence baseline (val split)"
python3 baselines/persistence_allknobs.py > logs_allknobs/07_persistence.log 2>&1

echo "[$(date)] DONE"
