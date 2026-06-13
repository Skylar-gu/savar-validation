#!/usr/bin/env bash
# Overnight diurnal pipeline: GPU check → CNN train → (PCMCI concurrent) →
# extract activations → per-mode SAE → SAE eval → cycle/PC0 analysis.
# Robust: each step logged separately; STATUS.txt is the at-a-glance summary.
set -uo pipefail
cd /home/ec2-user/savar-project
mkdir -p logs_diurnal
STATUS=logs_diurnal/STATUS.txt
ts() { date '+%Y-%m-%d %H:%M:%S'; }
: > "$STATUS"
echo "[$(ts)] pipeline started" | tee -a "$STATUS"

run() {  # run <name> <logfile> <cmd...>
  local name=$1 log=$2; shift 2
  echo "[$(ts)] START $name" | tee -a "$STATUS"
  if "$@" > "$log" 2>&1; then
    echo "[$(ts)] OK    $name" | tee -a "$STATUS"
  else
    echo "[$(ts)] FAIL  $name (rc=$? — see $log)" | tee -a "$STATUS"
    echo "[$(ts)] PIPELINE ABORTED" | tee -a "$STATUS"
    exit 1
  fi
}

# 0. Fail fast if the GPU isn't actually usable
run "00 verify_gpu" logs_diurnal/00_verify_gpu.log python3 train/verify_gpu.py

# PCMCI is CPU-only → run it concurrently with the GPU work
echo "[$(ts)] START 02 pcmci (background, CPU)" | tee -a "$STATUS"
( if python3 pcmci/run_pcmci_diurnal.py > logs_diurnal/02_pcmci.log 2>&1; then
    echo "[$(ts)] OK    02 pcmci" >> "$STATUS"
  else
    echo "[$(ts)] FAIL  02 pcmci (see logs_diurnal/02_pcmci.log)" >> "$STATUS"
  fi ) &
PCMCI_PID=$!

# 1. Train the CNN on the diurnal splits (GPU)
run "01 cnn_train" logs_diurnal/01_cnn_train.log python3 train/cnn_forecaster.py --diurnal

# 3. Extract res3 activations. The PCA gate may exit non-zero; that is advisory —
#    activations_full.npy is written before it, so we continue if the file exists.
echo "[$(ts)] START 03 extract" | tee -a "$STATUS"
python3 sae/extract_activations.py --diurnal > logs_diurnal/03_extract.log 2>&1
EX_RC=$?
if [ -f sae_data_diurnal/activations_full.npy ]; then
  [ $EX_RC -ne 0 ] && echo "[$(ts)] WARN  03 extract gate exit rc=$EX_RC (activations saved; continuing)" | tee -a "$STATUS"
  echo "[$(ts)] OK    03 extract" | tee -a "$STATUS"
else
  echo "[$(ts)] FAIL  03 extract (no activations_full.npy — see log)" | tee -a "$STATUS"
  echo "[$(ts)] PIPELINE ABORTED" | tee -a "$STATUS"; exit 1
fi

# 4. Per-mode SAEs (GPU)
run "04 sae_train" logs_diurnal/04_sae_train.log python3 sae/train_sae_per_mode.py --diurnal
# 5. SAE alignment / monosemanticity
run "05 sae_eval"  logs_diurnal/05_sae_eval.log  python3 sae/eval_sae_per_mode.py  --diurnal
# 6. Cycle vs PC0 collapse analysis
run "06 cycle_pc0" logs_diurnal/06_cycle_pc0.log python3 sae/analyze_cycle_pc0.py

# Wait for the concurrent PCMCI job
echo "[$(ts)] waiting for PCMCI ..." | tee -a "$STATUS"
wait "$PCMCI_PID" 2>/dev/null || true

echo "[$(ts)] PIPELINE COMPLETE" | tee -a "$STATUS"
echo "" | tee -a "$STATUS"
echo "Key results:" | tee -a "$STATUS"
grep -E "Best val RMSE" logs_diurnal/01_cnn_train.log 2>/dev/null | tail -1 | tee -a "$STATUS" || true
grep -E "SUMMARY" logs_diurnal/02_pcmci.log 2>/dev/null | tail -1 | tee -a "$STATUS" || true
grep -E "Aligned|Strong" logs_diurnal/05_sae_eval.log 2>/dev/null | tee -a "$STATUS" || true
grep -E "Mean PC0 var" logs_diurnal/06_cycle_pc0.log 2>/dev/null | tee -a "$STATUS" || true
