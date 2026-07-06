#!/usr/bin/env bash
# Deseasonalized SAE chain (reuses existing sae_data/diurnal/ activations):
#   deseason → per-mode SAE → eval → cycle/PC0 analysis.
# Tests whether mode-specific dynamics features re-emerge once the shared cycle
# is removed (compare to the raw-diurnal run in STATUS.txt).
set -uo pipefail
cd /home/ec2-user/savar-project
mkdir -p logs/diurnal
STATUS=logs/diurnal/STATUS_deseason.txt
ts() { date '+%Y-%m-%d %H:%M:%S'; }
: > "$STATUS"
echo "[$(ts)] deseason SAE chain started" | tee -a "$STATUS"

run() {  # run <name> <logfile> <cmd...>
  local name=$1 log=$2; shift 2
  echo "[$(ts)] START $name" | tee -a "$STATUS"
  if "$@" > "$log" 2>&1; then
    echo "[$(ts)] OK    $name" | tee -a "$STATUS"
  else
    echo "[$(ts)] FAIL  $name (rc=$? — see $log)" | tee -a "$STATUS"
    echo "[$(ts)] ABORTED" | tee -a "$STATUS"; exit 1
  fi
}

run "d1 deseason"  logs/diurnal/d1_deseason.log  python3 sae/deseason_activations.py --diurnal
run "d2 sae_train" logs/diurnal/d2_sae_train.log python3 sae/train_sae_per_mode.py --diurnal --deseason
run "d3 sae_eval"  logs/diurnal/d3_sae_eval.log  python3 sae/eval_sae_per_mode.py  --diurnal --deseason
run "d4 cycle_pc0" logs/diurnal/d4_cycle_pc0.log python3 sae/analyze_cycle_pc0.py  --diurnal --deseason

echo "[$(ts)] DESEASON SAE COMPLETE" | tee -a "$STATUS"
echo "" | tee -a "$STATUS"
echo "Deseasonalized results:" | tee -a "$STATUS"
grep -E "Aligned|Strong"            logs/diurnal/d3_sae_eval.log | tee -a "$STATUS" || true
grep -E "Monosemantic|Global / pol" logs/diurnal/d3_sae_eval.log | tee -a "$STATUS" || true
grep -E "Mean PC0 var|Mean R."      logs/diurnal/d4_cycle_pc0.log | tee -a "$STATUS" || true
