#!/usr/bin/env bash
# GNN ladder, full run: ladder -> nulls -> R4 -> PCMCI+ protocol pass.
# SAVAR_ROOT defaults to the package dir (parent of this script); logs go to
# $SAVAR_ROOT/results/ladder_gnn/ so the shipped log_*.txt (the published record) stay intact.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SAVAR_ROOT="${SAVAR_ROOT:-$(dirname "$HERE")}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # see README ops note
LOG="$SAVAR_ROOT/results/ladder_gnn"; mkdir -p "$LOG"
cd "$SAVAR_ROOT"
python3 "$HERE/ladder_gnn.py" --workers 24 > "$LOG/log_ladder.txt" 2>&1
python3 "$HERE/nulls_gnn.py" --r_null 10 --draws 150 --workers 24 > "$LOG/log_nulls.txt" 2>&1
python3 "$HERE/rung_r4_gnn.py" --n_real 15 --draws 100 --workers 24 > "$LOG/log_r4.txt" 2>&1
LADDER_PROTOCOL=plus python3 "$HERE/ladder_gnn.py" --tag gnn_eqvar_plus --workers 24 > "$LOG/log_ladder_plus.txt" 2>&1
echo ALLDONE > "$LOG/DONE"
