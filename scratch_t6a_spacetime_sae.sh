#!/usr/bin/env bash
# T6a — space-time mixed SAE on the movmech nets (spec v2 §T6).
# ST=1 baseline + ST=3/5 on plain, ST=1/3 on slot; 3 seeds each; sequential (one GPU).
# Checkpoints land in the datadirs as sae_mixed_topk_seed<N>[_st<ST>]{,_final}.pt
set -uo pipefail
cd /home/ec2-user/savar-project
# 20 epochs on stride-1 movmech data = 127k optimizer steps (> Block C's 76k on
# eqvar); identical across ST/variant/seed so the comparison is internal.
export SAE_EPOCHS=20

run () {  # DIR ST SEED
  local DIR=$1 ST=$2 SEED=$3
  echo "########## [$DIR ST=$ST seed=$SEED] $(date) ##########"
  SAE_SPACETIME=$ST python3 sae/train_sae_mixed.py --datadir $DIR --seed $SEED \
    2>&1 | tail -5
}

for SEED in 0 1 2; do run sae_data_movmech_place      1 $SEED; done
for SEED in 0 1 2; do run sae_data_movmech_place      3 $SEED; done
for SEED in 0 1 2; do run sae_data_movmech_place_slot 1 $SEED; done
for SEED in 0 1 2; do run sae_data_movmech_place_slot 3 $SEED; done
for SEED in 0 1 2; do run sae_data_movmech_place      5 $SEED; done
echo "########## T6a SWEEP DONE $(date) ##########"
