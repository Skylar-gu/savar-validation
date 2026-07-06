#!/usr/bin/env bash
# Architecture sweep (v2 step 3): train each variant on MOVE=place, extract
# stride-1 activations, run P1/P2/P3. Sequential (one GPU). Plain net done
# separately. Logs per variant; results/movmech_posinv_place_<variant>.npy.
set -uo pipefail
cd /home/ec2-user/savar-project
export GNN_EPOCHS=10 GNN_BATCH=64   # val corr plateaus by epoch ~3 (see plain net)
RAW=data/realisations_movmech_place
SPLIT=data/splits_movmech_place

for V in blurpool refframe slot; do
  CKPT=checkpoints/movmech_place_$V
  ADIR=sae_data_movmech_place_$V
  echo "############ [$V] TRAIN $(date) ############"
  GNN_VARIANT=$V GNN_CKPT_DIR=$CKPT GNN_SPLIT_DIR=$SPLIT \
    python3 train/mesh_gnn_variants.py > logs/train_movmech_$V.log 2>&1
  echo "############ [$V] EXTRACT stride1 $(date) ############"
  mkdir -p $ADIR
  python3 sae/extract_activations_gnn.py --ckpt $CKPT/best.pt --data $RAW \
    --out $ADIR --stride 1 --variant $V --hook decoder > logs/extract_movmech_$V.log 2>&1
  echo "############ [$V] P1/P2/P3 $(date) ############"
  python3 sae/movmech_position_invariance.py --datadir $ADIR --raw $RAW \
    --tag place_$V --ckpt $CKPT/best.pt --variant $V > logs/p1_movmech_$V.log 2>&1
  echo "############ [$V] DONE $(date) ############"
done
echo "############ SWEEP DONE $(date) ############"
