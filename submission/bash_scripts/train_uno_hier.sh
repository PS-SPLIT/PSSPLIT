#!/bin/bash
# Train the UNO-hier backbone on each dataset (seed=1, prop_train_labels=1.0).
# UNO uses Sinkhorn-Knopp pseudo-labels; ntxent_mode/supcon_mode do not apply.
set -e
export CUDA_VISIBLE_DEVICES=0

TRAIN_COMMON="--prop_train_labels 1.0
  --use_coarse_sampler False
  --lr 0.1 --weight_decay 5e-5 --seed 1
  --grad_from_block 11 --batch_size 128
  --transform imagenet --eval_funcs v1 v2
  --train_report_interval 25"

EXP_ID=uno_hier_seed1_25ep

echo "=== UNO-hier: aves_small ==="
python -m methods.uno_hier_training.uno_hier_training \
  --dataset_name aves_small --hier_split_type 5050 \
  --coarse_labels_path data/coarse_labels/aves_small.json \
  --epochs 25 $TRAIN_COMMON --exp_id $EXP_ID

for DATASET in cub scars aircraft; do
  echo "=== UNO-hier: $DATASET ==="
  python -m methods.uno_hier_training.uno_hier_training \
    --dataset_name $DATASET --hier_split_type 5050 \
    --coarse_labels_path data/coarse_labels/${DATASET}.json \
    --epochs 25 $TRAIN_COMMON --exp_id $EXP_ID
done

echo "=== UNO-hier: aves_full ==="
python -m methods.uno_hier_training.uno_hier_training \
  --dataset_name aves_full --hier_split_type 5050 \
  --coarse_labels_path data/coarse_labels/aves_full.json \
  --epochs 25 $TRAIN_COMMON --exp_id $EXP_ID
