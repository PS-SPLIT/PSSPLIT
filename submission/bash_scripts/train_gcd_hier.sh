#!/bin/bash
# Train the GCD-hier backbone on each dataset (seed=1, prop_train_labels=1.0).
# Run from the project root (the directory containing this folder's parent).
set -e
export CUDA_VISIBLE_DEVICES=0

# Best ablation config (see paper):
#   ntxent_mode=coarse, supcon_mode=global, csup=True, csupw=0.25, csamp=False
BEST="--ntxent_mode coarse --supcon_mode global
  --use_coarse_supcon True --coarse_supcon_weight 0.25
  --use_coarse_sampler False"

TRAIN_COMMON="--prop_train_labels 1.0
  --lr 0.1 --weight_decay 5e-5 --seed 1
  --grad_from_block 11 --batch_size 128
  --sup_con_weight 0.35 --contrast_unlabel_only False
  --transform imagenet --eval_funcs v1 v2
  --train_report_interval 100"

EXP_ID=gcd_hier_seed1_25ep

echo "=== GCD-hier: aves_small ==="
python -m methods.gcd_hier_training.gcd_hier_training \
  --dataset_name aves_small --hier_split_type 5050 \
  --coarse_labels_path data/coarse_labels/aves_small.json \
  --epochs 25 $TRAIN_COMMON $BEST --exp_id $EXP_ID

for DATASET in cub scars aircraft; do
  echo "=== GCD-hier: $DATASET ==="
  python -m methods.gcd_hier_training.gcd_hier_training \
    --dataset_name $DATASET --hier_split_type 5050 \
    --coarse_labels_path data/coarse_labels/${DATASET}.json \
    --epochs 25 $TRAIN_COMMON $BEST --exp_id $EXP_ID
done

echo "=== GCD-hier: aves_full ==="
python -m methods.gcd_hier_training.gcd_hier_training \
  --dataset_name aves_full --hier_split_type 5050 \
  --coarse_labels_path data/coarse_labels/aves_full.json \
  --epochs 25 $TRAIN_COMMON $BEST --exp_id $EXP_ID
