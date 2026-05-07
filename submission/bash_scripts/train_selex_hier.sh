#!/bin/bash
# Train the SelEx-hier backbone (contrastive_coarse) on each dataset
# (seed=1, prop_train_labels=1.0).
# SelEx fine-tunes 2 transformer blocks (--grad_from_block 10).
set -e
export CUDA_VISIBLE_DEVICES=0

TRAIN_COMMON="--prop_train_labels 1.0
  --lr 0.1 --weight_decay 5e-5 --seed 1
  --grad_from_block 10 --batch_size 128
  --sup_con_weight 0.35 --contrast_unlabel_only False
  --transform imagenet --eval_funcs v1 v2
  --train_report_interval 25
  --prototype_extraction_interval 1"

EXP_ID=selex_hier_seed1_25ep

echo "=== SelEx-hier: aves_small ==="
python -m methods.contrastive_hier_training.contrastive_coarse_training \
  --dataset_name aves_small --hier_split_type 5050 \
  --coarse_labels_path data/coarse_labels/aves_small.json \
  --epochs 25 $TRAIN_COMMON --exp_id $EXP_ID

for DATASET in cub scars aircraft; do
  echo "=== SelEx-hier: $DATASET ==="
  python -m methods.contrastive_hier_training.contrastive_coarse_training \
    --dataset_name $DATASET --hier_split_type 5050 \
    --coarse_labels_path data/coarse_labels/${DATASET}.json \
    --epochs 25 $TRAIN_COMMON --exp_id $EXP_ID
done

echo "=== SelEx-hier: aves_full ==="
python -m methods.contrastive_hier_training.contrastive_coarse_training \
  --dataset_name aves_full --hier_split_type 5050 \
  --coarse_labels_path data/coarse_labels/aves_full.json \
  --epochs 25 $TRAIN_COMMON --exp_id $EXP_ID
