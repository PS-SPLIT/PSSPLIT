#!/bin/bash
# Run all per-coarse K estimation methods on a given trained backbone.
#
# Usage:
#   bash bash_scripts/estimate_k_trained_backbone.sh \
#       <dataset_name> <coarse_labels_json> <checkpoint_path> <label>
#
# Example:
#   bash bash_scripts/estimate_k_trained_backbone.sh \
#       aves_small data/coarse_labels/aves_small.json \
#       osr_novel_categories/gcd_hier/dinov2/aves_small/log/<run_id>/checkpoints/model.pt \
#       gcd_hier_aves_small
#
# Runs feature extraction with the given checkpoint, then evaluates 8 K methods:
#   tuning_agglo, tuning_spectral, tuning_xmeans,
#   brent_hier_kmeans, brent_hier_agglo, brent_hier_spectral, brent_hier_gmm_tied,
#   transformer (Set Transformer ensemble).

set -e
export CUDA_VISIBLE_DEVICES=0

DATASET=$1
COARSE_JSON=$2
CKPT=$3
LABEL=$4

if [ -z "$DATASET" ] || [ -z "$COARSE_JSON" ] || [ -z "$CKPT" ] || [ -z "$LABEL" ]; then
    echo "Usage: $0 <dataset_name> <coarse_labels_json> <checkpoint_path> <label>"
    exit 1
fi

CLUSTER_METHODS=kmeans,agglo,spectral,gmm_tied
FEAT_BASE=osr_novel_categories/extracted_features_trained
K_BASE=osr_novel_categories/k_estimation_results

FEAT_ROOT="$FEAT_BASE/$LABEL"
K_DIR="$K_BASE/$LABEL"
mkdir -p "$K_DIR"

PASS=0; FAIL=0; FAILED=""

run() {
    local name=$1; shift
    echo ""
    echo "=================================================================="
    echo "  $name"
    echo "=================================================================="
    if "$@"; then
        PASS=$((PASS + 1))
        echo "  --> OK: $name"
    else
        FAIL=$((FAIL + 1))
        FAILED="$FAILED\n    $name"
        echo "  --> FAILED: $name (continuing)"
    fi
}

# Step 1: extract features from the trained backbone (last checkpoint).
run "extract_features [$LABEL]" \
    python -m methods.clustering.extract_features \
        --dataset           $DATASET \
        --model_name        vit_dino \
        --warmup_model_dir  "$CKPT" \
        --use_best_model    False \
        --root_dir          "$FEAT_ROOT" \
        --batch_size        256 \
        --num_workers       4

COMMON="--dataset_name $DATASET
  --coarse_labels_path $COARSE_JSON
  --hier_split_type 5050
  --model_name vit_dino
  --batch_size 256
  --num_workers 4
  --prop_train_labels 1.0
  --cluster_methods $CLUSTER_METHODS
  --root_dir $FEAT_ROOT"

# Step 2: K estimation methods.
run "[$LABEL] tuning_agglo" \
    python -m methods.estimate_k_hier.estimate_k_hier_tuning $COMMON \
        --estimate_k_method agglo \
        --agglo_n_grid 25 --agglo_lo 0.0 --agglo_hi 2.0 \
        --k_results_path "$K_DIR/tuning_agglo_k.json"

run "[$LABEL] tuning_spectral" \
    python -m methods.estimate_k_hier.estimate_k_hier_tuning $COMMON \
        --estimate_k_method spectral \
        --spectral_gap_n_grid 30 --spectral_gap_lo 0.001 --spectral_gap_hi 1.0 \
        --spectral_n_neighbors 10 \
        --k_results_path "$K_DIR/tuning_spectral_k.json"

run "[$LABEL] tuning_xmeans" \
    python -m methods.estimate_k_hier.estimate_k_hier_tuning $COMMON \
        --estimate_k_method xmeans \
        --pca_dim_candidates 0,32,64,128 \
        --k_results_path "$K_DIR/tuning_xmeans_k.json"

run "[$LABEL] brent_hier_kmeans" \
    python -m methods.estimate_k_hier.estimate_k_hier_brent $COMMON \
        --cluster_method kmeans --n_probe_total 4 --n_probe_related 2 \
        --k_results_path "$K_DIR/brent_hier_kmeans_k.json"

run "[$LABEL] brent_hier_agglo" \
    python -m methods.estimate_k_hier.estimate_k_hier_brent $COMMON \
        --cluster_method agglo --n_probe_total 4 --n_probe_related 2 \
        --k_results_path "$K_DIR/brent_hier_agglo_k.json"

run "[$LABEL] brent_hier_spectral" \
    python -m methods.estimate_k_hier.estimate_k_hier_brent $COMMON \
        --cluster_method spectral --n_probe_total 4 --n_probe_related 2 \
        --k_results_path "$K_DIR/brent_hier_spectral_k.json"

run "[$LABEL] brent_hier_gmm_tied" \
    python -m methods.estimate_k_hier.estimate_k_hier_brent $COMMON \
        --cluster_method gmm_tied --n_probe_total 4 --n_probe_related 2 \
        --k_results_path "$K_DIR/brent_hier_gmm_tied_k.json"

run "[$LABEL] transformer" \
    python -m methods.estimate_k_hier.estimate_k_hier_transformer \
        --dataset_name $DATASET \
        --coarse_labels_path $COARSE_JSON \
        --hier_split_type 5050 \
        --prop_train_labels 1.0 \
        --model_name vit_dino \
        --batch_size 256 --num_workers 4 \
        --root_dir "$FEAT_ROOT" \
        --use_oracle_k False \
        --n_models 5 --epochs 50 \
        --cluster_methods $CLUSTER_METHODS \
        --k_results_path "$K_DIR/transformer_k.json"

echo ""
echo "=================================================================="
echo "  SUMMARY [$LABEL]"
echo "=================================================================="
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
if [ -n "$FAILED" ]; then
    printf "  Failed methods:%b\n" "$FAILED"
fi
echo ""
echo "  Features  : $FEAT_ROOT"
echo "  K results : $K_DIR"
echo "=================================================================="
