#!/bin/bash
# Build the geographic (SINR) feature sets used by the "w/ Geo" / SINR rows of the paper.
#
# Run from submission/:
#   SINR_NPZ=<kaggle>/PSSPLITS/sinr/sinr_obs_embeddings.npz bash bash_scripts/build_geo_features.sh
#
# Produces, under $FEATURE_ROOT (default osr_novel_categories/extracted_features =
# config.feature_extract_dir), for each dataset in $DATASETS:
#   sinr_loc_<dataset>/           SINR only, 256-d          (paper row "SINR")
#   vit_dino_sinr_loc_<dataset>/  DINOv2 + SINR, 1024-d     (rows "DINOv2 SINR", "DINOv2, w/ Geo")
#   selex_sinr_loc_<dataset>/     SelEx + SINR, 1024-d      (rows "SelEx, w/ Geo"; only if the
#                                                            SelEx features exist, see below)
# Concatenated vectors are [image | SINR], L2-normalised.
#
# Prerequisites:
#   * DINOv2 features: python -m methods.clustering.extract_features --dataset <dataset> --model_name vit_dino
#     (writes $FEATURE_ROOT/vit_dino_<dataset>/).
#   * SelEx features (optional): train SelEx-hier (bash_scripts/train_selex_hier.sh) and extract
#     features with bash_scripts/estimate_k_trained_backbone.sh <dataset> ... selex_hier_ptl1_<dataset>,
#     which writes osr_novel_categories/extracted_features_trained/selex_hier_ptl1_<dataset>/vit_dino_<dataset>/.
#
# Environment variables:
#   SINR_NPZ           (required) released SINR embeddings
#   FEATURE_ROOT       output/input feature root   [osr_novel_categories/extracted_features]
#   DATASETS           datasets to process          ["aves_small aves_full"]
#   DATASET_ROOT       Kaggle PSSPLITS/ folder; if set, overrides the aves_* paths in config.py
#   WITH_SELEX         auto | yes | no              [auto: build selex_sinr_loc if SelEx features exist]
#   SELEX_DIR_PATTERN  SelEx feature folder, {dataset} is substituted
#                      [osr_novel_categories/extracted_features_trained/selex_hier_ptl1_{dataset}/vit_dino_{dataset}]
#   ON_EXISTING        error | skip | overwrite      [error]
#   PYTHON             python executable            [python]
#
# Then run K estimation with --model_name sinr_loc | vit_dino_sinr_loc | selex_sinr_loc
# and --root_dir $FEATURE_ROOT (see the README, section "Geographic (SINR) features").
set -euo pipefail

if [ ! -f config.py ] || [ ! -d scripts ]; then
    echo "ERROR: run this script from the submission/ folder." >&2
    exit 1
fi
if [ -z "${SINR_NPZ:-}" ]; then
    echo "ERROR: set SINR_NPZ to PSSPLITS/sinr/sinr_obs_embeddings.npz from the Kaggle dataset." >&2
    exit 1
fi

PYTHON=${PYTHON:-python}
FEATURE_ROOT=${FEATURE_ROOT:-osr_novel_categories/extracted_features}
DATASETS=${DATASETS:-"aves_small aves_full"}
WITH_SELEX=${WITH_SELEX:-auto}
SELEX_DIR_PATTERN=${SELEX_DIR_PATTERN:-"osr_novel_categories/extracted_features_trained/selex_hier_ptl1_{dataset}/vit_dino_{dataset}"}
ON_EXISTING=${ON_EXISTING:-error}

ROOT_ARGS=()
if [ -n "${DATASET_ROOT:-}" ]; then
    ROOT_ARGS=(--dataset_root "$DATASET_ROOT")
fi

count_npy() { if [ -d "$1" ]; then find -L "$1" -name "*.npy" | wc -l; else echo 0; fi; }

for DATASET in $DATASETS; do
    SINR_DIR="$FEATURE_ROOT/sinr_loc_${DATASET}"
    DINO_DIR="$FEATURE_ROOT/vit_dino_${DATASET}"
    SELEX_DIR="${SELEX_DIR_PATTERN//\{dataset\}/$DATASET}"

    echo "======================================================================"
    echo "  $DATASET: SINR features -> $SINR_DIR"
    echo "======================================================================"
    "$PYTHON" scripts/fanout_sinr_features.py \
        --sinr_npz "$SINR_NPZ" \
        --dataset_name "$DATASET" \
        --feature_root "$FEATURE_ROOT" \
        --model_name sinr_loc \
        --on_existing "$ON_EXISTING" \
        ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}

    echo "======================================================================"
    echo "  $DATASET: DINOv2 + SINR -> $FEATURE_ROOT/vit_dino_sinr_loc_${DATASET}"
    echo "======================================================================"
    if [ "$(count_npy "$DINO_DIR")" = "0" ]; then
        echo "ERROR: no DINOv2 features in $DINO_DIR. Run first:" >&2
        echo "  python -m methods.clustering.extract_features --dataset $DATASET --model_name vit_dino --root_dir $FEATURE_ROOT" >&2
        exit 1
    fi
    "$PYTHON" scripts/concat_features.py \
        --feat_dir_a "$DINO_DIR" \
        --feat_dir_b "$SINR_DIR" \
        --output_dir "$FEATURE_ROOT/vit_dino_sinr_loc_${DATASET}" \
        --on_existing "$ON_EXISTING"

    if [ "$WITH_SELEX" = "no" ]; then
        echo "  [selex_sinr_loc_${DATASET}] not built (WITH_SELEX=no)"
        continue
    fi
    if [ "$(count_npy "$SELEX_DIR")" = "0" ]; then
        if [ "$WITH_SELEX" = "yes" ]; then
            echo "ERROR: no SelEx features in $SELEX_DIR (WITH_SELEX=yes)." >&2
            exit 1
        fi
        echo "  *** SKIPPED selex_sinr_loc_${DATASET}: no SelEx features in $SELEX_DIR."
        echo "  *** Train SelEx-hier and extract its features first (see header of this script)."
        continue
    fi
    echo "======================================================================"
    echo "  $DATASET: SelEx + SINR -> $FEATURE_ROOT/selex_sinr_loc_${DATASET}"
    echo "======================================================================"
    "$PYTHON" scripts/concat_features.py \
        --feat_dir_a "$SELEX_DIR" \
        --feat_dir_b "$SINR_DIR" \
        --output_dir "$FEATURE_ROOT/selex_sinr_loc_${DATASET}" \
        --on_existing "$ON_EXISTING"
done

echo ""
echo "Done. Feature folders under $FEATURE_ROOT:"
for DATASET in $DATASETS; do
    for M in sinr_loc vit_dino_sinr_loc selex_sinr_loc; do
        D="$FEATURE_ROOT/${M}_${DATASET}"
        if [ -d "$D" ]; then echo "  $D  ($(count_npy "$D") files)"; fi
    done
done
