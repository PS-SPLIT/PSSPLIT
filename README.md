# Predicting Species Splits: A Challenging Fine-Grained Benchmark for Category Discovery

This folder contains the minimal source needed to (1) train the four
hierarchical (coarse-label) GCD backbones reported in the paper, and (2) run
all per-coarse K-estimation methods on the resulting features.

## Layout

```
.
├── config.py                          # paths to datasets, splits, and pretrained backbones
├── run_pipeline_hier.py               # end-to-end pipeline orchestrator
├── requirements.txt
├── environment.yml
├── data/                              # dataset classes, splits, augmentations
├── methods/
│   ├── clustering/                    # feature extraction + (semi-supervised) K-Means / GMM
│   ├── estimate_k/                    # global K methods (brent, agglo, spectral, x/g-means, ...)
│   ├── estimate_k_hier/               # per-coarse K methods (incl. Set Transformer ensemble)
│   ├── gcd_training/   simgcd_training/   uno_training/   contrastive_training/
│   ├── gcd_hier_training/             # GCD-hier
│   ├── simgcd_hier_training/          # SimGCD-hier
│   ├── uno_hier_training/             # UNO-hier
│   └── contrastive_hier_training/     # SelEx-hier (contrastive_coarse)
├── models/                            # ViT (DINO / DINOv2) + ResNet two-head
├── project_utils/                     # cluster + log helpers, schedulers, general utils
└── bash_scripts/
    ├── train_gcd_hier.sh
    ├── train_simgcd_hier.sh
    ├── train_uno_hier.sh
    ├── train_selex_hier.sh
    └── estimate_k_trained_backbone.sh
```

## Setup

1. Create the conda environment:

   ```
   conda env create -f environment.yml
   # or: pip install -r requirements.txt
   ```

2. Edit `config.py` and replace every `<PATH_TO_...>` placeholder with the
   absolute path to the corresponding dataset, SSB split directory, or
   pretrained DINO/DINOv2 checkpoint on your machine.

3. The dataset upload that accompanies this submission contains:
   - per-fine-class image folders for each dataset,
   - the `data/coarse_labels/*.json` files (fine-class → coarse-class maps), and
   - the coarse-respecting `*_hier_5050.pkl` split files (under
     `<PATH_TO_SSB_SPLITS>`).

   Place the coarse JSON files into `data/coarse_labels/` before running.

## Training the four hierarchical backbones

Each script trains the corresponding backbone on `aves_small`, `cub`, `scars`,
`aircraft`, and `aves_full` with seed=1 and the best hyper-parameters from the
ablation study. Run from this folder's root.

```
bash bash_scripts/train_gcd_hier.sh
bash bash_scripts/train_simgcd_hier.sh
bash bash_scripts/train_uno_hier.sh
bash bash_scripts/train_selex_hier.sh
```

Checkpoints land under `osr_novel_categories/<method>/dinov2/<dataset>/log/<exp_id>/checkpoints/model.pt`.

## Running K-estimation on a trained backbone

For any given backbone checkpoint, the following script extracts features and
runs all eight per-coarse K-estimation methods (3 tuning methods × {agglo,
spectral, xmeans} + 4 hierarchical Brent variants + Set Transformer ensemble):

```
bash bash_scripts/estimate_k_trained_backbone.sh \
    <dataset_name> \
    data/coarse_labels/<dataset>.json \
    <path/to/checkpoints/model.pt> \
    <label_for_output_dir>
```

K-estimation results land in
`osr_novel_categories/k_estimation_results/<label>/<method>_k.json`.

## End-to-end pipeline

`run_pipeline_hier.py` chains feature extraction → per-coarse K estimation →
optional training → re-extraction → evaluation in a single command. See
its `--help` for the full argument list.
