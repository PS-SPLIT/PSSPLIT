# PS-SPLIT: Predicting Species Splits

Code for **"Predicting Species Splits: A Challenging Fine-Grained Benchmark for Category Discovery"**
(Christian Lange, Joakim Bruslund Haurum, Bingchen Zhao, Oisin Mac Aodha; NeurIPS 2026, Evaluations & Datasets track).

- **Dataset** (images, metadata, splits): https://www.kaggle.com/datasets/pssplits/ps-splits
- **Paper**: <!-- AUTHOR TODO: OpenReview / arXiv link -->

PS-SPLIT is a benchmark for **discovery by category splitting**: given a set of coarse categories,
decide which ones should be split into finer categories, how many, and which images belong to each.
It is built from 257 real, expert-verified taxonomic splits of bird species on iNaturalist, each paired
with a closely related control species that was never split.

| | PS-SPLIT (full) | PS-SPLIT-Small |
|---|---|---|
| Images | 79,061 | 25,285 |
| Fine classes (species) | 771 | 192 |
| Coarse classes | 514 = 257 splitting + 257 control | 128 = 64 + 64 |
| Known / novel fine classes (`5050` split) | 384 / 387 | 96 / 96 |

A *splitting* coarse class is the species before the split, and its fine classes are the species it was
split into. A *control* coarse class was never split, so its single fine class equals the coarse class.

The repository also provides coarse-respecting multi-way splits for CUB-200-2011 (coarse = bird family),
Stanford Cars (car make) and FGVC Aircraft (aircraft family), with up to 30 / 22 / 8 fine classes per
coarse class.

## Repository layout

```
.
├── LICENSE, THIRD_PARTY_NOTICES.md, LICENSES/   # MIT for the code; upstream licences of adapted code
├── CITATION.cff
└── submission/                         # run everything from here
    ├── config.py                       # paths to data, splits and pretrained backbones (edit this)
    ├── run_pipeline_hier.py            # end-to-end pipeline: features -> per-coarse K -> clustering/eval
    ├── requirements.txt, environment.yml
    ├── data/
    │   ├── coarse_labels/*.json        # fine-class -> coarse-class maps (aves_small, aves_full, cub, scars, aircraft)
    │   ├── generate_*.py               # rebuild every split pickle from the CSVs (byte-identical to Kaggle)
    │   └── ...                         # dataset classes
    ├── methods/
    │   ├── clustering/                 # feature extraction, semi-supervised k-means / GMM
    │   ├── estimate_k_hier/            # per-coarse K estimation (tuning, Brent, Set Transformer, baselines)
    │   ├── {gcd,simgcd,uno,contrastive}_hier_training/   # the four hierarchical backbones
    │   └── ...                         # standard (non-hierarchical) GCD methods
    ├── scripts/                        # geographic (SINR) features and feature concatenation
    ├── bash_scripts/                   # training and K-estimation drivers
    └── dataset_construction/           # how PS-SPLIT was built from iNaturalist (scrape -> filter -> crop)
```

## 1. Installation

Tested on Linux x86_64 with an NVIDIA GPU (CUDA 11.7 wheels), Python 3.10.

```bash
conda env create -f submission/environment.yml && conda activate pssplit
# or, in any Python 3.10 environment:
pip install -r submission/requirements.txt
```

`xformers==0.0.19` is required (the DINOv2 ViT imports it) and is the release that matches `torch==2.0.0`.

## 2. Data

### 2.1 PS-SPLIT (Kaggle)

```python
import kagglehub
root = kagglehub.dataset_download("pssplits/ps-splits")   # contains PSSPLITS/
```

```
PSSPLITS/
├── aves_photos_cropped/{taxon_id}/{observation_id}.jpg   # image crops, 771 species
├── aves_metadata_full/metadata_{taxon_id}_updated.csv     # per-image metadata (PS-SPLIT full)
├── aves_metadata_small/metadata_{taxon_id}_updated.csv    # per-image metadata (PS-SPLIT-Small)
├── aves_splits_full.csv, aves_splits_small.csv            # one row per coarse class
├── ssb_splits/*.pkl                                       # known/novel class splits read by the code
├── coarse_labels/*.json                                   # same files as submission/data/coarse_labels/
├── taxonomy/taxonomy_{full,small}.json                    # order/family/genus/species tree with image counts
├── sinr/                                                  # precomputed SINR location embeddings
└── known_unknown_classes/*.json                           # human-readable copies of the splits
```

Croissant metadata with the Responsible-AI fields is in `croissant_rai.json` at the root of this repository.

### 2.2 Point `config.py` at the data

Edit `submission/config.py` (replace every `<PATH_TO_...>` you need). For PS-SPLIT, with `R` the Kaggle
download root:

| `config.py` variable | value |
|---|---|
| `aves_small_splits_csv` | `R/PSSPLITS/aves_splits_small.csv` |
| `aves_small_metadata_dir` | `R/PSSPLITS/aves_metadata_small` |
| `aves_small_images_dir` | `R/PSSPLITS/aves_photos_cropped` |
| `aves_full_splits_csv` | `R/PSSPLITS/aves_splits_full.csv` |
| `aves_full_metadata_dir` | `R/PSSPLITS/aves_metadata_full` |
| `aves_full_images_dir` | `R/PSSPLITS/aves_photos_cropped` |
| `osr_split_dir` | `R/PSSPLITS/ssb_splits` |
| `dino_pretrain_path2` | DINOv2 ViT-B/14 with registers: [`dinov2_vitb14_reg4_pretrain.pth`](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) |
| `dino_v1_vitb16_path` | DINO ViT-B/16 (only for the DINOv1 rows): `dino_vitbase16_pretrain.pth` from [facebookresearch/dino](https://github.com/facebookresearch/dino) |

### 2.3 Split files

| Dataset (`--dataset_name`) | Split file in `osr_split_dir` | Selected by |
|---|---|---|
| `aves_small` | `aves_small_5050.pkl` | `--hier_split_type 5050` |
| `aves_full` | `aves_full_5050.pkl` (`aves_full_all_known.pkl` for feature extraction over all classes) | `--hier_split_type 5050` |
| `cub` / `scars` / `aircraft` | `cub_hier_5050.pkl` / `scars_hier_5050.pkl` / `aircraft_hier_5050.pkl` | `--hier_split_type 5050` |
| `aves_*` standard GCD variants | `aves_{small,full}_gcd_{paired,single}.pkl` | see `data/get_datasets.py` |

Note the names: the bird splits are `aves_{small,full}_5050.pkl` (no `hier`); only the FGVC splits are
called `*_hier_5050.pkl`. All split pickles can be regenerated from the CSVs in
`dataset_construction/data/` and the coarse maps (deterministic; byte-identical to the Kaggle files):

```bash
cd submission
python data/generate_aves_small_splits.py
python data/generate_aves_full_splits.py
python data/generate_aves_full_5050.py          # after the two above
python data/generate_aves_std_splits.py
python data/generate_hier_splits.py --dataset cub --split_type 5050   # also scars, aircraft
```

### 2.4 CUB-200-2011, Stanford Cars, FGVC Aircraft (multi-way splits)

Only the coarse-label maps and splits are released; get the images from the original datasets under
their own terms and set `cub_root`, `car_root` and `aircraft_root` in `config.py`.

- CUB-200-2011: `cub_root` is the folder containing `CUB_200_2011/`. Fine index = `class_id - 1`.
- FGVC Aircraft: `aircraft_root` is `.../fgvc-aircraft-2013b`. Fine index = alphabetical order of variant names.
- Stanford Cars: `car_root = '.../cars_{}/'`; the loader (inherited from SelEx) expects `cars_train/`,
  `cars_test/`, `cars_annos.mat`, `cars_train.csv` and `cars_test.csv` in the same folder. Fine index = `class - 1`.
  <!-- AUTHOR TODO (Q6): say where cars_annos.mat / cars_{train,test}.csv come from (the official site is offline). -->

## 3. Quickstart (PS-SPLIT-Small, frozen DINOv2)

```bash
cd submission

# 1. Extract DINOv2 features (writes osr_novel_categories/extracted_features/vit_dino_aves_small/)
python -m methods.clustering.extract_features --dataset aves_small --model_name vit_dino

# 2. Per-coarse K estimation + clustering + metrics for one method (here Brent + KMeans)
python -m methods.estimate_k_hier.estimate_k_hier_brent \
    --dataset_name aves_small --coarse_labels_path data/coarse_labels/aves_small.json \
    --hier_split_type 5050 --model_name vit_dino --prop_train_labels 1.0 \
    --cluster_methods kmeans,agglo,spectral,gmm_tied \
    --cluster_method kmeans --n_probe_total 4 --n_probe_related 2 \
    --k_results_path osr_novel_categories/k_estimation_results/aves_small_dinov2/brent_hier_kmeans_k.json
```

Each K-estimation script writes `<method>_k.json` (estimated K per novel coarse class) and
`<method>_k_metrics.json` (Cl. Acc., AMI, K Acc., ΔΣK, Split Prec. per coarse class and overall), and
prints `K estimation complete: <path>`. `bash_scripts/estimate_k_trained_backbone.sh` runs all K
estimators (tuning, Brent, Set Transformer) for one feature set.

`run_pipeline_hier.py` chains feature extraction → per-coarse K estimation → optional training →
re-extraction → evaluation; see `--help`.

## 4. Training the hierarchical backbones

```bash
bash bash_scripts/train_gcd_hier.sh      # GCD
bash bash_scripts/train_simgcd_hier.sh   # SimGCD
bash bash_scripts/train_uno_hier.sh      # UNO+
bash bash_scripts/train_selex_hier.sh    # SelEx
```

These train each backbone on `aves_small`, `aves_full`, `cub`, `scars` and `aircraft`
(seed 1, 25 epochs, `--prop_train_labels 1.0`). Checkpoints land in
`osr_novel_categories/<method>/dinov2/<dataset>/log/<exp_id>/checkpoints/model.pt`; run
`bash_scripts/estimate_k_trained_backbone.sh <dataset> data/coarse_labels/<dataset>.json <model.pt> <label>`
to evaluate them.

## 5. Geographic (SINR) features

Location is only used as an extra, frozen feature: each observation's coordinates are embedded with a SINR
location encoder (Cole et al., ICML 2023) and the 256-d embedding is concatenated with the image features
before K estimation and clustering. The backbones are trained on images only. The embeddings are released,
so SINR itself is not needed: `PSSPLITS/sinr/sinr_obs_embeddings.npz` has keys `observation_id` (int64) and
`emb` (float32, N×256, not normalised), one row per iNaturalist observation in PS-SPLIT. See
`PSSPLITS/sinr/README.md` for how they were computed.

| Paper rows | feature set (`--model_name`) | contents |
|---|---|---|
| "SINR" | `sinr_loc` | SINR only, 256-d |
| "DINOv2 SINR", "DINOv2, w/ Geo" | `vit_dino_sinr_loc` | [DINOv2 \| SINR], 1024-d, L2-normalised |
| "SelEx, w/ Geo" | `selex_sinr_loc` | [SelEx-hier \| SINR], 1024-d, L2-normalised |

```bash
cd submission
R=<kaggle download root>          # contains PSSPLITS/
# 1. DINOv2 image features (skip if already done)
python -m methods.clustering.extract_features --dataset aves_small --model_name vit_dino
python -m methods.clustering.extract_features --dataset aves_full  --model_name vit_dino
# 2. (only for "SelEx, w/ Geo") train SelEx-hier and extract its features, e.g.
#    bash bash_scripts/train_selex_hier.sh
#    bash bash_scripts/estimate_k_trained_backbone.sh aves_small data/coarse_labels/aves_small.json <selex model.pt> selex_hier_ptl1_aves_small
# 3. SINR, DINOv2+SINR and (if step 2 was run) SelEx+SINR features
SINR_NPZ=$R/PSSPLITS/sinr/sinr_obs_embeddings.npz bash bash_scripts/build_geo_features.sh
#    -> osr_novel_categories/extracted_features/{sinr_loc,vit_dino_sinr_loc,selex_sinr_loc}_aves_{small,full}/
# 4. K estimation / clustering: any command from Section 3 with --model_name changed, e.g.
python -m methods.estimate_k_hier.estimate_k_hier_brent \
    --dataset_name aves_small --coarse_labels_path data/coarse_labels/aves_small.json \
    --hier_split_type 5050 --model_name vit_dino_sinr_loc --prop_train_labels 1.0 \
    --cluster_methods kmeans,agglo,spectral,gmm_tied \
    --cluster_method kmeans --n_probe_total 4 --n_probe_related 2 \
    --k_results_path osr_novel_categories/k_estimation_results/aves_small_vit_dino_sinr_loc/brent_hier_kmeans_k.json
```

The individual steps are `scripts/fanout_sinr_features.py` (npz → one L2-normalised vector per sample, in the
same `{train,test}/{class_idx}/{uq_idx}.npy` layout as the image features) and `scripts/concat_features.py`
(concatenate two feature folders); both take `--help`. Instead of editing `config.py` you can pass
`--dataset_root $R/PSSPLITS` (or `DATASET_ROOT=$R/PSSPLITS` for the bash script). The regenerated `sinr_loc`
features are bit-identical to the ones used in the paper. `scripts/compute_sinr_embeddings.py` documents how
the npz was made (SINR ResidualFCNet, 14-d multi-scale sinusoidal encoding of (lon/180, lat/90), 256-d
penultimate embedding) and can embed locations with another SINR checkpoint.

## 6. Licence

- **Code**: MIT (`LICENSE`). Code adapted from GCD, SimGCD, UNO, SelEx (MIT) and DINO / DINOv2
  (Apache-2.0) keeps its upstream licence; see `THIRD_PARTY_NOTICES.md`.
- **PS-SPLIT annotations, metadata, splits and coarse maps**: CC BY-NC 4.0, including the annotation files
  in this repository (`submission/dataset_construction/data/*.csv`, `submission/data/coarse_labels/*.json`).

PS-SPLIT is intended for benchmarking category discovery. It is not a source of taxonomic decisions.

## 7. Citation

```bibtex
@inproceedings{lange2026pssplit,
  title     = {Predicting Species Splits: A Challenging Fine-Grained Benchmark for Category Discovery},
  author    = {Lange, Christian and Haurum, Joakim Bruslund and Zhao, Bingchen and Mac Aodha, Oisin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```
