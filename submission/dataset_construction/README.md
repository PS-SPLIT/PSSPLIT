# PS-SPLIT Dataset Construction

This folder contains the code and reference data needed to construct
**PS-SPLIT** — a fine-grained category-discovery benchmark built from real
taxonomic splits documented on iNaturalist, restricted to class Aves (birds).

This is a **source-only release**: pipeline-produced intermediate files
(scraped taxonomic splits, filtered Aves splits, matched controls) are not
shipped — running the pipeline from step 1 reproduces them.

## What gets produced

After running the pipeline end-to-end you will have:

- A `photos/` directory of original iNaturalist photos (one per observation)
- A `crops/` directory of MegaDetector-cropped 256×256 images that pass the
  quality filters
- A per-species metadata directory (one CSV per species with provenance, MegaDetector
  results, and VLM accept/reject decisions)
- Final CSVs `aves_splits_full.csv` and `aves_splits_small.csv` describing the
  splitting pairs and their matched non-splitting controls, with taxonomy and
  pose information.  Reference copies of these CSVs are included in `data/`
  so you can verify that your reproduction matches the published dataset.

The dataset itself (final shape) is summarised in the paper:
- **Full:** 79,061 images, 771 species, 257 splits + 257 controls
- **Small:** 25,285 images, 192 species, 64 splits + 64 controls

## Pipeline overview

| Step | Script | Inputs | Outputs |
|---|---|---|---|
| 1 | `scrape_taxon_changes.py` | iNaturalist taxonomic-changes pages | `raw_inat_splits.json` |
| 2 | `filter_aves_splits.py` | `raw_inat_splits.json`, `data/2025_splits_dataset.csv` | `aves_splits.json` |
| 3 | `find_aves_controls.py` | `aves_splits.json`, `data/2025_splits_dataset.csv` | `aves_controls.json`, `aves_controls_metadata.csv` |
| 4 | `dedup_splits.py` | `aves_splits.json`, `aves_controls.json` | de-duplicated versions of the above |
| 5 | `download_aves_metadata_photos_local_vlm.py` | `aves_splits.json`, `aves_controls.json` | `photos/`, `crops/`, `metadata/*.csv` |
| 6 | `build_aves_dataset.py` | `aves_splits.json`, `aves_controls.json`, `metadata/` | `aves_splits_full.csv`, `aves_splits_small.csv`, filtered metadata |

Step 5 is the bulk of the work: for each splitting pair plus its matched control,
it iteratively downloads observations from the iNaturalist API, runs each photo
through MegaDetector v5 (`md_pass`, `md_conf`, `md_bbox_px`, `md_crop_path`),
sends the cropped image through a locally-hosted **Qwen2.5-VL-7B-Instruct** for
seven binary quality checks plus a coarse pose label (`vlm_pass`, `vlm_pose`),
and continues until enough images of the dominant pose pass for both species
in the split.

## Reference data shipped in `data/`

Only files that are NOT produced by this pipeline are shipped here:

- `2025_splits_dataset.csv` — a reference table from a prior multi-taxon
  release.  Step 2 uses it to identify Aves splits without an extra API call.
- `aves_splits_full.csv`, `aves_splits_small.csv` — the final dataset
  description from the paper, included for verification only.  Your run of
  step 6 should reproduce these (modulo any iNaturalist taxonomy or
  observation-count drift since the original snapshot).

Step 1–4 outputs (`raw_inat_splits.json`, `aves_splits.json`,
`aves_controls.json`, `aves_controls_metadata.csv`) are deliberately not
shipped — running the pipeline from step 1 produces them.

## Running the pipeline

### Set up the environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You will also need:

- A local GPU with enough memory to run Qwen2.5-VL-7B-Instruct (≥ 24 GB
  recommended; the 72B-AWQ checkpoint is supported via `--model_id`).
- Internet access to the iNaturalist API (`api.inaturalist.org`) for steps 1, 3,
  and 5.

### End-to-end reproduction (steps 1–6)

The commands below assume your working directory is `dataset_construction/`
and that intermediate files are written to `dataset_construction/work/`.

```bash
mkdir -p work

# Step 1: scrape iNaturalist taxon-changes (slow, rate-limited; ~hours).
python scripts/scrape_taxon_changes.py
# Move the resulting raw_inat_splits.json into work/.

# Step 2: filter to Aves splits
python scripts/filter_aves_splits.py \
    --raw_json     work/raw_inat_splits.json \
    --existing_csv data/2025_splits_dataset.csv \
    --output       work/aves_splits.json \
    --min_obs      20 \
    --cache        work/taxa_cache.json

# Step 3: find taxonomically close controls
python scripts/find_aves_controls.py \
    --aves_splits  work/aves_splits.json \
    --existing_csv data/2025_splits_dataset.csv \
    --output_json  work/aves_controls.json \
    --taxa_cache   work/taxa_cache.json

# Step 4: deduplicate splits that share a generated species (keep newest)
python scripts/dedup_splits.py --apply

# Step 5: download photos, run MegaDetector, run local Qwen2.5-VL filter
python scripts/download_aves_metadata_photos_local_vlm.py \
    --aves_splits   work/aves_splits.json \
    --aves_controls work/aves_controls.json \
    --photo_dir     work/photos \
    --crops_dir     work/crops \
    --metadata_dir  work/metadata \
    --target_cap    100 \
    --model_id      Qwen/Qwen2.5-VL-7B-Instruct \
    --hf_cache_dir  ~/.cache/huggingface

# Step 6: assemble the final small-dataset CSV (and full version)
python scripts/build_aves_dataset.py \
    --aves_splits         work/aves_splits.json \
    --aves_controls       work/aves_controls.json \
    --metadata_dir        work/metadata \
    --small_metadata_dir  work/metadata_small \
    --output              work/aves_splits_small.csv
```

Each script supports `--help` for the full set of options.  Photos are never
deleted by the pipeline; metadata CSVs carry per-image accept/reject decisions
so any step can be resumed after interruption.

## Compute and time

The end-to-end pipeline is dominated by step 5.  On a single A6000 GPU with a
fast network connection, processing all 514 coarse classes (257 splits +
257 controls) takes roughly 1–2 days, most of which is waiting for the
iNaturalist API and running Qwen2.5-VL on the MegaDetector crops.  Steps 1–4
together complete in well under an hour.

## Citing the dataset

If you use PS-SPLIT or this construction pipeline, please cite the accompanying
paper.

## Notes

- All included observations are research-grade iNaturalist records under
  licences that permit redistribution for research.
- The pipeline is deterministic given the same iNat snapshot, but iNaturalist
  taxonomy and observation counts change over time, so re-running step 1 in
  the future may produce slightly different splits and image counts than the
  reference CSVs in `data/`.
- See the paper appendix for the full quality-filter prompt template and a
  list of removed splits (visual-difference probe < 60% accuracy).
