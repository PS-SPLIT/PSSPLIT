"""
filter_aves_splits.py

Step 1 of the birds-only iNat dataset pipeline.

Reads raw_inat_splits.json (2068 total splits across all taxa), applies filters
(two-way, min_obs=20, active, not redundant), then identifies which remaining splits
are Aves using:
  - The existing 2025_splits_dataset.csv (no API call needed for known splits)
  - iNat API /v1/taxa/{ids} for splits not in the existing CSV

Writes aves_splits.json containing all qualifying Aves splits.

Usage:
    python filter_aves_splits.py \
        --raw_json raw_inat_splits.json \
        --existing_csv 2025_splits_dataset.csv \
        --output aves_splits.json \
        --min_obs 20 \
        --cache taxa_cache.json
"""

import json
import time
import argparse
import requests
from datetime import datetime

AVES_ICONIC_TAXON_ID = 3  # iNaturalist taxon ID for Aves (class Birds)
BATCH_SIZE = 30           # max IDs per /v1/taxa/ request


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def apply_basic_filters(data, min_obs=20):
    """
    Non-API filters applied to all 2068 raw splits:
      1. Exactly 2 generated species (two-way only)
      2. All generated species are Active
      3. Each generated species has >= min_obs observations
    Returns filtered dict and sorted list for redundancy pass.
    """
    passed = {}
    for split_id, split in data.items():
        gen = split.get('generated_species', [])
        if len(gen) != 2:
            continue
        if not all(sp.get('status') == 'Active' for sp in gen):
            continue
        if any(int(sp.get('observations', 0)) < min_obs for sp in gen):
            continue
        passed[split_id] = split
    print(f"After basic filters: {len(passed)} / {len(data)} splits remain")
    return passed


def remove_redundant_splits(data):
    """
    Sort splits newest-first. A split is redundant if any of its generated species
    appears as a source species in a more-recent split.
    Track encountered source species as we process oldest-to-newest (i.e. in reverse
    of sorted order).
    """
    def parse_date(s):
        try:
            return datetime.strptime(s, '%Y-%m-%d')
        except Exception:
            return datetime.min

    sorted_items = sorted(data.items(), key=lambda kv: parse_date(kv[1]['date']), reverse=True)

    encountered_source_ids = set()
    kept = {}

    for split_id, split in sorted_items:
        gen_ids = {sp['taxon_id'] for sp in split.get('generated_species', [])}
        if gen_ids & encountered_source_ids:
            # a generated species of this older split was a source in a newer split
            continue
        kept[split_id] = split
        for sp in split.get('source_species', []):
            encountered_source_ids.add(sp['taxon_id'])

    print(f"After redundancy removal: {len(kept)} splits remain")
    return kept


def batch_fetch_taxa(taxon_ids, cache, sleep=1.2):
    """
    Batch-fetch taxon data from iNat API. Returns dict: taxon_id (str) -> taxon dict.
    Uses cache to avoid redundant calls. Batches up to BATCH_SIZE IDs per request.
    """
    to_fetch = [tid for tid in taxon_ids if tid not in cache]
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i:i + BATCH_SIZE]
        url = "https://api.inaturalist.org/v1/taxa/" + ",".join(str(t) for t in batch)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for taxon in resp.json().get('results', []):
                cache[str(taxon['id'])] = taxon
        except Exception as e:
            print(f"  API error for batch {batch}: {e}")
        time.sleep(sleep)
    return cache


def is_aves(taxon_dict):
    """Return True if the taxon is Aves (iconic_taxon_id==3 or iconic_taxon_name=='Aves')."""
    if taxon_dict.get('iconic_taxon_name') == 'Aves':
        return True
    if taxon_dict.get('iconic_taxon_id') == AVES_ICONIC_TAXON_ID:
        return True
    return False


def is_species_level(taxon_dict):
    """Return True if rank_level <= 10 (species or below)."""
    return taxon_dict.get('rank_level', 999) <= 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_json', default='raw_inat_splits.json')
    parser.add_argument('--existing_csv', default='2025_splits_dataset.csv')
    parser.add_argument('--output', default='aves_splits.json')
    parser.add_argument('--min_obs', type=int, default=20)
    parser.add_argument('--cache', default='taxa_cache.json',
                        help='JSON file to persist API taxon cache across runs')
    args = parser.parse_args()

    # --- Load raw splits ---
    print(f"Loading {args.raw_json} …")
    raw = load_json(args.raw_json)
    print(f"  {len(raw)} total raw splits")

    # --- Apply basic non-API filters ---
    filtered = apply_basic_filters(raw, min_obs=args.min_obs)
    filtered = remove_redundant_splits(filtered)

    # --- Load existing CSV to find already-labelled Aves splits ---
    aves_split_ids_in_csv = set()
    non_aves_split_ids_in_csv = set()
    try:
        import csv as csv_mod
        with open(args.existing_csv, newline='', encoding='utf-8') as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                sid = str(row.get('split_id', '')).strip()
                if not sid:
                    continue
                iconic = row.get('iconic_taxon_name', '').strip()
                should_split = row.get('should_split', '').strip()
                if should_split == 'True':
                    if iconic == 'Aves':
                        aves_split_ids_in_csv.add(sid)
                    elif iconic:
                        non_aves_split_ids_in_csv.add(sid)
        print(f"Existing CSV: {len(aves_split_ids_in_csv)} known Aves splits, "
              f"{len(non_aves_split_ids_in_csv)} known non-Aves splits")
    except FileNotFoundError:
        print(f"Warning: {args.existing_csv} not found; will rely entirely on API")

    # --- Separate into known-Aves, known-non-Aves, unknown ---
    known_aves = {}
    unknown = {}
    for split_id, split in filtered.items():
        if split_id in aves_split_ids_in_csv:
            known_aves[split_id] = split
        elif split_id not in non_aves_split_ids_in_csv:
            unknown[split_id] = split
        # else: known non-Aves, skip

    print(f"Known Aves: {len(known_aves)}, Unknown (need API): {len(unknown)}, "
          f"Known non-Aves (skipped): {len(filtered) - len(known_aves) - len(unknown)}")

    # --- Load or init cache ---
    try:
        cache = load_json(args.cache)
        print(f"Loaded {len(cache)} entries from {args.cache}")
    except FileNotFoundError:
        cache = {}

    # --- Collect taxon IDs needing API lookup ---
    unknown_taxon_ids = set()
    for split in unknown.values():
        for sp in split.get('generated_species', []):
            unknown_taxon_ids.add(sp['taxon_id'])

    print(f"Need to look up {len(unknown_taxon_ids)} taxon IDs (not in cache: "
          f"{sum(1 for t in unknown_taxon_ids if str(t) not in cache)})")

    if unknown_taxon_ids:
        cache = batch_fetch_taxa(list(unknown_taxon_ids), cache)
        save_json(cache, args.cache)
        print(f"Cache now has {len(cache)} entries, saved to {args.cache}")

    # --- Classify unknown splits ---
    newly_aves = {}
    for split_id, split in unknown.items():
        gen = split.get('generated_species', [])
        taxon_dicts = [cache.get(str(sp['taxon_id']), {}) for sp in gen]

        if not all(taxon_dicts):
            print(f"  Split {split_id}: missing API data, skipping")
            continue
        if not all(is_aves(t) for t in taxon_dicts):
            continue
        if not all(is_species_level(t) for t in taxon_dicts):
            print(f"  Split {split_id}: not species-level, skipping")
            continue

        newly_aves[split_id] = split

    print(f"Newly identified Aves splits: {len(newly_aves)}")

    # --- Combine known + newly found Aves splits ---
    all_aves = {**known_aves, **newly_aves}
    print(f"\nTotal qualifying Aves splits: {len(all_aves)}")

    # --- Enrich with iconic_taxon_name and ancestor_ids ---
    # Get ancestor_ids for each split from the API cache (use first generated species)
    for split_id, split in all_aves.items():
        if 'iconic_taxon_name' not in split:
            split['iconic_taxon_name'] = 'Aves'
        if 'ancestor_ids' not in split:
            gen = split.get('generated_species', [])
            for sp in gen:
                t = cache.get(str(sp['taxon_id']), {})
                if t.get('ancestor_ids'):
                    split['ancestor_ids'] = t['ancestor_ids']
                    break

    # --- Save ---
    save_json(all_aves, args.output)
    print(f"Written to {args.output}")

    # --- Summary ---
    print("\n--- Summary ---")
    obs_counts = []
    for split in all_aves.values():
        for sp in split.get('generated_species', []):
            obs_counts.append(int(sp.get('observations', 0)))
    if obs_counts:
        import statistics
        print(f"Observations per generated species: "
              f"min={min(obs_counts)}, median={statistics.median(obs_counts):.0f}, "
              f"max={max(obs_counts)}")


if __name__ == '__main__':
    main()
