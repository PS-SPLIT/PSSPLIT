"""
find_aves_controls.py

Step 2 of the birds-only iNat dataset pipeline.

For each qualifying Aves split in aves_splits.json that does NOT already have a
non-splitting control species in the existing 2025_splits_dataset.csv, finds a
taxonomically close Aves species with:
  - No recorded taxon changes (taxon_changes_count == 0)
  - Active status
  - >= total_obs of the split's generated species (enough images available)
  - Not already used as a control or as a generated/source species in any split

Starts from the closest common ancestor and expands outwards until a match is found.

Writes aves_controls.json: dict mapping split_id -> control taxon info.
Also writes aves_controls_metadata.csv for downstream metadata download.

Usage:
    python find_aves_controls.py \
        --aves_splits aves_splits.json \
        --existing_csv 2025_splits_dataset.csv \
        --output_json aves_controls.json \
        --taxa_cache taxa_cache.json
"""

import json
import time
import argparse
import csv
import requests
import ast
import os

AVES_TAXON_ID = 3
SPECIES_RANK_LEVEL = 10


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_taxa_by_ancestor(ancestor_id, per_page=200, max_pages=5, sleep=1.2):
    """Fetch all active species-level taxa under a given ancestor from iNat API."""
    url = "https://api.inaturalist.org/v1/taxa"
    results = []
    for page in range(1, max_pages + 1):
        params = {
            "taxon_id": ancestor_id,
            "is_active": "true",
            "rank_level": SPECIES_RANK_LEVEL,
            "per_page": per_page,
            "page": page,
            "order_by": "observations_count",
            "order": "desc",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            page_results = data.get('results', [])
            results.extend(page_results)
            if len(results) >= data.get('total_results', 0) or not page_results:
                break
        except Exception as e:
            print(f"    API error for ancestor {ancestor_id} page {page}: {e}")
            break
        time.sleep(sleep)
    return results


def ancestors_of(taxon_id, cache):
    """Return ancestor_ids list for a taxon from cache (or empty list)."""
    t = cache.get(str(taxon_id), {})
    return t.get('ancestor_ids', [])


def find_control_for_split(split, used_ids, candidate_taxon_ids_to_exclude, cache,
                           min_total_obs=20, max_obs_per_species=100):
    """
    Walk up the ancestor chain of the first generated species, broadening at each step,
    until a suitable control species is found.

    The control must have at least as many observations as the total we'll actually
    download from the splitting species. We cap each species at max_obs_per_species
    so the control requirement doesn't balloon for species with huge raw obs counts.

    Returns a dict with control taxon info, or None if nothing found.
    """
    gen_species = split.get('generated_species', [])
    # Cap each species' contribution at max_obs_per_species to reflect the actual
    # download cap — otherwise a split with 2x800 raw obs would require a control
    # with 1600 obs when we'll only download 200 images from it.
    capped_total = sum(min(int(sp.get('observations', 0)), max_obs_per_species)
                       for sp in gen_species)
    min_obs_needed = max(min_total_obs, capped_total)

    # Get ancestor chain of first generated species
    first_taxon_id = gen_species[0]['taxon_id']
    ancestor_chain = ancestors_of(first_taxon_id, cache)
    # Remove the Aves root and above so we don't go too broad
    if AVES_TAXON_ID in ancestor_chain:
        aves_idx = ancestor_chain.index(AVES_TAXON_ID)
        ancestor_chain = ancestor_chain[aves_idx:]  # from Aves downward
    # Reverse so we go from most-specific ancestor upward
    ancestor_chain = list(reversed(ancestor_chain))

    all_generated_ids = {int(sp['taxon_id']) for sp in gen_species}

    for ancestor_id in ancestor_chain:
        candidates = get_taxa_by_ancestor(ancestor_id)
        for taxon in candidates:
            tid = taxon.get('id')
            if tid is None:
                continue
            if tid in used_ids:
                continue
            if tid in all_generated_ids:
                continue
            if tid in candidate_taxon_ids_to_exclude:
                continue
            if not taxon.get('is_active', False):
                continue
            if taxon.get('rank_level', 999) > SPECIES_RANK_LEVEL:
                continue
            if taxon.get('taxon_changes_count', 1) != 0:
                continue
            if taxon.get('observations_count', 0) < min_obs_needed:
                continue
            # Found a valid control
            return {
                'taxon_id': tid,
                'name': taxon.get('name', ''),
                'common_name': (taxon.get('preferred_common_name') or
                                taxon.get('english_common_name') or ''),
                'observations_count': taxon.get('observations_count', 0),
                'ancestor_ids': taxon.get('ancestor_ids', []),
                'iconic_taxon_name': taxon.get('iconic_taxon_name', 'Aves'),
            }
        # Not found at this level, try the next broader ancestor

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aves_splits', default='aves_splits.json')
    parser.add_argument('--existing_csv', default='2025_splits_dataset.csv')
    parser.add_argument('--output_json', default='aves_controls.json')
    parser.add_argument('--taxa_cache', default='taxa_cache.json')
    parser.add_argument('--output_csv', default='aves_controls_metadata.csv',
                        help='CSV listing (split_id, taxon_id) for control species')
    parser.add_argument('--max_obs_per_species', type=int, default=100,
                        help='Download cap per species; used to set control obs requirement')
    args = parser.parse_args()

    aves_splits = load_json(args.aves_splits)

    try:
        cache = load_json(args.taxa_cache)
    except FileNotFoundError:
        cache = {}

    # --- Build exclusion sets from existing CSV ---
    # In the existing CSV, control rows (should_split=False) use the control
    # taxon's own taxon_id as their split_id — NOT the paired split's iNat
    # split_id.  The link between a split row and its control is alignment_id.
    # We must group by alignment_id to correctly map iNat split_id → control.
    existing_control_by_split = {}   # iNat split_id -> taxon_id of existing control
    all_existing_taxon_ids = set()   # all taxon IDs in any role in existing CSV
    existing_aves_split_ids = set()

    try:
        with open(args.existing_csv, newline='', encoding='utf-8') as f:
            all_rows = list(csv.DictReader(f))

        # First pass: collect taxon IDs and group Aves rows by alignment_id
        from collections import defaultdict
        aves_by_align = defaultdict(list)
        for row in all_rows:
            # Track every taxon ID (for global exclusion)
            src_id = row.get('source_species_taxon_id', '')
            if src_id:
                try:
                    all_existing_taxon_ids.add(int(src_id))
                except ValueError:
                    pass
            gen_ids_str = row.get('generated_species_taxon_ids', '')
            try:
                gen_ids = ast.literal_eval(gen_ids_str)
                all_existing_taxon_ids.update(int(x) for x in gen_ids)
            except Exception:
                pass

            if row.get('iconic_taxon_name', '').strip() == 'Aves':
                aves_by_align[row.get('alignment_id', '')].append(row)

        # Second pass: within each alignment group, pair split↔control via alignment_id
        for align_id, pair in aves_by_align.items():
            split_row = next((r for r in pair if r.get('should_split', '') == 'True'), None)
            ctrl_row  = next((r for r in pair if r.get('should_split', '') == 'False'), None)
            if split_row:
                existing_aves_split_ids.add(str(split_row.get('split_id', '')).strip())
            if split_row and ctrl_row:
                inat_split_id = str(split_row.get('split_id', '')).strip()
                ctrl_src = ctrl_row.get('source_species_taxon_id', '')
                if inat_split_id and ctrl_src:
                    try:
                        existing_control_by_split[inat_split_id] = int(ctrl_src)
                    except ValueError:
                        pass

        print(f"Existing CSV: {len(existing_control_by_split)} Aves controls already assigned, "
              f"{len(existing_aves_split_ids)} Aves splits")
    except FileNotFoundError:
        print(f"Warning: {args.existing_csv} not found")

    # --- Load or init output (resume support) ---
    output_so_far = {}
    if os.path.exists(args.output_json):
        output_so_far = load_json(args.output_json)
        print(f"Resuming: {len(output_so_far)} controls already found")

    used_control_ids = set(output_so_far.get(sid, {}).get('taxon_id')
                           for sid in output_so_far if output_so_far[sid])
    used_control_ids.discard(None)

    # Also exclude all taxon IDs already in any split (generated or source)
    all_split_taxon_ids = set()
    for split in aves_splits.values():
        for sp in split.get('generated_species', []) + split.get('source_species', []):
            try:
                all_split_taxon_ids.add(int(sp['taxon_id']))
            except Exception:
                pass

    # --- Process each new split needing a control ---
    results = dict(output_so_far)
    new_found = 0
    for split_id, split in aves_splits.items():
        if split_id in results:
            continue  # already handled
        if split_id in existing_control_by_split:
            # Carry over existing control
            results[split_id] = {
                'taxon_id': existing_control_by_split[split_id],
                'source': 'existing_csv',
            }
            continue

        print(f"\nFinding control for split {split_id} …")
        gen_names = [sp['taxon_id'] for sp in split.get('generated_species', [])]
        print(f"  Generated species: {gen_names}")

        ctrl = find_control_for_split(
            split=split,
            used_ids=used_control_ids | all_existing_taxon_ids,
            candidate_taxon_ids_to_exclude=all_split_taxon_ids,
            cache=cache,
            min_total_obs=20,
            max_obs_per_species=args.max_obs_per_species,
        )

        if ctrl:
            print(f"  Found: taxon_id={ctrl['taxon_id']}, name={ctrl['name']}, "
                  f"obs={ctrl['observations_count']}")
            results[split_id] = ctrl
            used_control_ids.add(ctrl['taxon_id'])
            new_found += 1
        else:
            print(f"  WARNING: no control found for split {split_id}")
            results[split_id] = None

        # Save incrementally
        save_json(results, args.output_json)

    print(f"\nDone. New controls found: {new_found}")
    print(f"Total: {len(results)} splits mapped, "
          f"{sum(1 for v in results.values() if v is None)} with no control found")

    # --- Write CSV summary ---
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split_id', 'control_taxon_id', 'name', 'observations_count', 'source'])
        for sid, ctrl in results.items():
            if ctrl is None:
                writer.writerow([sid, '', '', '', 'not_found'])
            elif ctrl.get('source') == 'existing_csv':
                tid = ctrl['taxon_id']
                writer.writerow([sid, tid, '', '', 'existing_csv'])
            else:
                writer.writerow([sid, ctrl['taxon_id'], ctrl.get('name', ''),
                                 ctrl.get('observations_count', ''), 'new'])
    print(f"Written CSV to {args.output_csv}")


if __name__ == '__main__':
    main()
