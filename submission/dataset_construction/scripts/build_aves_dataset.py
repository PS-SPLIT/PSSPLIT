"""
build_aves_dataset.py

Assembles aves_splits_small.csv and filtered metadata for the small dataset snapshot.

Sources:
  - aves_splits.json          (qualifying Aves splits)
  - aves_controls.json        (non-splitting control species for each split)
  - aves_metadata/            (post-VLM metadata CSVs with dataset_pass column)

A split is included when both generated species have >= min_images_per_species
accepted images (dataset_pass=True). Control rows are included when they have > 0
accepted images; low-count controls are kept rather than discarding the whole split.

Output:
  - aves_splits_small.csv     (CSV with pose column and taxonomy text)
  - aves_metadata_small/      (filtered metadata: dataset_pass=True rows only)

Usage:
    python build_aves_dataset.py \\
        --aves_splits aves_splits.json \\
        --aves_controls aves_controls.json \\
        --metadata_dir aves_metadata \\
        --small_metadata_dir aves_metadata_small \\
        --output aves_splits_small.csv \\
        --taxa_cache taxa_cache.json \\
        --min_images_per_species 10
"""

import json
import os
import csv
import time
import argparse
import requests
from collections import Counter

AVES_ICONIC_NAME = 'Aves'

RANK_ORDER = 'order'
RANK_FAMILY = 'family'
RANK_GENUS = 'genus'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def fetch_taxon_batch(taxon_ids, cache, sleep=1.2, batch_size=30):
    """Batch-fetch taxon dicts from iNat API, filling cache in place."""
    to_fetch = [tid for tid in taxon_ids if str(tid) not in cache]
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        url = "https://api.inaturalist.org/v1/taxa/" + ",".join(str(t) for t in batch)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for t in resp.json().get('results', []):
                cache[str(t['id'])] = t
        except Exception as e:
            print(f"  Batch API error: {e}")
        time.sleep(sleep)
    return cache


def get_taxonomy_names(taxon_dict, ancestor_cache):
    """Return (order_name, family_name, genus_name) by walking the ancestor chain."""
    ancestor_ids = taxon_dict.get('ancestor_ids', [])
    order_name = family_name = genus_name = ''
    for aid in reversed(ancestor_ids):
        anc = ancestor_cache.get(str(aid), {})
        rank = anc.get('rank', '')
        name = anc.get('name', '')
        if rank == RANK_GENUS and not genus_name:
            genus_name = name
        elif rank == RANK_FAMILY and not family_name:
            family_name = name
        elif rank == RANK_ORDER and not order_name:
            order_name = name
        if order_name and family_name and genus_name:
            break
    return order_name, family_name, genus_name


def load_metadata_rows(taxon_id, metadata_dir):
    """Return list of row dicts from the metadata CSV, or [] if missing."""
    meta_file = os.path.join(metadata_dir, f'metadata_{taxon_id}_updated.csv')
    if not os.path.exists(meta_file):
        return []
    with open(meta_file, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def count_accepted(rows):
    """Count rows where dataset_pass=True."""
    return sum(1 for r in rows if r.get('dataset_pass', '').strip() == 'True')


def get_majority_pose(taxon_ids, metadata_dir):
    """Most common pose among dataset_pass=True rows across the given taxon IDs."""
    counter = Counter()
    for tid in taxon_ids:
        for r in load_metadata_rows(tid, metadata_dir):
            if r.get('dataset_pass', '').strip() == 'True':
                pose = r.get('vlm_pose', '').strip()
                if pose:
                    counter[pose] += 1
    return counter.most_common(1)[0][0] if counter else ''


def write_filtered_metadata(taxon_id, metadata_dir, small_metadata_dir):
    """Write dataset_pass=True rows to small_metadata_dir. Returns count written."""
    rows = load_metadata_rows(taxon_id, metadata_dir)
    accepted = [r for r in rows if r.get('dataset_pass', '').strip() == 'True']
    if not accepted:
        return 0
    os.makedirs(small_metadata_dir, exist_ok=True)
    out_path = os.path.join(small_metadata_dir, f'metadata_{taxon_id}_updated.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(accepted[0].keys()))
        writer.writeheader()
        writer.writerows(accepted)
    return len(accepted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aves_splits',        default='aves_splits.json')
    parser.add_argument('--aves_controls',       default='aves_controls.json')
    parser.add_argument('--metadata_dir',        default='aves_metadata')
    parser.add_argument('--small_metadata_dir',  default='aves_metadata_small')
    parser.add_argument('--output',              default='aves_splits_small.csv')
    parser.add_argument('--taxa_cache',          default='taxa_cache.json')
    parser.add_argument('--min_images_per_species', type=int, default=10,
                        help='Skip a split if any generated species has fewer accepted images')
    parser.add_argument('--split_ids', nargs='*', default=None,
                        help='Only include these split IDs (default: all qualifying splits)')
    args = parser.parse_args()

    aves_splits   = load_json(args.aves_splits)
    aves_controls = load_json(args.aves_controls)

    if args.split_ids is not None:
        keep = set(str(s) for s in args.split_ids)
        aves_splits = {k: v for k, v in aves_splits.items() if k in keep}

    try:
        cache = load_json(args.taxa_cache)
    except FileNotFoundError:
        cache = {}

    # Collect all taxon IDs to pre-fetch
    all_taxon_ids = set()
    for split in aves_splits.values():
        for sp in split.get('generated_species', []) + split.get('source_species', []):
            try:
                all_taxon_ids.add(int(sp['taxon_id']))
            except Exception:
                pass
    for ctrl in aves_controls.values():
        if ctrl and ctrl.get('taxon_id'):
            try:
                all_taxon_ids.add(int(ctrl['taxon_id']))
            except Exception:
                pass

    print(f"Fetching taxon data for {len(all_taxon_ids)} species …")
    cache = fetch_taxon_batch(list(all_taxon_ids), cache)

    ancestor_ids_needed = set()
    for tid in all_taxon_ids:
        ancestor_ids_needed.update(cache.get(str(tid), {}).get('ancestor_ids', []))
    print(f"Fetching {len(ancestor_ids_needed)} ancestor taxa for taxonomy text …")
    cache = fetch_taxon_batch(list(ancestor_ids_needed), cache, batch_size=30)
    save_json(cache, args.taxa_cache)
    print(f"Cache saved ({len(cache)} entries)\n")

    fieldnames = [
        'alignment_id',
        'split_id',
        'source_species_taxon_id',
        'generated_species_taxon_ids',
        'date',
        'source',
        'iconic_taxon_name',
        'ancestor_ids',
        'generated_species_counts',
        'set',
        'should_split',
        'pose',
        'order_name',
        'family_name',
        'genus_name',
        'common_name',
        'common_name_1',
        'common_name_2',
    ]

    out_rows = []
    alignment_id = 1
    skipped_splits = 0

    for split_id, split in aves_splits.items():
        gen_species   = split.get('generated_species', [])
        source_species = split.get('source_species', [])

        gen_ids    = [int(sp['taxon_id']) for sp in gen_species]
        gen_counts = [count_accepted(load_metadata_rows(tid, args.metadata_dir)) for tid in gen_ids]

        if any(c < args.min_images_per_species for c in gen_counts):
            skipped_splits += 1
            continue

        split_pose = get_majority_pose(gen_ids, args.metadata_dir)
        src_id     = source_species[0]['taxon_id'] if source_species else gen_ids[0]

        first_taxon  = cache.get(str(gen_ids[0]), {})
        second_taxon = cache.get(str(gen_ids[1]), {}) if len(gen_ids) > 1 else {}
        order_name, family_name, genus_name = get_taxonomy_names(first_taxon, cache)
        common_name   = (first_taxon.get('preferred_common_name') or
                         first_taxon.get('english_common_name') or '')
        common_name_1 = common_name
        common_name_2 = (second_taxon.get('preferred_common_name') or
                         second_taxon.get('english_common_name') or '')

        anc_ids = split.get('ancestor_ids', first_taxon.get('ancestor_ids', []))
        anc_str = ','.join(str(a) for a in anc_ids) if isinstance(anc_ids, list) else str(anc_ids)

        out_rows.append({
            'alignment_id':              alignment_id,
            'split_id':                  split_id,
            'source_species_taxon_id':   src_id,
            'generated_species_taxon_ids': json.dumps(gen_ids),
            'date':                      split.get('date', ''),
            'source':                    split.get('source', ''),
            'iconic_taxon_name':         AVES_ICONIC_NAME,
            'ancestor_ids':              anc_str,
            'generated_species_counts':  json.dumps(gen_counts),
            'set':                       'train',
            'should_split':              'True',
            'pose':                      split_pose,
            'order_name':                order_name,
            'family_name':               family_name,
            'genus_name':                genus_name,
            'common_name':               common_name,
            'common_name_1':             common_name_1,
            'common_name_2':             common_name_2,
        })

        for tid in gen_ids:
            n = write_filtered_metadata(tid, args.metadata_dir, args.small_metadata_dir)
            print(f"  gen  {tid}: {n} accepted rows → small metadata")

        # Control row: included if any accepted images exist (even if below min_images)
        ctrl = aves_controls.get(split_id)
        if ctrl and ctrl.get('taxon_id'):
            ctrl_id    = int(ctrl['taxon_id'])
            ctrl_count = count_accepted(load_metadata_rows(ctrl_id, args.metadata_dir))
            ctrl_pose  = get_majority_pose([ctrl_id], args.metadata_dir)

            ctrl_taxon = cache.get(str(ctrl_id), {})
            ctrl_order, ctrl_family, ctrl_genus = get_taxonomy_names(ctrl_taxon, cache)
            ctrl_common = (ctrl_taxon.get('preferred_common_name') or
                           ctrl_taxon.get('english_common_name') or
                           ctrl.get('common_name', ''))
            ctrl_anc_str = ','.join(str(a) for a in ctrl_taxon.get('ancestor_ids', []))

            if ctrl_count > 0:
                out_rows.append({
                    'alignment_id':              alignment_id,
                    'split_id':                  ctrl_id,
                    'source_species_taxon_id':   ctrl_id,
                    'generated_species_taxon_ids': json.dumps([ctrl_id]),
                    'date':                      '',
                    'source':                    '',
                    'iconic_taxon_name':         AVES_ICONIC_NAME,
                    'ancestor_ids':              ctrl_anc_str,
                    'generated_species_counts':  json.dumps([ctrl_count]),
                    'set':                       'train',
                    'should_split':              'False',
                    'pose':                      ctrl_pose,
                    'order_name':                ctrl_order,
                    'family_name':               ctrl_family,
                    'genus_name':                ctrl_genus,
                    'common_name':               ctrl_common,
                    'common_name_1':             ctrl_common,
                    'common_name_2':             '',
                })
                n = write_filtered_metadata(ctrl_id, args.metadata_dir, args.small_metadata_dir)
                print(f"  ctrl {ctrl_id}: {n} accepted rows → small metadata")
            else:
                print(f"  ctrl {ctrl_id}: no accepted images — omitting control row")

        alignment_id += 1

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_split = sum(1 for r in out_rows if r['should_split'] == 'True')
    n_ctrl  = sum(1 for r in out_rows if r['should_split'] == 'False')
    total_images = sum(sum(json.loads(r['generated_species_counts'])) for r in out_rows)

    print(f"\nWritten {len(out_rows)} rows to {args.output}")
    print(f"  Splits included : {alignment_id - 1}  (skipped: {skipped_splits})")
    print(f"  Splitting rows  : {n_split}")
    print(f"  Control rows    : {n_ctrl}")
    print(f"  Total accepted images: {total_images}")
    print(f"  Filtered metadata written to: {args.small_metadata_dir}/")


if __name__ == '__main__':
    main()
