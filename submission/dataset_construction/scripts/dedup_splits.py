"""
dedup_splits.py

Find and remove splits that share a generated species with a newer split.

When a generated species appears in multiple splits, the most recent split is
kept (by date, then by split_id as tiebreaker). Older duplicates are removed.

For the splits CSV (aves_splits_small.csv or similar):
  - The splitting row AND the paired control row (same alignment_id) are both removed.

For aves_splits.json and aves_controls.json:
  - The split entry is removed by split_id key.

Usage:
    # Dry-run (show what would change):
    python dedup_splits.py

    # Apply to both small CSV and full JSON:
    python dedup_splits.py --apply

    # Apply only to the small CSV:
    python dedup_splits.py --apply --no_json

    # Apply only to the JSON files (not the small CSV):
    python dedup_splits.py --apply --no_csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict


# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def save_csv(rows, path, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ── core logic ────────────────────────────────────────────────────────────────

def find_duplicate_splits_json(splits_json, explicit_keep=()):
    """
    Scan aves_splits.json for generated species appearing in >1 split.
    Returns a set of split_id strings to remove (older duplicates).
    explicit_keep: split_ids that must be kept regardless of date ordering.
    """
    taxon_to_splits = defaultdict(list)
    for split_id, split in splits_json.items():
        gen_ids = [str(sp['taxon_id']) for sp in split.get('generated_species', [])]
        date    = split.get('date', '')
        for tid in gen_ids:
            taxon_to_splits[tid].append({
                'split_id': split_id,
                'date': date,
                'gen_ids': gen_ids,
            })

    remove_ids = set()
    groups = {}
    for tid, entries in taxon_to_splits.items():
        if len(entries) < 2:
            continue
        # If any entry is in explicit_keep, force it to sort first
        def sort_key(e):
            forced = 1 if e['split_id'] in explicit_keep else 0
            return (forced, e['date'], int(e['split_id']))
        sorted_entries = sorted(entries, key=sort_key, reverse=True)
        keep = sorted_entries[0]
        old  = sorted_entries[1:]
        groups[tid] = {'keep': keep, 'remove': old}
        for e in old:
            remove_ids.add(e['split_id'])

    return remove_ids, groups


def find_duplicate_splits_csv(csv_rows, explicit_keep=()):
    """
    Scan a splits CSV for generated species appearing in >1 splitting row.
    Returns a set of alignment_id strings to remove (older duplicates).
    explicit_keep: split_ids (not alignment_ids) that must be kept.
    """
    taxon_to_alignments = defaultdict(list)
    for r in csv_rows:
        if r.get('should_split') != 'True':
            continue
        gen_ids = json.loads(r['generated_species_taxon_ids'])
        date    = r.get('date', '')
        aid     = r['alignment_id']
        sid     = r['split_id']
        for tid in gen_ids:
            taxon_to_alignments[str(tid)].append({
                'alignment_id': aid,
                'split_id': sid,
                'date': date,
                'gen_ids': gen_ids,
            })

    remove_alignments = set()
    groups = {}
    for tid, entries in taxon_to_alignments.items():
        if len(entries) < 2:
            continue
        def sort_key(e):
            forced = 1 if e['split_id'] in explicit_keep else 0
            return (forced, e['date'], int(e['split_id']))
        sorted_entries = sorted(entries, key=sort_key, reverse=True)
        keep = sorted_entries[0]
        old  = sorted_entries[1:]
        groups[tid] = {'keep': keep, 'remove': old}
        for e in old:
            remove_alignments.add(e['alignment_id'])

    return remove_alignments, groups


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--splits_csv',    default='aves_splits_small.csv')
    parser.add_argument('--splits_json',   default='aves_splits.json')
    parser.add_argument('--controls_json', default='aves_controls.json')
    parser.add_argument('--apply',    action='store_true',
                        help='Write changes (default is dry-run)')
    parser.add_argument('--no_csv',   action='store_true',
                        help='Skip modifying the splits CSV')
    parser.add_argument('--no_json',  action='store_true',
                        help='Skip modifying the JSON files')
    parser.add_argument('--explicit_keep', nargs='*', default=[],
                        help='Split IDs to always keep (overrides date/id tiebreaker)')
    args = parser.parse_args()

    # ── CSV ──────────────────────────────────────────────────────────────────
    if not args.no_csv:
        print(f"=== {args.splits_csv} ===")
        if not os.path.exists(args.splits_csv):
            print(f"  Not found — skipping.")
        else:
            csv_rows    = load_csv(args.splits_csv)
            fieldnames  = list(csv_rows[0].keys()) if csv_rows else []
            remove_aids, csv_groups = find_duplicate_splits_csv(csv_rows, set(args.explicit_keep))

            if not csv_groups:
                print("  No duplicate generated species found.")
            else:
                for tid, g in csv_groups.items():
                    keep = g['keep']
                    print(f"  taxon {tid}:")
                    print(f"    KEEP   alignment={keep['alignment_id']}  split_id={keep['split_id']}"
                          f"  date={keep['date']}  gen_ids={keep['gen_ids']}")
                    for old in g['remove']:
                        print(f"    REMOVE alignment={old['alignment_id']}  split_id={old['split_id']}"
                              f"  date={old['date']}  gen_ids={old['gen_ids']}")

                # Count rows affected (both gen + ctrl rows share the alignment_id)
                affected = [r for r in csv_rows if r['alignment_id'] in remove_aids]
                print(f"\n  Alignments to remove: {sorted(remove_aids)}")
                print(f"  CSV rows affected   : {len(affected)} "
                      f"({sum(1 for r in affected if r['should_split']=='True')} gen "
                      f"+ {sum(1 for r in affected if r['should_split']=='False')} ctrl)")

                if args.apply:
                    new_rows = [r for r in csv_rows if r['alignment_id'] not in remove_aids]
                    save_csv(new_rows, args.splits_csv, fieldnames)
                    print(f"  Written: {len(new_rows)} rows remain in {args.splits_csv}")
                else:
                    print("  (dry-run) Pass --apply to write changes.")
        print()

    # ── JSON ─────────────────────────────────────────────────────────────────
    if not args.no_json:
        print(f"=== {args.splits_json} + {args.controls_json} ===")
        missing = [p for p in [args.splits_json, args.controls_json]
                   if not os.path.exists(p)]
        if missing:
            print(f"  Not found: {missing} — skipping.")
        else:
            splits_json   = load_json(args.splits_json)
            controls_json = load_json(args.controls_json)

            remove_sids, json_groups = find_duplicate_splits_json(splits_json, set(args.explicit_keep))

            if not json_groups:
                print("  No duplicate generated species found in aves_splits.json.")
            else:
                for tid, g in json_groups.items():
                    keep = g['keep']
                    print(f"  taxon {tid}:")
                    print(f"    KEEP   split_id={keep['split_id']}  date={keep['date']}"
                          f"  gen_ids={keep['gen_ids']}")
                    for old in g['remove']:
                        print(f"    REMOVE split_id={old['split_id']}  date={old['date']}"
                              f"  gen_ids={old['gen_ids']}")

                ctrl_to_remove = [sid for sid in remove_sids if sid in controls_json]
                print(f"\n  Split IDs to remove from aves_splits.json  : {sorted(remove_sids)}")
                print(f"  Split IDs to remove from aves_controls.json: {sorted(ctrl_to_remove)}")

                if args.apply:
                    for sid in remove_sids:
                        splits_json.pop(sid, None)
                        controls_json.pop(sid, None)
                    save_json(splits_json,   args.splits_json)
                    save_json(controls_json, args.controls_json)
                    print(f"  Written: {len(splits_json)} splits in {args.splits_json}")
                    print(f"  Written: {len(controls_json)} entries in {args.controls_json}")
                else:
                    print("  (dry-run) Pass --apply to write changes.")


if __name__ == '__main__':
    main()
