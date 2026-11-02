"""
Generate coarse-labels JSON and 50/50 train/unlabeled split pkl for the aves_small dataset.

Run once from SelEx/:
    conda run -n slex_old python data/generate_aves_small_splits.py

Outputs:
    data/coarse_labels/aves_small.json
    {osr_split_dir}/aves_small_5050.pkl

Split strategy: 50/50 at the alignment_id level (33/33 alignment pairs).
Paired splitting and non-splitting coarses always land in the same split.
Image counts are balanced across splits using a greedy algorithm.
"""
import ast
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import aves_small_splits_csv, osr_split_dir

COARSE_JSON_PATH = os.path.join(os.path.dirname(__file__), 'coarse_labels', 'aves_small.json')
PKL_PATH = os.path.join(osr_split_dir, 'aves_small_5050.pkl')


def balanced_equal_partition(aln_records):
    """
    Greedily split alignment records into two equal-sized halves with balanced
    total image counts. Processes largest-image-count pairs first and always
    assigns each pair to whichever half currently has the smaller total.

    aln_records: list of dicts with keys 'alignment_id' and 'total_imgs'
    Returns (split_a_ids, split_b_ids) — each a list of alignment_id values.
    """
    records = sorted(aln_records, key=lambda x: x['total_imgs'], reverse=True)
    n = len(records)
    half = n // 2  # both halves get exactly half (n must be even for aves_small)
    n_a, n_b = half, n - half  # n_a == n_b == 33 for 66 pairs

    split_a, split_b = [], []
    sum_a, sum_b = 0, 0

    for rec in records:
        can_a = len(split_a) < n_a
        can_b = len(split_b) < n_b
        if can_a and (not can_b or sum_a <= sum_b):
            split_a.append(rec['alignment_id'])
            sum_a += rec['total_imgs']
        else:
            split_b.append(rec['alignment_id'])
            sum_b += rec['total_imgs']

    return split_a, split_b, sum_a, sum_b


def main():
    df = pd.read_csv(aves_small_splits_csv)
    df['generated_species_taxon_ids'] = df['generated_species_taxon_ids'].apply(ast.literal_eval)
    df['generated_species_counts'] = df['generated_species_counts'].apply(ast.literal_eval)

    all_taxon_ids = sorted(set(tid for ids in df['generated_species_taxon_ids'] for tid in ids))
    taxon_to_idx = {tid: i for i, tid in enumerate(all_taxon_ids)}

    all_split_ids = sorted(df['split_id'].unique())
    split_to_coarse_idx = {sid: i for i, sid in enumerate(all_split_ids)}

    # Build coarse_map: fine_class_int_idx -> coarse_int_idx
    coarse_map = {}
    for _, row in df.iterrows():
        coarse_idx = split_to_coarse_idx[row['split_id']]
        for taxon_id in row['generated_species_taxon_ids']:
            fine_idx = taxon_to_idx[taxon_id]
            coarse_map[str(fine_idx)] = coarse_idx

    os.makedirs(os.path.dirname(COARSE_JSON_PATH), exist_ok=True)
    with open(COARSE_JSON_PATH, 'w') as f:
        json.dump(coarse_map, f, indent=2, sort_keys=True)
    print(f"Wrote coarse labels: {len(all_taxon_ids)} fine classes, "
          f"{len(all_split_ids)} coarses → {COARSE_JSON_PATH}")

    # Per-alignment image count for balancing
    aln_records = []
    for aln_id, grp in df.groupby('alignment_id'):
        total_imgs = sum(c for counts in grp['generated_species_counts'] for c in counts)
        aln_records.append({'alignment_id': aln_id, 'total_imgs': total_imgs})

    known_aln_ids, novel_aln_ids, sum_known, sum_novel = balanced_equal_partition(aln_records)

    # Coarse → fine lookup
    coarse_to_fine = {}
    for fine_str, coarse_idx in coarse_map.items():
        coarse_to_fine.setdefault(coarse_idx, []).append(int(fine_str))

    # Alignment → coarse indices
    aln_to_coarses = {}
    for _, row in df.iterrows():
        aln_id = row['alignment_id']
        coarse_idx = split_to_coarse_idx[row['split_id']]
        aln_to_coarses.setdefault(aln_id, []).append(coarse_idx)

    known_coarse_idxs = sorted(c for aln in known_aln_ids for c in aln_to_coarses[aln])
    novel_coarse_idxs = sorted(c for aln in novel_aln_ids for c in aln_to_coarses[aln])

    train_classes = sorted(f for c in known_coarse_idxs for f in coarse_to_fine.get(c, []))
    unlabeled_classes = sorted(f for c in novel_coarse_idxs for f in coarse_to_fine.get(c, []))

    split_info = {
        'train_classes':      train_classes,
        'unlabeled_classes':  unlabeled_classes,
        'train_coarses':      known_coarse_idxs,
        'unlabeled_coarses':  novel_coarse_idxs,
    }

    os.makedirs(osr_split_dir, exist_ok=True)
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(split_info, f)

    print(f"Wrote 50/50 split:")
    print(f"  Known:  {len(known_aln_ids)} alignments, "
          f"{len(known_coarse_idxs)} coarses, "
          f"{len(train_classes)} fine classes, "
          f"~{sum_known} images")
    print(f"  Novel:  {len(novel_aln_ids)} alignments, "
          f"{len(novel_coarse_idxs)} coarses, "
          f"{len(unlabeled_classes)} fine classes, "
          f"~{sum_novel} images")
    print(f"  Image imbalance: {abs(sum_known - sum_novel)} images")
    print(f"→ {PKL_PATH}")


if __name__ == '__main__':
    main()
