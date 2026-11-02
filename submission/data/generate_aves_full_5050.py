"""
Generate the permanent 50/50 known/novel split for aves_full.

Constraints:
  1. aves_small alignment pairs keep their exact known/novel assignment from aves_small_5050.pkl.
  2. New aves_full-only alignment pairs are split 50/50 at the alignment level using a
     greedy image-count-balanced partition.
  3. All alignment pairs land on the same side (no mixed pairs).
  4. Extended aves index is used: aves_small IDs preserved at 0..191, new at 192+.

Run from SelEx/:
    conda run -n slex_old python data/generate_aves_full_5050.py

Output:
    {osr_split_dir}/aves_full_5050.pkl
"""
import ast
import json
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import aves_full_splits_csv, aves_small_splits_csv, osr_split_dir
from data.aves_full import _build_extended_aves_index

PKL_PATH = os.path.join(osr_split_dir, 'aves_full_5050.pkl')
SMALL_PKL_PATH = os.path.join(osr_split_dir, 'aves_small_5050.pkl')


def main():
    # ---------- Build extended index ----------
    all_taxon_ids, taxon_to_idx, all_split_ids, split_to_coarse_idx = _build_extended_aves_index()
    n_fine = len(all_taxon_ids)

    # ---------- Load aves_full CSV ----------
    df_full = pd.read_csv(aves_full_splits_csv)
    df_full['generated_species_taxon_ids'] = df_full['generated_species_taxon_ids'].apply(ast.literal_eval)
    df_full['generated_species_counts']    = df_full['generated_species_counts'].apply(ast.literal_eval)

    # ---------- Load aves_small 5050 assignment ----------
    with open(SMALL_PKL_PATH, 'rb') as f:
        small_split = pickle.load(f)
    small_known_set  = set(small_split['train_classes'])
    small_novel_set  = set(small_split['unlabeled_classes'])

    # ---------- Map aves_small alignment_ids to known/novel ----------
    df_small = pd.read_csv(aves_small_splits_csv)
    df_small['generated_species_taxon_ids'] = df_small['generated_species_taxon_ids'].apply(ast.literal_eval)

    small_taxon_ids = sorted(set(t for ids in df_small['generated_species_taxon_ids'] for t in ids))
    small_taxon_to_idx = {t: i for i, t in enumerate(small_taxon_ids)}

    small_aln_ids = set(df_small['alignment_id'].unique())
    aln_to_known_novel = {}  # alignment_id -> 'known' | 'novel'

    for aln_id, grp in df_small.groupby('alignment_id'):
        fine_idxs = {small_taxon_to_idx[t]
                     for ids in grp['generated_species_taxon_ids'] for t in ids}
        if fine_idxs <= small_known_set:
            aln_to_known_novel[aln_id] = 'known'
        elif fine_idxs <= small_novel_set:
            aln_to_known_novel[aln_id] = 'novel'
        else:
            raise ValueError(
                f"aves_small alignment {aln_id} has mixed known/novel fine classes — "
                f"constraint violation in aves_small_5050.pkl"
            )

    n_small_known = sum(1 for v in aln_to_known_novel.values() if v == 'known')
    n_small_novel = sum(1 for v in aln_to_known_novel.values() if v == 'novel')
    print(f"aves_small alignments: {len(aln_to_known_novel)} total | "
          f"{n_small_known} known, {n_small_novel} novel")

    # ---------- Image counts per alignment (full index) ----------
    aln_img_counts = {}
    for aln_id, grp in df_full.groupby('alignment_id'):
        total_imgs = sum(c for counts in grp['generated_species_counts'] for c in counts)
        aln_img_counts[aln_id] = total_imgs

    # Image totals already committed by aves_small
    sum_known = sum(aln_img_counts[a] for a in aln_to_known_novel if aln_to_known_novel[a] == 'known')
    sum_novel = sum(aln_img_counts[a] for a in aln_to_known_novel if aln_to_known_novel[a] == 'novel')

    # ---------- New aves_full-only alignments ----------
    new_aln_ids = sorted(
        set(df_full['alignment_id'].unique()) - small_aln_ids,
        key=lambda a: -aln_img_counts[a],  # descending by image count
    )
    n_new = len(new_aln_ids)
    total_aln = len(aln_to_known_novel) + n_new  # 257

    # Target counts for new alignments: split so total is as close to 50/50 as possible
    n_known_target = total_aln // 2 - n_small_known   # e.g. 128 - 32 = 96
    n_novel_target = (total_aln - total_aln // 2) - n_small_novel  # e.g. 129 - 32 = 97

    print(f"New aves_full-only alignments: {n_new} | "
          f"target known={n_known_target}, novel={n_novel_target}")

    # Greedy balanced partition over new alignments
    n_known_new, n_novel_new = 0, 0
    for aln_id in new_aln_ids:
        can_known = n_known_new < n_known_target
        can_novel = n_novel_new < n_novel_target
        if can_known and (not can_novel or sum_known <= sum_novel):
            aln_to_known_novel[aln_id] = 'known'
            sum_known += aln_img_counts[aln_id]
            n_known_new += 1
        else:
            aln_to_known_novel[aln_id] = 'novel'
            sum_novel += aln_img_counts[aln_id]
            n_novel_new += 1

    n_known_total = n_small_known + n_known_new
    n_novel_total = n_small_novel + n_novel_new
    print(f"Final split: {n_known_total} known alignments (~{sum_known} imgs) | "
          f"{n_novel_total} novel alignments (~{sum_novel} imgs)")
    print(f"Image imbalance: {abs(sum_known - sum_novel)} images")

    # ---------- Build coarse / fine class lists ----------
    coarse_map_json_path = os.path.join(
        os.path.dirname(__file__), 'coarse_labels', 'aves_full.json')
    with open(coarse_map_json_path) as f:
        coarse_map = json.load(f)  # {str(fine_idx): coarse_idx}

    fine_to_coarse = {int(k): v for k, v in coarse_map.items()}
    coarse_to_fines = {}
    for fi, ci in fine_to_coarse.items():
        coarse_to_fines.setdefault(ci, []).append(fi)

    # Alignment -> coarse indices (full extended index).
    # Use sets to handle the edge case where one split_id appears on multiple rows
    # of the same alignment (e.g. alignment 237 has split_id=4141 on both rows).
    aln_to_coarses = {}
    for _, row in df_full.iterrows():
        aln_id = row['alignment_id']
        coarse_idx = split_to_coarse_idx[row['split_id']]
        aln_to_coarses.setdefault(aln_id, set()).add(coarse_idx)

    known_coarse_idxs = sorted(
        c for aln, side in aln_to_known_novel.items()
        if side == 'known'
        for c in aln_to_coarses[aln]
    )
    novel_coarse_idxs = sorted(
        c for aln, side in aln_to_known_novel.items()
        if side == 'novel'
        for c in aln_to_coarses[aln]
    )

    train_classes      = sorted(f for c in known_coarse_idxs for f in coarse_to_fines.get(c, []))
    unlabeled_classes  = sorted(f for c in novel_coarse_idxs for f in coarse_to_fines.get(c, []))

    assert len(train_classes) + len(unlabeled_classes) == n_fine, (
        f"Class count mismatch: {len(train_classes)} + {len(unlabeled_classes)} != {n_fine}"
    )

    # Verify aves_small classes are on the same side as in aves_small_5050.pkl
    aves_small_fine_idxs = set(range(len(small_taxon_ids)))  # 0..191
    train_small   = aves_small_fine_idxs & set(train_classes)
    novel_small   = aves_small_fine_idxs & set(unlabeled_classes)
    assert train_small   == small_known_set,  "aves_small known classes mismatch!"
    assert novel_small   == small_novel_set,  "aves_small novel classes mismatch!"
    print(f"aves_small consistency check: PASSED "
          f"({len(train_small)} known, {len(novel_small)} novel)")

    split_info = {
        'train_classes':      train_classes,
        'unlabeled_classes':  unlabeled_classes,
        'train_coarses':      known_coarse_idxs,
        'unlabeled_coarses':  novel_coarse_idxs,
    }

    os.makedirs(osr_split_dir, exist_ok=True)
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(split_info, f)

    print(f"\nWrote aves_full_5050.pkl:")
    print(f"  Known:  {n_known_total} alignments, {len(known_coarse_idxs)} coarses, "
          f"{len(train_classes)} fine classes")
    print(f"  Novel:  {n_novel_total} alignments, {len(novel_coarse_idxs)} coarses, "
          f"{len(unlabeled_classes)} fine classes")
    print(f"→ {PKL_PATH}")


if __name__ == '__main__':
    main()
