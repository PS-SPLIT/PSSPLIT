"""
Generate standard (non-hierarchical) GCD split pkl files for aves_small and aves_full.

Two new split types are produced for each dataset:

  gcd_paired : Both splitting species always on the same side; control always on the
               opposite side.  50% of alignments have splitting=known, 50% splitting=novel,
               assigned greedily to balance total image counts.

  gcd_single : The two splitting species are on *opposite* sides (one known, one novel),
               alternating which one is known across alignments (sorted by alignment_id).
               Control species are assigned 50/50 by greedy image-count balance.

Each pkl contains: {'train_classes': [...], 'unlabeled_classes': [...]}
No coarse info — standard GCD pipelines do not need it.

Run once from SelEx/:
    conda run -n slex_old python data/generate_aves_std_splits.py
"""
import ast
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import aves_small_splits_csv, aves_full_splits_csv, osr_split_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df(csv_path):
    df = pd.read_csv(csv_path)
    df['generated_species_taxon_ids'] = df['generated_species_taxon_ids'].apply(ast.literal_eval)
    df['generated_species_counts'] = df['generated_species_counts'].apply(ast.literal_eval)
    return df


def _greedy_partition(records, key_fn, cap=None):
    """
    Greedily partition `records` into (A, B) with balanced sums of key_fn(r).
    If `cap` is given, each side gets at most `cap` records; otherwise balanced
    by count (half each).

    Returns (side_a_ids, side_b_ids, sum_a, sum_b).
    """
    records = sorted(records, key=lambda r: key_fn(r), reverse=True)
    n = len(records)
    if cap is None:
        n_a = n // 2
        n_b = n - n_a
    else:
        n_a = n_b = cap

    side_a, side_b = [], []
    sum_a, sum_b = 0, 0

    for r in records:
        can_a = len(side_a) < n_a
        can_b = len(side_b) < n_b
        if can_a and (not can_b or sum_a <= sum_b):
            side_a.append(r)
            sum_a += key_fn(r)
        else:
            side_b.append(r)
            sum_b += key_fn(r)

    return side_a, side_b, sum_a, sum_b


# ---------------------------------------------------------------------------
# gcd_paired
# ---------------------------------------------------------------------------

def make_gcd_paired(df, taxon_to_idx):
    """
    Both splitting species land on the same side; the control on the other.
    Greedy image-count balance determines which 50% get splitting=known.

    Returns (train_classes, unlabeled_classes).
    """
    # Per-alignment: splitting taxon_ids, control taxon_id, total image count
    aln_info = {}
    for _, row in df.iterrows():
        aln_id = row['alignment_id']
        if aln_id not in aln_info:
            aln_info[aln_id] = {'splitting': None, 'control': None, 'total_imgs': 0}
        imgs = sum(row['generated_species_counts'])
        aln_info[aln_id]['total_imgs'] += imgs
        if row['should_split']:
            aln_info[aln_id]['splitting'] = row['generated_species_taxon_ids']  # list of 2
        else:
            aln_info[aln_id]['control'] = row['generated_species_taxon_ids'][0]  # single

    # Only keep alignments that have both sides populated
    valid = {aid: v for aid, v in aln_info.items()
             if v['splitting'] is not None and v['control'] is not None}

    records = list(valid.items())  # (aln_id, info_dict)

    # Greedy partition: known-splitting half vs novel-splitting half
    known_split_recs, novel_split_recs, sum_k, sum_n = _greedy_partition(
        records, key_fn=lambda r: r[1]['total_imgs']
    )

    train_classes = sorted(
        {taxon_to_idx[t] for aln_id, info in known_split_recs for t in info['splitting']}
        | {taxon_to_idx[info['control']] for aln_id, info in novel_split_recs}
    )
    unlabeled_classes = sorted(
        {taxon_to_idx[info['control']] for aln_id, info in known_split_recs}
        | {taxon_to_idx[t] for aln_id, info in novel_split_recs for t in info['splitting']}
    )

    return train_classes, unlabeled_classes, sum_k, sum_n


# ---------------------------------------------------------------------------
# gcd_single
# ---------------------------------------------------------------------------

def make_gcd_single(df, taxon_to_idx):
    """
    The two splitting species go on opposite sides: for alignment i (0-indexed,
    sorted by alignment_id), species_A=known if i is even, species_B=known if odd.
    Control species are greedy-balanced across sides.

    Returns (train_classes, unlabeled_classes).
    """
    aln_info = {}
    for _, row in df.iterrows():
        aln_id = row['alignment_id']
        if aln_id not in aln_info:
            aln_info[aln_id] = {'splitting': None, 'control': None,
                                 'control_imgs': 0, 'total_imgs': 0}
        imgs = sum(row['generated_species_counts'])
        aln_info[aln_id]['total_imgs'] += imgs
        if row['should_split']:
            aln_info[aln_id]['splitting'] = row['generated_species_taxon_ids']  # [A, B]
        else:
            aln_info[aln_id]['control'] = row['generated_species_taxon_ids'][0]
            aln_info[aln_id]['control_imgs'] = sum(row['generated_species_counts'])

    valid = {aid: v for aid, v in aln_info.items()
             if v['splitting'] is not None and v['control'] is not None}

    sorted_aln_ids = sorted(valid.keys())

    train_splitting = set()
    unlabeled_splitting = set()
    for i, aln_id in enumerate(sorted_aln_ids):
        sp = valid[aln_id]['splitting']  # [A, B]
        if i % 2 == 0:
            train_splitting.add(taxon_to_idx[sp[0]])      # A → known
            unlabeled_splitting.add(taxon_to_idx[sp[1]])  # B → novel
        else:
            train_splitting.add(taxon_to_idx[sp[1]])      # B → known
            unlabeled_splitting.add(taxon_to_idx[sp[0]])  # A → novel

    # Greedy-balance the control species
    ctrl_records = [(aln_id, valid[aln_id]['control'], valid[aln_id]['control_imgs'])
                    for aln_id in sorted_aln_ids]
    known_ctrl_recs, novel_ctrl_recs, sum_k_ctrl, sum_n_ctrl = _greedy_partition(
        ctrl_records, key_fn=lambda r: r[2]
    )

    train_ctrl = {taxon_to_idx[r[1]] for r in known_ctrl_recs}
    unlabeled_ctrl = {taxon_to_idx[r[1]] for r in novel_ctrl_recs}

    train_classes = sorted(train_splitting | train_ctrl)
    unlabeled_classes = sorted(unlabeled_splitting | unlabeled_ctrl)

    return train_classes, unlabeled_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_split(name, train_classes, unlabeled_classes, n_total):
    print(f"  [{name}]  train={len(train_classes)}  novel={len(unlabeled_classes)}"
          f"  total={len(train_classes)+len(unlabeled_classes)}"
          f"  (expected {n_total})")
    assert len(train_classes) + len(unlabeled_classes) == n_total, \
        f"Class count mismatch: {len(train_classes)+len(unlabeled_classes)} != {n_total}"
    overlap = set(train_classes) & set(unlabeled_classes)
    assert not overlap, f"Overlap between train and unlabeled: {overlap}"


def generate_for_dataset(dataset_name, csv_path, taxon_to_idx, out_prefix):
    df = _load_df(csv_path)
    n_total = len(taxon_to_idx)

    print(f"\n=== {dataset_name} (n_fine={n_total}) ===")

    # --- gcd_paired ---
    train_p, unlab_p, sum_k_p, sum_n_p = make_gcd_paired(df, taxon_to_idx)
    _print_split('gcd_paired', train_p, unlab_p, n_total)
    print(f"    image balance: known~{sum_k_p}  novel~{sum_n_p}  Δ={abs(sum_k_p-sum_n_p)}")

    pkl_paired = os.path.join(osr_split_dir, f'{out_prefix}_gcd_paired.pkl')
    with open(pkl_paired, 'wb') as f:
        pickle.dump({'train_classes': train_p, 'unlabeled_classes': unlab_p}, f)
    print(f"    → {pkl_paired}")

    # --- gcd_single ---
    train_s, unlab_s = make_gcd_single(df, taxon_to_idx)
    _print_split('gcd_single', train_s, unlab_s, n_total)

    pkl_single = os.path.join(osr_split_dir, f'{out_prefix}_gcd_single.pkl')
    with open(pkl_single, 'wb') as f:
        pickle.dump({'train_classes': train_s, 'unlabeled_classes': unlab_s}, f)
    print(f"    → {pkl_single}")


def main():
    os.makedirs(osr_split_dir, exist_ok=True)

    # ---- aves_small ----
    df_small = _load_df(aves_small_splits_csv)
    all_taxon_ids_small = sorted(
        set(tid for ids in df_small['generated_species_taxon_ids'] for tid in ids)
    )
    taxon_to_idx_small = {tid: i for i, tid in enumerate(all_taxon_ids_small)}

    generate_for_dataset('aves_small', aves_small_splits_csv,
                         taxon_to_idx_small, 'aves_small')

    # ---- aves_full ----
    # Must use _build_extended_aves_index() to preserve aves_small index positions 0..191
    from data.aves_full import _build_extended_aves_index
    all_taxon_ids_full, taxon_to_idx_full, _, _ = _build_extended_aves_index()

    generate_for_dataset('aves_full', aves_full_splits_csv,
                         taxon_to_idx_full, 'aves_full')

    print("\nDone.")


if __name__ == '__main__':
    main()
