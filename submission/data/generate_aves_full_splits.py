"""
Generate coarse-labels JSON and all-known split pkl for the aves_full dataset.

Run once from SelEx/:
    conda run -n slex_old python data/generate_aves_full_splits.py

Outputs:
    data/coarse_labels/aves_full.json
    {osr_split_dir}/aves_full_all_known.pkl

The coarse-labels JSON uses the extended aves index (aves_small IDs preserved at 0..191,
new aves_full-only IDs at 192..788).  It covers ALL 789 fine classes across 525 coarses.

The all-known pkl puts every fine class into train_classes so that feature extraction
and supervised probing can be run over ALL coarses before the known/novel split is decided.
After probing, run generate_aves_full_5050.py to produce the permanent split.
"""
import ast
import json
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import aves_full_splits_csv, osr_split_dir
from data.aves_full import _build_extended_aves_index

COARSE_JSON_PATH = os.path.join(os.path.dirname(__file__), 'coarse_labels', 'aves_full.json')
ALL_KNOWN_PKL_PATH = os.path.join(osr_split_dir, 'aves_full_all_known.pkl')


def main():
    # Build extended index (aves_small IDs preserved, new IDs appended)
    all_taxon_ids, taxon_to_idx, all_split_ids, split_to_coarse_idx = _build_extended_aves_index()

    n_fine   = len(all_taxon_ids)
    n_coarse = len(all_split_ids)

    # Build coarse map from aves_full CSV (covers all 263 alignment pairs = 789 fine classes)
    df = pd.read_csv(aves_full_splits_csv)
    df['generated_species_taxon_ids'] = df['generated_species_taxon_ids'].apply(ast.literal_eval)

    coarse_map = {}
    for _, row in df.iterrows():
        coarse_idx = split_to_coarse_idx[row['split_id']]
        for taxon_id in row['generated_species_taxon_ids']:
            fine_idx = taxon_to_idx[taxon_id]
            coarse_map[str(fine_idx)] = coarse_idx

    assert len(coarse_map) == n_fine, (
        f"coarse_map covers {len(coarse_map)} fine classes, expected {n_fine}"
    )

    os.makedirs(os.path.dirname(COARSE_JSON_PATH), exist_ok=True)
    with open(COARSE_JSON_PATH, 'w') as f:
        json.dump(coarse_map, f, indent=2, sort_keys=True)
    print(f"Wrote coarse labels: {n_fine} fine classes, {n_coarse} coarses → {COARSE_JSON_PATH}")

    # All-known split: every fine class is 'train' (no unlabeled split yet)
    all_known = {
        'train_classes':      list(range(n_fine)),
        'unlabeled_classes':  [],
        'train_coarses':      list(range(n_coarse)),
        'unlabeled_coarses':  [],
    }

    os.makedirs(osr_split_dir, exist_ok=True)
    with open(ALL_KNOWN_PKL_PATH, 'wb') as f:
        pickle.dump(all_known, f)
    print(f"Wrote all-known split: {n_fine} fine classes → {ALL_KNOWN_PKL_PATH}")
    print(f"  aves_small fine classes: 0..191 (unchanged)")
    print(f"  aves_full-only new fine classes: 192..{n_fine - 1}")
    print(f"  aves_small coarses: 0..127 (unchanged)")
    print(f"  aves_full-only new coarses: 128..{n_coarse - 1}")


if __name__ == '__main__':
    main()
