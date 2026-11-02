"""
generate_hier_splits.py — Generate coarse-respecting train/unlabeled split files.

For each dataset × split type, assigns entire coarse categories to either
train_classes or unlabeled_classes, ensuring the hierarchical constraint:
  every fine class within a coarse is on exactly one side.

Selection criterion: greedy bin-packing by coarse size (largest first),
targeting proportional fine-class counts (not just proportional coarse counts).

Output: pickle files in data/ssb_splits/ with structure:
  {
    'train_classes':     [fine_class_idx, ...],
    'unlabeled_classes': [fine_class_idx, ...],
    'train_coarses':     [coarse_idx, ...],
    'unlabeled_coarses': [coarse_idx, ...],
    'fine_class_counts': {'train': N, 'unlabeled': M},
  }

Usage:
    python data/generate_hier_splits.py --dataset cub --split_type 5050
    python data/generate_hier_splits.py --dataset scars --split_type 6040
    python data/generate_hier_splits.py --dataset aircraft --split_type 5050
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import osr_split_dir


# ---------------------------------------------------------------------------
# Greedy bin-packing split
# ---------------------------------------------------------------------------

def generate_split(coarse_to_fines, train_frac):
    """
    Assign coarses to train or unlabeled using a greedy descending strategy
    that targets proportional fine-class counts.

    Args:
        coarse_to_fines: dict {coarse_idx: [fine_class_idx, ...]}
        train_frac: target fraction of fine classes to put in train (e.g. 0.5, 0.6)

    Returns:
        train_coarses, unlabeled_coarses (lists of coarse indices)
    """
    total_fine = sum(len(v) for v in coarse_to_fines.values())
    train_target = round(total_fine * train_frac)
    unlabeled_target = total_fine - train_target

    # Sort coarses largest-first for better packing accuracy
    coarses_sorted = sorted(coarse_to_fines.keys(),
                            key=lambda c: len(coarse_to_fines[c]),
                            reverse=True)

    train_count = 0
    unlabeled_count = 0
    train_coarses = []
    unlabeled_coarses = []

    for c in coarses_sorted:
        size = len(coarse_to_fines[c])
        # Assign to whichever bucket is furthest below its proportional target
        train_gap = train_target - train_count
        unlabeled_gap = unlabeled_target - unlabeled_count

        if train_gap >= unlabeled_gap:
            train_coarses.append(c)
            train_count += size
        else:
            unlabeled_coarses.append(c)
            unlabeled_count += size

    return train_coarses, unlabeled_coarses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SPLIT_FRACS = {
    '5050': 0.5,
    '6040': 0.6,
}

DATASETS = {
    'cub':      'data/coarse_labels/cub.json',
    'scars':    'data/coarse_labels/scars.json',
    'aircraft': 'data/coarse_labels/aircraft.json',
}


def main():
    parser = argparse.ArgumentParser(description='Generate hierarchical train/unlabeled splits.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help='Dataset name')
    parser.add_argument('--split_type', type=str, required=True,
                        choices=list(SPLIT_FRACS.keys()),
                        help='Split type: 5050 (50/50) or 6040 (60/40)')
    parser.add_argument('--coarse_json', type=str, default=None,
                        help='Path to coarse label JSON (auto-resolved if None)')
    parser.add_argument('--out_dir', type=str, default=osr_split_dir,
                        help='Output directory for pkl files (should match osr_split_dir in config.py)')
    args = parser.parse_args()

    json_path = args.coarse_json or DATASETS[args.dataset]
    if not os.path.isabs(json_path):
        # Resolve relative to the SelEx/ directory (where this script lives)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        selex_dir = os.path.dirname(script_dir)
        json_path = os.path.join(selex_dir, json_path)

    out_dir = args.out_dir

    print(f'Loading coarse labels from: {json_path}')
    with open(json_path) as f:
        raw = json.load(f)

    # Strip comment keys and build fine→coarse map
    fine_to_coarse = {int(k): v for k, v in raw.items() if not k.startswith('_')}

    # Build coarse→fines map
    coarse_to_fines = {}
    for fine, coarse in fine_to_coarse.items():
        coarse_to_fines.setdefault(coarse, []).append(fine)

    total_fine = sum(len(v) for v in coarse_to_fines.values())
    total_coarse = len(coarse_to_fines)
    print(f'Dataset {args.dataset}: {total_fine} fine classes, {total_coarse} coarse classes')

    train_frac = SPLIT_FRACS[args.split_type]
    train_coarses, unlabeled_coarses = generate_split(coarse_to_fines, train_frac)

    train_classes = sorted(f for c in train_coarses for f in coarse_to_fines[c])
    unlabeled_classes = sorted(f for c in unlabeled_coarses for f in coarse_to_fines[c])

    n_train = len(train_classes)
    n_unlabeled = len(unlabeled_classes)
    actual_frac = n_train / (n_train + n_unlabeled)

    print(f'Split {args.split_type} (target {train_frac:.0%} train):')
    print(f'  Train coarses:     {len(train_coarses):3d}  ({n_train:3d} fine classes, {actual_frac:.1%} of total)')
    print(f'  Unlabeled coarses: {len(unlabeled_coarses):3d}  ({n_unlabeled:3d} fine classes, {1-actual_frac:.1%} of total)')

    result = {
        'train_classes':     train_classes,
        'unlabeled_classes': unlabeled_classes,
        'train_coarses':     sorted(train_coarses),
        'unlabeled_coarses': sorted(unlabeled_coarses),
        'fine_class_counts': {'train': n_train, 'unlabeled': n_unlabeled},
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset}_hier_{args.split_type}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'Saved to: {out_path}')


if __name__ == '__main__':
    main()
