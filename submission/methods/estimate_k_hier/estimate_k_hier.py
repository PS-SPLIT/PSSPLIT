"""
estimate_k_hier.py — Per-coarse K estimation for hierarchical category discovery.

For each unknown coarse category (one whose fine classes are all in unlabeled_classes),
estimates the number of fine clusters using an existing SelEx K estimation method applied
to the features of that coarse category in isolation.

Probe classes: fine classes from known coarse categories (all fine classes there are labeled).
Unknown coarse: only unlabeled fine classes appear.

Also supports --use_oracle_k to derive K directly from the coarse JSON (counts unlabeled
fine classes per coarse — exact when fine-class counts match ground truth K).

Run standalone:
    python -m methods.estimate_k_hier.estimate_k_hier \
        --dataset_name scars \
        --coarse_labels_path data/coarse_labels/scars.json \
        --k_results_path /tmp/scars_k_per_coarse.json \
        --search_mode autonovel

Or called from run_pipeline_hier.py with --estimate_k_per_coarse True.
"""

import argparse
import json
import os
from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import feature_extract_dir
from data.data_utils import build_coarse_of, load_coarse_map, oracle_k_per_coarse
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from project_utils.general_utils import str2bool


# ---------------------------------------------------------------------------
# In-memory dataset — mimics FeatureVectorDataset over MergedDataset
# ---------------------------------------------------------------------------

class NumpyFeatureDataset(Dataset):
    """
    In-memory dataset from numpy arrays.
    Returns (feat_tensor, label_int, uq_idx_int, mask_lab_bool) to match
    the format expected by all estimate_k_* methods.
    """
    def __init__(self, feats, labels, mask_lab):
        self.feats   = feats.astype(np.float32)
        self.labels  = labels.astype(np.int64)
        self.mask_lab = mask_lab.astype(bool)

    def __getitem__(self, i):
        return (torch.from_numpy(self.feats[i]),
                int(self.labels[i]),
                i,
                bool(self.mask_lab[i]))

    def __len__(self):
        return len(self.feats)


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def load_all_features(train_dataset, feature_root, batch_size, num_workers, device):
    """
    Load all features from disk for the merged train dataset.
    Returns (feats [N,D] float32, labels [N] int64 remapped, mask_lab [N] bool).

    MergedDataset.target_transform is always None; the per-class target transform
    lives on the sub-datasets.  We save it before FeatureVectorDataset nullifies
    the sub-datasets' transforms, then apply it after loading so that all_labels
    are correctly remapped (0..num_classes-1) rather than original class indices.
    """
    from data.data_utils import MergedDataset as _MergedDataset

    # Save target_transform BEFORE FeatureVectorDataset nullifies the sub-datasets.
    target_transform = None
    if isinstance(train_dataset, _MergedDataset):
        target_transform = getattr(train_dataset.labelled_dataset, 'target_transform', None)
        if target_transform is None:
            target_transform = getattr(train_dataset.unlabelled_dataset, 'target_transform', None)

    feat_dataset = FeatureVectorDataset(base_dataset=train_dataset,
                                        feature_root=os.path.join(feature_root, 'train'))
    loader = DataLoader(feat_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    feats_list, labels_list, mask_list = [], [], []
    for feats, label, _, mask_lab in loader:
        feats = F.normalize(feats.float().to(device), dim=-1)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(label.numpy() if isinstance(label, torch.Tensor) else np.array(label))
        mask_list.append(np.array(mask_lab).astype(bool))

    all_labels = np.concatenate(labels_list, axis=0)

    # Apply target_transform to produce remapped labels (0..num_classes-1).
    if target_transform is not None:
        all_labels = np.array([target_transform(int(l)) for l in all_labels], dtype=np.int64)

    return (np.concatenate(feats_list,  axis=0),
            all_labels,
            np.concatenate(mask_list,   axis=0).astype(bool))


# ---------------------------------------------------------------------------
# Dispatch to existing K estimation methods
# ---------------------------------------------------------------------------

def _run_method(loader, per_args, device, search_mode):
    """
    Call the appropriate SelEx K estimation method.
    Returns (estimated_k_total, estimated_c_u).
    """
    if search_mode == 'autonovel':
        from methods.estimate_k.estimate_k_autonovel import estimate_k_autonovel
        k_total, c_u, _ = estimate_k_autonovel(loader, per_args, device)
    elif search_mode == 'agglo':
        from methods.estimate_k.estimate_k_agglo import estimate_k_agglo
        k_total, c_u, _ = estimate_k_agglo(loader, per_args, device)
    elif search_mode == 'spectral':
        from methods.estimate_k.estimate_k_spectral import estimate_k_spectral
        k_total, c_u, _ = estimate_k_spectral(loader, per_args, device)
    elif search_mode == 'xmeans':
        from methods.estimate_k.estimate_k_xgmeans import estimate_k_xmeans
        k_total, c_u, _ = estimate_k_xmeans(loader, per_args, device)
    elif search_mode == 'gmeans':
        from methods.estimate_k.estimate_k_xgmeans import estimate_k_gmeans
        k_total, c_u, _ = estimate_k_gmeans(loader, per_args, device)
    elif search_mode == 'brent':
        from methods.estimate_k.estimate_k import scipy_optimise
        k_total, c_u, _ = scipy_optimise(merge_test_loader=loader, args=per_args)
    else:  # binary (default)
        from methods.estimate_k.estimate_k import binary_search
        k_total, c_u, _ = binary_search(loader, per_args)
    return int(k_total), int(c_u)


# ---------------------------------------------------------------------------
# Main per-coarse estimation
# ---------------------------------------------------------------------------

def estimate_k_per_coarse(args, coarse_of, all_feats, all_labels, all_mask, device):
    """
    Estimate K independently for each unknown coarse category.

    Known coarse categories (all fine classes are labeled) provide probe features.
    For each unknown coarse c, build a per-coarse DataLoader from:
      - all probe features (labeled)
      - features of all fine classes in coarse c (unlabeled)
    Then call the chosen K estimation method.

    Returns {str(coarse_id): k_i} dict.
    """
    # Map each remapped label to its coarse ID
    unique_labels = np.unique(all_labels)
    label_to_coarse = {int(lbl): int(coarse_of[lbl]) for lbl in unique_labels}

    known_labels  = set(all_labels[all_mask].tolist())
    unknown_labels = set(all_labels[~all_mask].tolist())

    # Partition coarses into known (all fine = labeled) and unknown (all fine = unlabeled)
    known_coarses, unknown_coarses = set(), set()
    for lbl in unique_labels:
        c = label_to_coarse[int(lbl)]
        if int(lbl) in known_labels:
            known_coarses.add(c)
        else:
            unknown_coarses.add(c)

    mixed = known_coarses & unknown_coarses
    if mixed:
        raise ValueError(
            f"Coarse categories {mixed} contain both known and unknown fine classes. "
            "This violates the dataset structure assumption."
        )

    print(f"Known coarses: {sorted(known_coarses)}")
    print(f"Unknown coarses: {sorted(unknown_coarses)}")

    # Build probe feature set from known coarses (optionally subsampled)
    probe_coarse_prop = float(getattr(args, 'probe_coarse_prop', 1.0))
    known_coarse_list = sorted(known_coarses)
    n_probe = max(1, round(probe_coarse_prop * len(known_coarse_list)))
    probe_coarses = set(known_coarse_list[:n_probe])

    probe_idx = all_mask & np.array(
        [label_to_coarse[int(lbl)] in probe_coarses for lbl in all_labels], dtype=bool
    )
    probe_feats  = all_feats[probe_idx]
    probe_labels = all_labels[probe_idx]   # remapped global labels

    unique_probe_labels = sorted(set(probe_labels.tolist()))
    probe_idx_map = {int(lbl): i for i, lbl in enumerate(unique_probe_labels)}
    K_probe = len(unique_probe_labels)
    print(f"\nProbe: {K_probe} fine classes from {len(probe_coarses)} known coarse categories")

    k_per_coarse = {}

    for c in sorted(unknown_coarses):
        print(f"\n--- Coarse {c} ---")
        c_mask = ~all_mask & np.array(
            [label_to_coarse[int(lbl)] == c for lbl in all_labels], dtype=bool
        )
        unk_feats  = all_feats[c_mask]
        unk_labels = all_labels[c_mask]

        unique_unk = sorted(set(unk_labels.tolist()))
        unk_idx_map = {int(lbl): K_probe + i for i, lbl in enumerate(unique_unk)}
        K_c = len(unique_unk)

        print(f"  Known probe classes: {K_probe} | Unknown fine classes: {K_c}")

        # Build combined feature array with sequential labels
        feats_sub  = np.concatenate([probe_feats, unk_feats], axis=0)
        labels_sub = np.concatenate([
            np.array([probe_idx_map[int(lbl)] for lbl in probe_labels], dtype=np.int64),
            np.array([unk_idx_map[int(lbl)]   for lbl in unk_labels],   dtype=np.int64),
        ])
        mask_sub = np.concatenate([
            np.ones(len(probe_feats), dtype=bool),
            np.zeros(len(unk_feats),  dtype=bool),
        ])

        dataset = NumpyFeatureDataset(feats_sub, labels_sub, mask_sub)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

        per_args = copy(args)
        per_args.train_classes     = list(range(K_probe))
        per_args.unlabeled_classes = list(range(K_probe, K_probe + K_c))
        per_args.num_labeled_classes   = K_probe
        per_args.num_unlabeled_classes = K_c
        per_args.max_classes = getattr(args, 'max_classes', K_c * 3)

        try:
            _, estimated_c_u = _run_method(loader, per_args, device, args.search_mode)
        except Exception as e:
            print(f"  WARNING: K estimation failed ({e}). Falling back to K={K_c}.")
            estimated_c_u = K_c

        k_i = max(1, estimated_c_u)
        print(f"  Estimated K = {k_i}  (true = {K_c})")
        k_per_coarse[str(c)] = k_i

    return k_per_coarse


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-coarse K estimation for hierarchical category discovery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / features
    parser.add_argument('--dataset_name',      type=str, default='scars')
    parser.add_argument('--model_name',        type=str, default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits',    type=str2bool, default=True)
    parser.add_argument('--num_novel_classes', type=int, default=None)
    parser.add_argument('--root_dir',          type=str, default=feature_extract_dir)
    parser.add_argument('--batch_size',        type=int, default=128)
    parser.add_argument('--num_workers',       type=int, default=4)
    # Coarse labels
    parser.add_argument('--coarse_labels_path', type=str, required=True,
                        help='Path to JSON mapping fine_class_idx -> coarse_class_idx')
    parser.add_argument('--k_results_path',    type=str, required=True,
                        help='Output path for per-coarse K JSON {coarse_id: k}')
    parser.add_argument('--use_oracle_k',      type=str2bool, default=False,
                        help='Use true fine-class count per coarse instead of estimation')
    parser.add_argument('--probe_coarse_prop', type=float, default=1.0,
                        help='Fraction of known coarse categories used as probe pool')
    # K estimation method (reuse existing args)
    parser.add_argument('--search_mode', type=str, default='autonovel',
                        choices=['brent', 'binary', 'autonovel', 'agglo',
                                 'xmeans', 'gmeans', 'spectral'])
    parser.add_argument('--max_classes',          type=int,   default=20)
    parser.add_argument('--probe_prop',           type=float, default=0.2)
    parser.add_argument('--val_prop',             type=float, default=0.2)
    parser.add_argument('--outlier_tau',          type=float, default=0.01)
    parser.add_argument('--pca_dim',              type=int,   default=0)
    parser.add_argument('--agglo_linkages',       type=str,   default='ward,complete,average')
    parser.add_argument('--agglo_n_grid',         type=int,   default=25)
    parser.add_argument('--agglo_lo',             type=float, default=0.0)
    parser.add_argument('--agglo_hi',             type=float, default=10.0)
    parser.add_argument('--agglo_use_combined',   action='store_true')
    parser.add_argument('--agglo_max_samples',    type=int,   default=5000)
    parser.add_argument('--spectral_n_neighbors', type=int,   default=10)
    parser.add_argument('--final_cluster_method', type=str,   default='auto',
                        choices=['auto', 'kmeans', 'spectral', 'gmm'])

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # Load coarse label map
    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    # ---- Oracle mode: count unlabeled fine classes per coarse ----
    if args.use_oracle_k:
        k_per_coarse = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)
        k_per_coarse = {str(k): int(v) for k, v in k_per_coarse.items()}
        print(f"Oracle K per coarse: {k_per_coarse}")
    else:
        # ---- Load all pre-extracted features ----
        args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
        print(f'Loading features from {args.save_dir} ...')
        train_dataset, _, _, _ = get_datasets(
            args.dataset_name, train_transform=None, test_transform=None, args=args
        )
        all_feats, all_labels, all_mask = load_all_features(
            train_dataset, args.save_dir, args.batch_size, args.num_workers, device
        )
        print(f'Loaded {len(all_feats)} feature vectors (dim={all_feats.shape[1]})')
        print(f'  Labeled: {all_mask.sum()} | Unlabeled: {(~all_mask).sum()}')

        k_per_coarse = estimate_k_per_coarse(
            args, coarse_of, all_feats, all_labels, all_mask, device
        )

    # ---- Save results ----
    os.makedirs(os.path.dirname(os.path.abspath(args.k_results_path)), exist_ok=True)
    with open(args.k_results_path, 'w') as f:
        json.dump(k_per_coarse, f, indent=2)

    total_k_unknown = sum(k_per_coarse.values())
    print(f"\nPer-coarse K: {k_per_coarse}")
    print(f"Total unknown clusters: {total_k_unknown}")
    print(f"K estimation complete: {args.k_results_path}")


if __name__ == '__main__':
    main()
