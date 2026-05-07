"""
estimate_k_autonovel.py — AutoNovel k estimation method.

Reference:
    "AutoNovel: Automatically Discovering and Learning Novel Visual Categories"
    https://arxiv.org/pdf/2106.15252

Algorithm summary:
    1. Split known classes into probe (last probe_prop fraction) and training.
    2. Split probe into anchor (4/5) and validation (1/5).
    3. Run constrained semi-supervised k-means on D^l_r ∪ D^u, sweeping k.
       - Anchor probe instances are FIXED to their ground-truth cluster labels.
       - Validation probe + unlabelled instances are freely assigned.
    4. For each k compute:
       - ACC:  clustering accuracy on validation probe (Hungarian matching).
       - SIL:  silhouette score on truly unlabelled data.
    5. k* = round(mean(argmax ACC, argmax SIL)), then drop outlier clusters.
    6. estimated_k_total = C_l + (k* - C_l_r)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from config import feature_extract_dir
from data.get_datasets import get_class_splits, get_datasets
from methods.clustering.faster_mix_k_means_pytorch import K_Means
from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from project_utils.cluster_utils import cluster_acc

# Default hyperparameters (all configurable via CLI)
_PROBE_PROP  = 0.2   # fraction of C_l classes held out as probe
_VAL_PROP    = 0.2   # fraction of probe classes used for validation
_OUTLIER_TAU = 0.01  # drop unlabelled clusters with mass < tau * largest


# ---------------------------------------------------------------------------
# Class split helpers
# ---------------------------------------------------------------------------

def split_known_classes(train_classes, probe_prop=_PROBE_PROP, val_prop=_VAL_PROP):
    """
    Deterministically split known class IDs into training / anchor / validation
    probe subsets.

    Uses the *last* n_probe entries of the sorted class list so the split is
    reproducible without a random seed.

    Parameters
    ----------
    train_classes : iterable of int
    probe_prop    : float  fraction of C_l used as probe  (D^l_r / D^l)
    val_prop      : float  fraction of probe used for validation (D^l_rv / D^l_r)

    Returns
    -------
    training_cls : list[int]   classes NOT used for probe
    anchor_cls   : list[int]   anchor probe classes (D^l_ra)
    val_cls      : list[int]   validation probe classes (D^l_rv)
    """
    cls_list = sorted(train_classes)
    C_l = len(cls_list)

    n_probe  = max(2, round(probe_prop * C_l))
    n_val    = max(1, round(val_prop * n_probe))
    n_anchor = n_probe - n_val

    probe_cls    = cls_list[-n_probe:]
    training_cls = cls_list[:-n_probe]
    anchor_cls   = probe_cls[:n_anchor]
    val_cls      = probe_cls[n_anchor:]

    return training_cls, anchor_cls, val_cls


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def collect_all_features(train_loader, device):
    """
    Single pass over a MergedDataset FeatureVectorDataset train_loader.

    Returns
    -------
    all_feats  : np.ndarray  (N, D)  L2-normalised
    all_labels : np.ndarray  (N,)    transformed (sequential) class IDs
    all_mask   : np.ndarray  (N,)    bool — True = labelled instance
    """
    feats_list, labels_list, mask_list = [], [], []
    for feats, label, _, mask_lab in train_loader:
        feats = F.normalize(feats.float().to(device), dim=-1)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(label.cpu().numpy())
        mask_list.append(mask_lab.cpu().bool().numpy())
    all_feats  = np.concatenate(feats_list,  axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_mask   = np.concatenate(mask_list,   axis=0).astype(bool)
    return all_feats, all_labels, all_mask


def build_fit_mix_inputs(all_feats, all_labels, all_mask,
                         anchor_cls_transformed, val_cls_transformed):
    """
    Partition features into the three groups required by K_Means.fit_mix().

    Parameters
    ----------
    all_feats, all_labels, all_mask : outputs of collect_all_features()
    anchor_cls_transformed : set of transformed (sequential) anchor class IDs
    val_cls_transformed    : set of transformed (sequential) validation class IDs

    Returns
    -------
    anchor_feats   : np.ndarray  (n_a, D)
    anchor_targets : np.ndarray  (n_a,)   sequential 0..C_l_ra-1 for fit_mix
    val_feats      : np.ndarray  (n_v, D)
    val_targets    : np.ndarray  (n_v,)   transformed IDs (for cluster_acc)
    u_feats        : np.ndarray  (n_u, D)
    """
    anchor_mask = all_mask & np.isin(all_labels, list(anchor_cls_transformed))
    val_mask    = all_mask & np.isin(all_labels, list(val_cls_transformed))
    u_mask      = ~all_mask

    anchor_feats    = all_feats[anchor_mask]
    anchor_raw_ids  = all_labels[anchor_mask]
    val_feats       = all_feats[val_mask]
    val_targets     = all_labels[val_mask]
    u_feats         = all_feats[u_mask]

    # fit_mix requires l_targets to be 0-indexed within the labelled set
    sorted_anchor = sorted(anchor_cls_transformed)
    remap = {cls: idx for idx, cls in enumerate(sorted_anchor)}
    anchor_targets = np.array([remap[c] for c in anchor_raw_ids])

    return anchor_feats, anchor_targets, val_feats, val_targets, u_feats


# ---------------------------------------------------------------------------
# Single constrained k-means evaluation
# ---------------------------------------------------------------------------

def constrained_kmeans_eval(k, anchor_feats, anchor_targets,
                             val_feats, val_targets, u_feats,
                             device, n_init=1):
    """
    Run constrained k-means with k total clusters and evaluate quality.

    The 'free' side passed to fit_mix is concat(val_feats, u_feats) so that
    both the validation probe and unlabelled instances are freely assigned,
    matching the paper's specification.

    Parameters
    ----------
    k              : int   total number of clusters
    anchor_feats   : (n_a, D)
    anchor_targets : (n_a,)  0..C_l_ra-1
    val_feats      : (n_v, D)
    val_targets    : (n_v,)  for ACC evaluation
    u_feats        : (n_u, D)  truly unlabelled (for silhouette)
    device         : torch.device

    Returns
    -------
    acc   : float  clustering accuracy on validation probe
    sil   : float  silhouette score on unlabelled data (nan if < 2 clusters)
    u_preds : np.ndarray (n_u,)  cluster assignments for unlabelled data
    """
    n_anchor = len(anchor_feats)
    n_val    = len(val_feats)

    free_feats = np.concatenate([val_feats, u_feats], axis=0)

    t_anchor_feats   = torch.from_numpy(anchor_feats).float().to(device)
    t_anchor_targets = torch.from_numpy(anchor_targets).long().to(device)
    t_free_feats     = torch.from_numpy(free_feats).float().to(device)

    km = K_Means(k=k, n_init=n_init, random_state=0)
    km.fit_mix(u_feats=t_free_feats, l_feats=t_anchor_feats,
               l_targets=t_anchor_targets)

    labels_all = km.labels_.cpu().numpy()
    # Layout: [anchor (fixed, n_anchor), val (n_val), unlabelled (n_u)]
    val_preds = labels_all[n_anchor: n_anchor + n_val]
    u_preds   = labels_all[n_anchor + n_val:]

    acc = cluster_acc(val_targets.astype(int), val_preds.astype(int))

    unique_clusters = np.unique(u_preds)
    if len(unique_clusters) < 2:
        sil = np.nan
    else:
        sample_size = min(len(u_preds), 5000)
        sil = silhouette_score(u_feats, u_preds, metric='euclidean',
                               sample_size=sample_size, random_state=0)

    return acc, sil, u_preds


# ---------------------------------------------------------------------------
# Outlier cluster dropping
# ---------------------------------------------------------------------------

def drop_outlier_clusters(k_star, cached_u_preds, tau=_OUTLIER_TAU):
    """
    Drop unlabelled clusters whose mass is < tau * largest_cluster_mass.

    Parameters
    ----------
    k_star         : int    current k estimate
    cached_u_preds : np.ndarray (n_u,)  cluster assignments for unlabelled data
    tau            : float

    Returns
    -------
    adjusted_k : int
    """
    _, counts = np.unique(cached_u_preds, return_counts=True)
    largest   = counts.max()
    n_dropped = int((counts < tau * largest).sum())
    adjusted  = max(1, k_star - n_dropped)
    print(f"  Outlier dropping: {n_dropped} cluster(s) dropped "
          f"(tau={tau}), adjusted k*={adjusted}")
    return adjusted


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_k_autonovel(train_loader, args, device):
    """
    Run the full AutoNovel k estimation algorithm.

    Prints "Optimal K is {k}" which is parsed by run_pipeline.py.

    Parameters
    ----------
    train_loader : DataLoader over a MergedDataset FeatureVectorDataset
    args         : namespace with train_classes, unlabeled_classes,
                   num_labeled_classes, max_classes,
                   and optionally probe_prop, val_prop, outlier_tau
    device       : torch.device

    Returns
    -------
    estimated_k_total : int  (also printed)
    estimated_C_u     : int  estimated number of novel/unlabelled classes
    """
    probe_prop  = getattr(args, 'probe_prop',  _PROBE_PROP)
    val_prop    = getattr(args, 'val_prop',    _VAL_PROP)
    outlier_tau = getattr(args, 'outlier_tau', _OUTLIER_TAU)

    # ------------------------------------------------------------------
    # 1. Collect all training features
    # ------------------------------------------------------------------
    print("AutoNovel k-estimation: collecting features...")
    all_feats, all_labels, all_mask = collect_all_features(train_loader, device)

    # ------------------------------------------------------------------
    # 1b. Optional PCA (fit on labelled, transform both)
    # ------------------------------------------------------------------
    pca     = None
    pca_dim = int(getattr(args, 'pca_dim', 0))
    if pca_dim > 0:
        from methods.estimate_k.estimate_k_agglo import apply_pca_and_norm
        l_pca, u_pca, pca = apply_pca_and_norm(
            all_feats[all_mask], all_feats[~all_mask], pca_dim
        )
        all_feats_new = np.empty((len(all_feats), l_pca.shape[1]), dtype=np.float32)
        all_feats_new[all_mask]  = l_pca
        all_feats_new[~all_mask] = u_pca
        all_feats = all_feats_new
        print(f"  PCA applied: features are now {all_feats.shape[1]}-dimensional")

    # ------------------------------------------------------------------
    # 2. Class splits
    # ------------------------------------------------------------------
    _, anchor_cls, val_cls = split_known_classes(
        args.train_classes, probe_prop=probe_prop, val_prop=val_prop
    )
    C_l    = args.num_labeled_classes
    C_l_r  = len(anchor_cls) + len(val_cls)
    C_l_ra = len(anchor_cls)
    C_l_rv = len(val_cls)
    print(f"  C_l={C_l}, probe={C_l_r} (anchor={C_l_ra}, val={C_l_rv})")

    # Map original class IDs → sequential transformed IDs used in the loader
    # get_datasets uses: {cls: i for i, cls in enumerate(train_classes + unlabeled_classes)}
    all_cls_sorted   = sorted(args.train_classes) + sorted(args.unlabeled_classes)
    transform_map    = {cls: i for i, cls in enumerate(all_cls_sorted)}
    anchor_cls_t     = {transform_map[c] for c in anchor_cls}
    val_cls_t        = {transform_map[c] for c in val_cls}

    # ------------------------------------------------------------------
    # 3. Build feature partitions
    # ------------------------------------------------------------------
    anchor_feats, anchor_targets, val_feats, val_targets, u_feats = \
        build_fit_mix_inputs(all_feats, all_labels, all_mask,
                             anchor_cls_t, val_cls_t)

    print(f"  anchor instances={len(anchor_feats)}, "
          f"val instances={len(val_feats)}, "
          f"unlabelled instances={len(u_feats)}")

    # ------------------------------------------------------------------
    # 4. Sweep k
    # ------------------------------------------------------------------
    k_min = C_l_ra          # fit_mix requires at least one center per anchor class
    k_max = args.max_classes
    print(f"  Sweeping k in [{k_min}, {k_max}]...")

    acc_scores  = {}
    sil_scores  = {}
    cached_preds = {}   # store u_preds for each k to avoid re-running at k*

    for k in range(k_min, k_max + 1):
        acc, sil, u_preds = constrained_kmeans_eval(
            k, anchor_feats, anchor_targets,
            val_feats, val_targets, u_feats, device
        )
        acc_scores[k]   = acc
        sil_scores[k]   = sil
        cached_preds[k] = u_preds
        print(f"  k={k:4d}: val_acc={acc:.4f}  sil={sil:.4f}")

    # ------------------------------------------------------------------
    # 5. Select k*
    # ------------------------------------------------------------------
    k_acc_star = max(acc_scores, key=lambda k: acc_scores[k])

    valid_sil  = {k: v for k, v in sil_scores.items() if not np.isnan(v)}
    if valid_sil:
        k_sil_star = max(valid_sil, key=lambda k: valid_sil[k])
    else:
        k_sil_star = k_acc_star
        print("  Warning: all silhouette scores were NaN; "
              "falling back to k_acc* for k_sil*")

    k_star = round((k_acc_star + k_sil_star) / 2.0)
    print(f"  k_acc*={k_acc_star}, k_sil*={k_sil_star}, k*={k_star}")

    # ------------------------------------------------------------------
    # 6. Outlier cluster dropping
    #    Re-use cached u_preds from the sweep if available, otherwise re-run
    # ------------------------------------------------------------------
    if k_star in cached_preds:
        u_preds_star = cached_preds[k_star]
    else:
        # k* was rounded to a value not in the sweep (shouldn't normally happen)
        _, _, u_preds_star = constrained_kmeans_eval(
            k_star, anchor_feats, anchor_targets,
            val_feats, val_targets, u_feats, device
        )

    k_star = drop_outlier_clusters(k_star, u_preds_star, tau=outlier_tau)

    # ------------------------------------------------------------------
    # 7. Convert to total-k space
    # ------------------------------------------------------------------
    estimated_C_u     = max(1, k_star - C_l_r)
    estimated_k_total = C_l + estimated_C_u

    print(f"Optimal K is {estimated_k_total}")

    # ------------------------------------------------------------------
    # 8. Final clustering on all training features for evaluation
    # ------------------------------------------------------------------
    from methods.estimate_k.estimate_k_tuning import _cluster_with_method
    cluster_method = getattr(args, 'final_cluster_method', 'auto')
    n_neighbors    = int(getattr(args, 'spectral_n_neighbors', 10))
    resolved_method = 'kmeans' if cluster_method == 'auto' else cluster_method
    print(f"  Fitting final clustering (method={resolved_method}, k={estimated_k_total}) on all training features...")
    preds, fitted_model = _cluster_with_method(all_feats, estimated_k_total, resolved_method,
                                               n_neighbors=n_neighbors, return_model=True)
    result = dict(
        all_feats=all_feats,
        all_labels=all_labels.astype(np.int64),
        mask_lab=all_mask,
        preds=preds,
        pca=pca,
        fitted_model=fitted_model,
    )
    return estimated_k_total, estimated_C_u, result


# ---------------------------------------------------------------------------
# CLI entry point (mirrors estimate_k.py structure)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='AutoNovel k estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_classes', default=1000, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_exp_id', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino')
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--probe_prop', type=float, default=_PROBE_PROP,
                        help='Fraction of known classes held out as probe')
    parser.add_argument('--val_prop', type=float, default=_VAL_PROP,
                        help='Fraction of probe classes used for validation')
    parser.add_argument('--outlier_tau', type=float, default=_OUTLIER_TAU,
                        help='Outlier cluster dropping threshold')
    parser.add_argument('--pca_dim', type=int, default=0,
                        help='PCA output dimensionality (0 = skip PCA)')
    parser.add_argument('--spectral_n_neighbors', type=int, default=10,
                        help='Number of neighbors for spectral clustering')
    parser.add_argument('--final_cluster_method', type=str, default='auto',
                        choices=['auto', 'kmeans', 'spectral', 'gmm'],
                        help='Clustering algorithm for final result. '
                             '"auto" uses kmeans (historical default).')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
    args = get_class_splits(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    if args.warmup_model_exp_id is not None:
        args.save_dir += '_' + args.warmup_model_exp_id
        print(f'Using features from experiment: {args.warmup_model_exp_id}')
    else:
        print(f'Using pretrained {args.model_name} features...')

    train_transform, test_transform = None, None
    train_dataset, _, _, _ = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )
    train_dataset = FeatureVectorDataset(
        base_dataset=train_dataset,
        feature_root=os.path.join(args.save_dir, 'train'),
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    estimate_k_autonovel(train_loader, args, device)  # result dict ignored at CLI
