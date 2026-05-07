"""
estimate_k_agglo.py — Agglomerative clustering k estimation method.

Algorithm:
    1. Load labelled and unlabelled training features from disk.
    2. Build a candidate grid over (linkage, distance threshold, pca_dim).
    3. Select the best configuration using one of three criteria:
         labelled_k   – recover known k on labelled features (default)
         test_cluster – old-class ACC on test split via nearest-centroid
         combined     – weighted combination of both
    4. Apply the best configuration to unlabelled (or combined) data to
       estimate the number of novel classes.

The tuning criterion is controlled by --tuning_criterion (default: labelled_k),
which preserves the original behaviour exactly when left at its default.
"""

import argparse
import itertools
import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from config import feature_extract_dir
from data.get_datasets import get_class_splits, get_datasets
from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from methods.estimate_k.estimate_k_autonovel import collect_all_features
from methods.estimate_k.estimate_k_tuning import (
    CRITERION_LABELLED, CRITERION_TEST, CRITERION_COMBINED,
    _cluster_with_method,
    collect_test_features_np,
    select_best_config,
)

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def collect_and_split_features(train_loader, device):
    """
    Load all training features and split into labelled / unlabelled.

    Returns
    -------
    l_feats   : np.ndarray  (n_l, D)  labelled features (L2-normalised)
    l_targets : np.ndarray  (n_l,)    transformed class IDs
    u_feats   : np.ndarray  (n_u, D)  unlabelled features
    u_targets : np.ndarray  (n_u,)    transformed class IDs for unlabelled
    all_mask  : np.ndarray  (N,) bool  True = labelled instance
    """
    all_feats, all_labels, all_mask = collect_all_features(train_loader, device)
    l_feats   = all_feats[all_mask]
    l_targets = all_labels[all_mask]
    u_feats   = all_feats[~all_mask]
    u_targets = all_labels[~all_mask]
    return l_feats, l_targets, u_feats, u_targets, all_mask


def apply_pca_and_norm(l_feats, u_feats, pca_dim, seed=0, verbose=True):
    """
    Fit PCA on labelled features; transform both; L2-normalise.

    Parameters
    ----------
    l_feats  : (n_l, D)
    u_feats  : (n_u, D)
    pca_dim  : int
    verbose  : bool  if True, print a summary line (suppress for cached calls)

    Returns
    -------
    Xp_l : (n_l, pca_dim)  float32
    Xp_u : (n_u, pca_dim)  float32
    pca  : fitted PCA object
    """
    D = l_feats.shape[1]
    n_comp = int(min(pca_dim, D, max(1, l_feats.shape[0] - 1)))
    pca = PCA(n_components=n_comp, random_state=seed)
    Xp_l = pca.fit_transform(l_feats)
    Xp_l = normalize(Xp_l, norm='l2', axis=1).astype(np.float32)

    Xp_u = pca.transform(u_feats)
    Xp_u = normalize(Xp_u, norm='l2', axis=1).astype(np.float32)

    if verbose:
        print(f"  PCA: {D}d → {n_comp}d "
              f"(var explained: {pca.explained_variance_ratio_.sum():.3f})")
    return Xp_l, Xp_u, pca


def subsample_labelled(l_feats, l_targets, max_samples, seed=0):
    """
    Stratified subsample of labelled features to cap O(N²) agglomerative cost.

    Takes at most floor(max_samples / n_classes) per class, then fills
    remaining quota from classes with the most data.

    Returns
    -------
    sub_feats   : (≤max_samples, D)
    sub_targets : (≤max_samples,)
    """
    if len(l_feats) <= max_samples:
        return l_feats, l_targets

    rng = np.random.default_rng(seed)
    classes = np.unique(l_targets)
    n_cls = len(classes)
    per_class = max(1, max_samples // n_cls)

    idxs = []
    for c in classes:
        c_idxs = np.where(l_targets == c)[0]
        take = min(per_class, len(c_idxs))
        chosen = rng.choice(c_idxs, size=take, replace=False)
        idxs.append(chosen)

    idxs = np.concatenate(idxs)
    # Fill quota if we have leftover capacity
    if len(idxs) < max_samples:
        remaining = np.setdiff1d(np.arange(len(l_feats)), idxs)
        extra = min(max_samples - len(idxs), len(remaining))
        idxs = np.concatenate([idxs, rng.choice(remaining, size=extra, replace=False)])

    idxs = np.sort(idxs)
    return l_feats[idxs], l_targets[idxs]


# ---------------------------------------------------------------------------
# Agglomerative clustering
# ---------------------------------------------------------------------------

def agglo_cluster(X, distance_threshold, linkage, metric='euclidean'):
    """
    Run agglomerative clustering with a distance threshold.

    Parameters
    ----------
    metric : 'euclidean' (default) | 'cosine' | any metric accepted by
             AgglomerativeClustering.  Note: 'ward' linkage requires 'euclidean'.

    Returns
    -------
    labels : np.ndarray (N,) int64
    k_hat  : int  number of clusters found
    """
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), 0
    if X.shape[0] == 1:
        return np.zeros(1, dtype=np.int64), 1

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        linkage=linkage,
        metric=metric,
    )
    labels = model.fit_predict(X).astype(np.int64)
    k_hat  = int(np.unique(labels).size)
    return labels, k_hat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_pca_candidate(l_feats_raw, u_feats_raw, pca_dim):
    """Apply PCA (or just cast) for a given pca_dim candidate."""
    if pca_dim > 0:
        return apply_pca_and_norm(l_feats_raw, u_feats_raw, pca_dim)
    else:
        Xp_l = l_feats_raw.astype(np.float32)
        Xp_u = u_feats_raw.astype(np.float32)
        return Xp_l, Xp_u, None


def _build_agglo_preds_full(Xp_l, Xp_u, labels_u):
    """
    Build a full preds array (N_l + N_u,) when clustering was done on
    unlabelled features only.

    Labelled points are assigned to the nearest centroid of the unlabelled
    clusters, so that X_all = vstack([Xp_l, Xp_u]) has predictions for every
    sample — needed for nearest-centroid test evaluation.
    """
    unique_u = np.unique(labels_u)
    centroids = np.array([
        Xp_u[labels_u == i].mean(axis=0) for i in unique_u
    ], dtype=np.float32)
    diffs    = Xp_l[:, None, :] - centroids[None, :, :]
    sq_dists = (diffs ** 2).sum(axis=-1)
    labels_l = unique_u[np.argmin(sq_dists, axis=1)].astype(np.int64)

    return np.concatenate([labels_l, labels_u.astype(np.int64)], axis=0)


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_k_agglo(train_loader, args, device=None, test_loader=None):
    """
    Run the full agglomerative k estimation algorithm.

    Prints "Optimal K is {k}" which is parsed by run_pipeline.py.

    Parameters
    ----------
    train_loader : DataLoader over a MergedDataset FeatureVectorDataset
    args         : namespace (see required fields below)
    device       : torch.device or None (only used for feature loading)
    test_loader  : DataLoader for the test split; required when
                   args.tuning_criterion is 'test_cluster' or 'combined'

    Required args fields
    --------------------
    train_classes, unlabeled_classes, num_labeled_classes,
    pca_dim, agglo_linkages, agglo_n_grid, agglo_lo, agglo_hi,
    agglo_use_combined, agglo_max_samples

    Optional args fields (new)
    --------------------------
    tuning_criterion       : 'labelled_k' (default) | 'test_cluster' | 'combined'
    tuning_weight          : float  (weight for labelled_k in combined, default 0.5)
    tune_pca               : bool   (include pca_dim in tuning grid, default False)
    pca_dim_candidates     : str    (comma-separated dims, default str(pca_dim))
    agglo_max_total_samples: int    (cap X_all size for combined mode, default 10000)

    Returns
    -------
    estimated_k_total : int
    estimated_C_u     : int
    result            : dict with keys all_feats, all_labels, mask_lab, preds, pca
    """
    import torch
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pca_dim              = int(getattr(args, 'pca_dim', 64))
    linkages             = [s.strip() for s in getattr(args, 'agglo_linkages', 'ward,complete,average').split(',')]
    n_grid               = int(getattr(args, 'agglo_n_grid', 25))
    lo                   = float(getattr(args, 'agglo_lo', 0.0))
    hi                   = float(getattr(args, 'agglo_hi', 10.0))
    use_combined         = bool(getattr(args, 'agglo_use_combined', False))
    max_samples          = int(getattr(args, 'agglo_max_samples', 5000))
    max_total_samples    = int(getattr(args, 'agglo_max_total_samples', 10_000))

    tuning_criterion  = getattr(args, 'tuning_criterion', CRITERION_LABELLED)
    tuning_weight     = float(getattr(args, 'tuning_weight', 0.5))
    tune_pca          = bool(getattr(args, 'tune_pca', False))
    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', str(pca_dim))

    agglo_metric = getattr(args, 'agglo_metric', 'euclidean')

    pca_dims = (
        [int(x.strip()) for x in pca_dim_cands_str.split(',')]
        if tune_pca else [pca_dim]
    )

    # ------------------------------------------------------------------
    # 1. Load raw features
    # ------------------------------------------------------------------
    print("Agglo k-estimation: collecting features...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask = \
        collect_and_split_features(train_loader, device)
    l_feats_raw = l_feats_raw.astype(np.float32)
    u_feats_raw = u_feats_raw.astype(np.float32)
    print(f"  labelled={len(l_feats_raw)}, unlabelled={len(u_feats_raw)}, "
          f"dim={l_feats_raw.shape[1]}")

    k_true = args.num_labeled_classes

    # ------------------------------------------------------------------
    # 2. Pre-collect test features if needed
    # ------------------------------------------------------------------
    test_feats_raw = None
    test_labels    = None
    if tuning_criterion in (CRITERION_TEST, CRITERION_COMBINED):
        if test_loader is None:
            raise ValueError(
                "estimate_k_agglo: test_loader must be provided when "
                f"tuning_criterion={tuning_criterion!r}. "
                "Pass test_loader from estimate_k.py."
            )
        print("  Pre-collecting test features for tuning criterion...")
        test_feats_raw, test_labels = collect_test_features_np(
            test_loader, pca=None, device=device
        )

    # ------------------------------------------------------------------
    # 3. Pre-cache PCA transforms (one fit per unique pca_dim, not per candidate)
    # ------------------------------------------------------------------
    print(f"  Pre-computing PCA for {len(set(pca_dims))} unique dim(s): "
          f"{sorted(set(pca_dims))}")
    _pca_cache = {}
    for pd in sorted(set(pca_dims)):
        _pca_cache[pd] = _apply_pca_candidate(l_feats_raw, u_feats_raw, pd)

    # ------------------------------------------------------------------
    # 4. Guard: ward linkage requires euclidean metric
    # ------------------------------------------------------------------
    if agglo_metric != 'euclidean':
        ward_removed = [lnk for lnk in linkages if lnk == 'ward']
        if ward_removed:
            print(f"  WARNING: 'ward' linkage requires euclidean metric — "
                  f"removing it from linkage candidates (metric={agglo_metric!r})")
            linkages = [lnk for lnk in linkages if lnk != 'ward']
        if not linkages:
            raise ValueError(
                f"--agglo_metric={agglo_metric!r} is incompatible with all specified "
                f"linkages ({getattr(args, 'agglo_linkages', 'ward,complete,average')}). "
                "Use --agglo_linkages complete,average when using cosine metric."
            )

    # ------------------------------------------------------------------
    # 5. Build param grid
    #    Sort threshold ascending so that ties are broken by smaller threshold
    #    (preserves the original tune_on_labelled preference for simpler models).
    # ------------------------------------------------------------------
    threshold_grid = sorted(np.linspace(lo, hi, n_grid).tolist())
    param_grid = [
        {'linkage': lnk, 'threshold': thr, 'pca_dim': pd}
        for lnk, thr, pd in itertools.product(linkages, threshold_grid, pca_dims)
    ]
    print(f"  Tuning: {len(linkages)} linkage(s) × {n_grid} thresholds × "
          f"{len(pca_dims)} pca_dim(s) = {len(param_grid)} candidates "
          f"(criterion={tuning_criterion}, k_true={k_true})")

    # ------------------------------------------------------------------
    # 6. Define labelled_fn and full_fn closures (use PCA cache)
    # ------------------------------------------------------------------

    def labelled_fn(params):
        """Agglo on labelled-only features → k_hat_l."""
        Xp_l, _Xp_u, _ = _pca_cache[params['pca_dim']]
        l_sub, _ = subsample_labelled(Xp_l, l_targets, max_samples)
        _, k_hat_l = agglo_cluster(l_sub, params['threshold'], params['linkage'],
                                   metric=agglo_metric)
        return k_hat_l

    def full_fn(params):
        """Agglo on X_all (or Xp_u) → k_hat, full preds, X_all, pca."""
        Xp_l, Xp_u, pca = _pca_cache[params['pca_dim']]
        X_all = np.vstack([Xp_l, Xp_u])

        if use_combined:
            # Cap size to avoid O(N²) blowup on large datasets (e.g. cifar10 ~50k)
            if len(X_all) > max_total_samples:
                rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(len(X_all), size=max_total_samples, replace=False))
                X_fit = X_all[idx]
            else:
                X_fit = X_all
            labels_fit, k_hat = agglo_cluster(
                X_fit, params['threshold'], params['linkage'], metric=agglo_metric
            )
            if len(X_fit) < len(X_all):
                # Assign all points via nearest centroid from the subsampled clustering
                unique_c = np.unique(labels_fit)
                centroids = np.array(
                    [X_fit[labels_fit == c].mean(axis=0) for c in unique_c],
                    dtype=np.float32,
                )
                diffs    = X_all[:, None, :] - centroids[None, :, :]
                preds    = unique_c[np.argmin((diffs ** 2).sum(axis=-1), axis=1)].astype(np.int64)
            else:
                preds = labels_fit.astype(np.int64)
        else:
            labels_u, k_hat_u = agglo_cluster(
                Xp_u, params['threshold'], params['linkage'], metric=agglo_metric
            )
            k_hat = k_true + max(1, k_hat_u)
            preds = _build_agglo_preds_full(Xp_l, Xp_u, labels_u)

        return k_hat, preds, X_all, pca, None

    # ------------------------------------------------------------------
    # 7. Run selection
    # ------------------------------------------------------------------
    best = select_best_config(
        param_grid=param_grid,
        k_true=k_true,
        criterion=tuning_criterion,
        weight=tuning_weight,
        labelled_fn=labelled_fn,
        full_fn=full_fn,
        test_feats_raw=test_feats_raw,
        test_labels=test_labels,
        num_labeled_classes=args.num_labeled_classes,
    )

    import json as _json
    print(f"Best params: {_json.dumps(best.params)}")

    estimated_k_total = best.k_hat
    if use_combined:
        estimated_C_u = max(1, estimated_k_total - k_true)
    else:
        estimated_C_u = max(1, estimated_k_total - k_true)

    print(f"  estimated_C_u={estimated_C_u}, estimated_k_total={estimated_k_total}")
    print(f"Optimal K is {estimated_k_total}")

    # ------------------------------------------------------------------
    # 8. Build result dict in original loader order
    #    best.X_all is shaped (N_l+N_u, D): first N_l rows = labelled,
    #    rest = unlabelled.  Scatter into all_mask ordering.
    # ------------------------------------------------------------------
    n_l      = int(all_mask.sum())
    feat_dim = best.X_all.shape[1]

    Xp_full = np.empty((len(all_mask), feat_dim), dtype=np.float32)
    Xp_full[all_mask]  = best.X_all[:n_l]
    Xp_full[~all_mask] = best.X_all[n_l:]

    all_labels_full = np.empty(len(all_mask), dtype=np.int64)
    all_labels_full[all_mask]  = l_targets.astype(np.int64)
    all_labels_full[~all_mask] = u_targets.astype(np.int64)

    preds_full = np.empty(len(all_mask), dtype=np.int64)
    preds_full[all_mask]  = best.preds[:n_l]
    preds_full[~all_mask] = best.preds[n_l:]

    result = dict(
        all_feats=Xp_full,
        all_labels=all_labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=best.pca,
        fitted_model=best.fitted_model,
    )
    return estimated_k_total, estimated_C_u, result


# ---------------------------------------------------------------------------
# Fixed-K variant: tune linkage/metric/pca_dim at oracle K
# ---------------------------------------------------------------------------

def estimate_k_agglo_fixed_k(train_loader, args, device=None, test_loader=None, k=None):
    """
    Agglomerative clustering with a *fixed* K (oracle K known).

    Instead of searching for K via a distance threshold, we set n_clusters=k
    directly and tune over linkage × metric × pca_dim.

    The tuning criterion is always test_cluster (labelled_fn is not needed
    because we are not searching for K).

    Parameters
    ----------
    k : int  the fixed number of clusters (total = labeled + unlabeled)

    Returns
    -------
    k : int  (same as input)
    k_novel : int
    result  : dict with keys all_feats, all_labels, mask_lab, preds, pca
    """
    import json as _json
    import torch
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if k is None:
        raise ValueError("estimate_k_agglo_fixed_k: k must be provided (use --fixed_k or --use_true_k)")

    pca_dim           = int(getattr(args, 'pca_dim', 64))
    linkages          = [s.strip() for s in getattr(args, 'agglo_linkages', 'ward,complete,average').split(',')]
    max_total_samples = int(getattr(args, 'agglo_max_total_samples', 10_000))
    tune_pca          = bool(getattr(args, 'tune_pca', False))
    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', str(pca_dim))
    agglo_metric      = getattr(args, 'agglo_metric', 'euclidean')
    tuning_weight     = float(getattr(args, 'tuning_weight', 0.5))

    pca_dims = (
        [int(x.strip()) for x in pca_dim_cands_str.split(',')]
        if tune_pca else [pca_dim]
    )

    # ------------------------------------------------------------------
    # 1. Load raw features
    # ------------------------------------------------------------------
    print(f"Agglo fixed-K k-estimation (k={k}): collecting features...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask = \
        collect_and_split_features(train_loader, device)
    l_feats_raw = l_feats_raw.astype(np.float32)
    u_feats_raw = u_feats_raw.astype(np.float32)
    print(f"  labelled={len(l_feats_raw)}, unlabelled={len(u_feats_raw)}, "
          f"dim={l_feats_raw.shape[1]}")

    k_true = args.num_labeled_classes
    k_novel = max(1, k - k_true)

    # ------------------------------------------------------------------
    # 2. Always use test_cluster criterion — collect test features
    # ------------------------------------------------------------------
    if test_loader is None:
        raise ValueError(
            "estimate_k_agglo_fixed_k: test_loader must be provided. "
            "Fixed-K tuning always uses test_cluster criterion."
        )
    print("  Pre-collecting test features for test_cluster criterion...")
    test_feats_raw, test_labels = collect_test_features_np(
        test_loader, pca=None, device=device
    )

    # ------------------------------------------------------------------
    # 3. Guard: ward linkage requires euclidean metric
    # ------------------------------------------------------------------
    if agglo_metric != 'euclidean':
        linkages = [lnk for lnk in linkages if lnk != 'ward']
        if not linkages:
            raise ValueError(
                f"--agglo_metric={agglo_metric!r} is incompatible with all specified linkages. "
                "Use --agglo_linkages complete,average with cosine metric."
            )

    # ------------------------------------------------------------------
    # 4. Pre-cache PCA transforms
    # ------------------------------------------------------------------
    metrics_list = [agglo_metric]
    # For multi-metric support, also allow cosine if euclidean is default
    # (controlled by the metric arg; we tune over linkage × pca_dim × metric)
    # Build param_grid: linkage × pca_dim (single metric from args)
    print(f"  Pre-computing PCA for {len(set(pca_dims))} unique dim(s): "
          f"{sorted(set(pca_dims))}")
    _pca_cache = {}
    for pd in sorted(set(pca_dims)):
        _pca_cache[pd] = _apply_pca_candidate(l_feats_raw, u_feats_raw, pd)

    # ------------------------------------------------------------------
    # 5. Build param grid: linkage × pca_dim (metric fixed from args)
    # ------------------------------------------------------------------
    param_grid = [
        {'linkage': lnk, 'pca_dim': pd}
        for lnk, pd in itertools.product(linkages, pca_dims)
    ]
    print(f"  Fixed-K tuning: {len(linkages)} linkage(s) × {len(pca_dims)} pca_dim(s) "
          f"= {len(param_grid)} candidates (k={k}, metric={agglo_metric}, "
          f"criterion=test_cluster)")

    # ------------------------------------------------------------------
    # 6. Define full_fn (no labelled_fn — test_cluster criterion only)
    # ------------------------------------------------------------------

    def full_fn(params):
        """AgglomerativeClustering with fixed n_clusters=k."""
        Xp_l, Xp_u, pca = _pca_cache[params['pca_dim']]
        X_all = np.vstack([Xp_l, Xp_u])

        # Cap size to avoid O(N²) blowup
        if len(X_all) > max_total_samples:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(len(X_all), size=max_total_samples, replace=False))
            X_fit = X_all[idx]
        else:
            X_fit = X_all

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=params['linkage'],
            metric=agglo_metric,
        )
        labels_fit = model.fit_predict(X_fit).astype(np.int64)

        if len(X_fit) < len(X_all):
            # Centroid-assign remaining points
            unique_c = np.unique(labels_fit)
            centroids = np.array(
                [X_fit[labels_fit == c].mean(axis=0) for c in unique_c],
                dtype=np.float32,
            )
            diffs = X_all[:, None, :] - centroids[None, :, :]
            preds = unique_c[np.argmin((diffs ** 2).sum(axis=-1), axis=1)].astype(np.int64)
        else:
            preds = labels_fit

        print(f"    agglo fixed-K: linkage={params['linkage']}, "
              f"pca_dim={params['pca_dim']}, metric={agglo_metric}, "
              f"actual_k={len(np.unique(preds))}")
        return k, preds, X_all, pca, None

    # ------------------------------------------------------------------
    # 7. Run selection (test_cluster criterion, labelled_fn=None)
    # ------------------------------------------------------------------
    best = select_best_config(
        param_grid=param_grid,
        k_true=k_true,
        criterion=CRITERION_TEST,
        weight=tuning_weight,
        labelled_fn=None,
        full_fn=full_fn,
        test_feats_raw=test_feats_raw,
        test_labels=test_labels,
        num_labeled_classes=args.num_labeled_classes,
    )

    print(f"Best params: {_json.dumps(best.params)}")
    print(f"  fixed k={k}, k_novel={k_novel}")
    print(f"Optimal K is {k}")

    # ------------------------------------------------------------------
    # 8. Build result dict in original loader order
    # ------------------------------------------------------------------
    n_l = int(all_mask.sum())
    feat_dim = best.X_all.shape[1]

    Xp_full = np.empty((len(all_mask), feat_dim), dtype=np.float32)
    Xp_full[all_mask]  = best.X_all[:n_l]
    Xp_full[~all_mask] = best.X_all[n_l:]

    all_labels_full = np.empty(len(all_mask), dtype=np.int64)
    all_labels_full[all_mask]  = l_targets.astype(np.int64)
    all_labels_full[~all_mask] = u_targets.astype(np.int64)

    preds_full = np.empty(len(all_mask), dtype=np.int64)
    preds_full[all_mask]  = best.preds[:n_l]
    preds_full[~all_mask] = best.preds[n_l:]

    result = dict(
        all_feats=Xp_full,
        all_labels=all_labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=best.pca,
        fitted_model=best.fitted_model,
    )
    return k, k_novel, result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch

    parser = argparse.ArgumentParser(
        description='Agglomerative k estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--batch_size',            default=128,  type=int)
    parser.add_argument('--num_workers',           default=8,    type=int)
    parser.add_argument('--root_dir',              type=str,     default=feature_extract_dir)
    parser.add_argument('--warmup_model_exp_id',   type=str,     default=None)
    parser.add_argument('--model_name',            type=str,     default='vit_dino')
    parser.add_argument('--dataset_name',          type=str,     default='cifar10')
    parser.add_argument('--prop_train_labels',     type=float,   default=0.5)
    # Agglo-specific
    parser.add_argument('--pca_dim',               type=int,     default=64,
                        help='PCA output dimensionality (0 = skip PCA)')
    parser.add_argument('--agglo_linkages',        type=str,     default='ward,complete,average',
                        help='Comma-separated linkage types to try during tuning')
    parser.add_argument('--agglo_n_grid',          type=int,     default=25,
                        help='Number of threshold values in grid')
    parser.add_argument('--agglo_lo',              type=float,   default=0.0,
                        help='Lower bound of threshold grid')
    parser.add_argument('--agglo_hi',              type=float,   default=10.0,
                        help='Upper bound of threshold grid')
    parser.add_argument('--agglo_use_combined',    action='store_true',
                        help='Cluster labelled+unlabelled together for estimation')
    parser.add_argument('--agglo_max_samples',        type=int, default=5000,
                        help='Max total labelled samples for labelled_fn tuning (O(N²) guard)')
    parser.add_argument('--agglo_max_total_samples',  type=int, default=10_000,
                        help='Max total (lab+unlab) samples for combined full_fn agglo (O(N²) guard)')
    # Tuning framework
    parser.add_argument('--tuning_criterion',      type=str,     default='labelled_k',
                        choices=['labelled_k', 'test_cluster', 'combined'],
                        help='Criterion for hyperparameter selection')
    parser.add_argument('--tuning_weight',         type=float,   default=0.5,
                        help='Weight for labelled_k score in combined criterion')
    parser.add_argument('--tune_pca',              action='store_true',
                        help='Include pca_dim as a tunable hyperparameter')
    parser.add_argument('--pca_dim_candidates',    type=str,     default='32,64,128',
                        help='Comma-separated PCA dims to try when --tune_pca is set')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
    args = get_class_splits(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    if args.warmup_model_exp_id is not None:
        args.save_dir += '_' + args.warmup_model_exp_id

    train_transform, test_transform = None, None
    train_dataset, test_dataset, _, _ = get_datasets(
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

    test_loader = None
    if args.tuning_criterion in ('test_cluster', 'combined'):
        test_dataset = FeatureVectorDataset(
            base_dataset=test_dataset,
            feature_root=os.path.join(args.save_dir, 'test'),
        )
        test_loader = DataLoader(
            test_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )

    estimate_k_agglo(train_loader, args, device, test_loader=test_loader)
