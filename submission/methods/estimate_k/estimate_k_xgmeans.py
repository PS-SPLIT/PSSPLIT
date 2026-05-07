"""
estimate_k_xgmeans.py — X-Means and G-Means k estimation methods.

Both methods run on the combined labelled + unlabelled features and determine
k automatically.

Tunable hyperparameters
-----------------------
xmeans:
  criterion  : splitting criterion — 0 = Bayesian Information Criterion (BIC,
               default), 1 = Minimum Noiseless Description Length (MNDL).
               Controlled via --xmeans_criterion (single value) or
               --xmeans_criterion_candidates (comma-separated list to search over).
  pca_dim    : PCA dimensionality; shared tuning mechanism (--tune_pca /
               --pca_dim_candidates).

gmeans:
  pca_dim    : only tunable parameter; same mechanism as above.

Tolerance is NOT exposed as a tunable parameter for either method.

Tuning criterion
----------------
All three options are supported (labelled_k, test_cluster, combined) via the
shared estimate_k_tuning framework; see that module for definitions.

For labelled_k: the method is run on *labelled* features only; score = how
close k_hat is to k_true.  Uses skip_on_error=True because pyclustering may
fail on small feature sets.

Error handling
--------------
ImportError from pyclustering is always re-raised (fail loud, never fake).
Per-candidate failures with skip_on_error=True log a warning and skip the
candidate; if ALL candidates fail a RuntimeError is raised.
"""

import itertools

import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize

from project_utils.cluster_utils import cluster_acc
from methods.estimate_k.estimate_k_agglo import (
    collect_and_split_features,
    apply_pca_and_norm,
)
from methods.estimate_k.estimate_k_tuning import (
    CRITERION_LABELLED, CRITERION_TEST, CRITERION_COMBINED,
    collect_test_features_np,
    select_best_config,
)

import warnings as _warnings
# provide the legacy attribute so pyclustering stops crashing
if not hasattr(np, "warnings"):
    np.warnings = _warnings

# ---------------------------------------------------------------------------
# Shared feature helpers
# ---------------------------------------------------------------------------

def _collect_features_raw(train_loader, args, device):
    """
    Load labelled and unlabelled features without applying PCA.

    Returns
    -------
    l_feats_raw : (n_l, D)  float32, L2-normed
    l_targets   : (n_l,)
    u_feats_raw : (n_u, D)  float32, L2-normed
    u_targets   : (n_u,)
    all_mask    : (N,) bool
    n_l         : int
    k_l         : int  number of known classes
    """
    l_feats, l_targets, u_feats, u_targets, all_mask = \
        collect_and_split_features(train_loader, device)
    print(f"  labelled={len(l_feats)}, unlabelled={len(u_feats)}, "
          f"dim={l_feats.shape[1]}")
    l_feats_raw = normalize(l_feats, norm='l2', axis=1).astype(np.float32)
    u_feats_raw = normalize(u_feats, norm='l2', axis=1).astype(np.float32)
    n_l = len(l_feats_raw)
    k_l = len(args.train_classes)
    return l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask, n_l, k_l


def _apply_pca_or_norm(l_feats_raw, u_feats_raw, pca_dim):
    """Apply PCA (or just identity) for a given pca_dim candidate."""
    if pca_dim > 0:
        return apply_pca_and_norm(l_feats_raw, u_feats_raw, pca_dim)
    else:
        return l_feats_raw, u_feats_raw, None


def _collect_features(train_loader, args, device):
    """
    Convenience wrapper: load features and apply PCA using args.pca_dim.
    Kept for backward-compatibility with any external callers.
    """
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask, n_l, k_l = \
        _collect_features_raw(train_loader, args, device)
    pca_dim = int(getattr(args, 'pca_dim', 64))
    Xp_l, Xp_u, pca = _apply_pca_or_norm(l_feats_raw, u_feats_raw, pca_dim)
    return Xp_l, l_targets, Xp_u, u_targets, all_mask, n_l, k_l, pca


# ---------------------------------------------------------------------------
# Shared post-estimation helpers
# ---------------------------------------------------------------------------

def _eval_and_report(preds_all, l_targets, n_l, k_l, method_name):
    """
    Compute AMI and clustering accuracy on labelled portion, print results,
    and return the estimated total K and number of novel classes.

    Returns
    -------
    k_hat   : int  total estimated K
    k_novel : int  estimated number of novel classes
    """
    preds_l = preds_all[:n_l].astype(int)
    l_targets_int = l_targets.astype(int)

    ami = adjusted_mutual_info_score(l_targets_int, preds_l)
    acc = cluster_acc(l_targets_int, preds_l)
    print(f"  [{method_name}] labelled AMI={ami:.4f}, ACC={acc:.4f}")

    k_hat   = int(len(np.unique(preds_all)))
    k_novel = max(1, k_hat - k_l)
    print(f"  [{method_name}] k_hat={k_hat}, k_novel={k_novel}")
    print(f"Optimal K is {k_hat}")
    return k_hat, k_novel


def _build_result(preds_all, Xp_l, Xp_u, l_targets, u_targets, all_mask, pca):
    """
    Pack clustering output into a result dict for downstream evaluation.

    preds_all is ordered as [labelled | unlabelled] (matching X_all = vstack([Xp_l, Xp_u])).
    We reconstruct the original loader order using all_mask.
    """
    n_l = int(all_mask.sum())
    feat_dim = Xp_l.shape[1]

    Xp_full = np.empty((len(all_mask), feat_dim), dtype=np.float32)
    Xp_full[all_mask]  = Xp_l
    Xp_full[~all_mask] = Xp_u

    labels_full = np.empty(len(all_mask), dtype=np.int64)
    labels_full[all_mask]  = l_targets.astype(np.int64)
    labels_full[~all_mask] = u_targets.astype(np.int64)

    preds_full = np.empty(len(all_mask), dtype=np.int64)
    preds_full[all_mask]  = preds_all[:n_l].astype(np.int64)
    preds_full[~all_mask] = preds_all[n_l:].astype(np.int64)

    return dict(
        all_feats=Xp_full,
        all_labels=labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=pca,
    )


def _build_result_from_best(best, l_targets, u_targets, all_mask):
    """Build result dict from a TuningResult (best config)."""
    n_l = int(all_mask.sum())
    feat_dim = best.X_all.shape[1]

    Xp_full = np.empty((len(all_mask), feat_dim), dtype=np.float32)
    Xp_full[all_mask]  = best.X_all[:n_l]
    Xp_full[~all_mask] = best.X_all[n_l:]

    labels_full = np.empty(len(all_mask), dtype=np.int64)
    labels_full[all_mask]  = l_targets.astype(np.int64)
    labels_full[~all_mask] = u_targets.astype(np.int64)

    preds_full = np.empty(len(all_mask), dtype=np.int64)
    preds_full[all_mask]  = best.preds[:n_l]
    preds_full[~all_mask] = best.preds[n_l:]

    return dict(
        all_feats=Xp_full,
        all_labels=labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=best.pca,
        fitted_model=best.fitted_model,
    )


# ---------------------------------------------------------------------------
# X-Means
# ---------------------------------------------------------------------------

def estimate_k_xmeans(train_loader, args, device=None, test_loader=None):
    """
    Estimate K using X-Means (pyclustering).

    Prints "Optimal K is X" for parsing by run_pipeline.py.

    Parameters
    ----------
    train_loader : DataLoader over a MergedDataset FeatureVectorDataset
    args         : namespace  (uses pca_dim, max_classes, train_classes)
    device       : torch.device or None
    test_loader  : DataLoader for test split; required when
                   args.tuning_criterion is 'test_cluster' or 'combined'

    Optional args fields (new)
    --------------------------
    xmeans_criterion            : int  0=BIC (default), 1=MNDL
    xmeans_criterion_candidates : str  comma-separated list (default: str(xmeans_criterion))
    tuning_criterion            : 'labelled_k' (default) | 'test_cluster' | 'combined'
    tuning_weight               : float (default 0.5)
    tune_pca                    : bool  (default False)
    pca_dim_candidates          : str   (default str(pca_dim))

    Returns
    -------
    estimated_k_total : int
    estimated_C_u     : int
    result            : dict with keys all_feats, all_labels, mask_lab, preds, pca
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pca_dim           = int(getattr(args, 'pca_dim', 64))
    kmax              = int(getattr(args, 'max_classes', 50))
    tuning_criterion  = getattr(args, 'tuning_criterion', CRITERION_LABELLED)
    tuning_weight     = float(getattr(args, 'tuning_weight', 0.5))
    tune_pca          = bool(getattr(args, 'tune_pca', False))
    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', str(pca_dim))
    xmeans_criterion  = int(getattr(args, 'xmeans_criterion', 0))
    crit_cands_str    = getattr(args, 'xmeans_criterion_candidates',
                                str(xmeans_criterion))

    pca_dims  = (
        [int(x.strip()) for x in pca_dim_cands_str.split(',')]
        if tune_pca else [pca_dim]
    )
    crit_list = [int(x.strip()) for x in crit_cands_str.split(',')]

    print("XMeans k-estimation: collecting features...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask, n_l, k_l = \
        _collect_features_raw(train_loader, args, device)

    # ------------------------------------------------------------------
    # Pre-collect test features if needed
    # ------------------------------------------------------------------
    test_feats_raw = None
    test_labels    = None
    if tuning_criterion in (CRITERION_TEST, CRITERION_COMBINED):
        if test_loader is None:
            raise ValueError(
                "estimate_k_xmeans: test_loader must be provided when "
                f"tuning_criterion={tuning_criterion!r}. "
                "Pass test_loader from estimate_k.py."
            )
        print("  Pre-collecting test features for tuning criterion...")
        test_feats_raw, test_labels = collect_test_features_np(
            test_loader, pca=None, device=device
        )

    # ------------------------------------------------------------------
    # Pre-cache PCA transforms (one fit per unique pca_dim, not per candidate)
    # ------------------------------------------------------------------
    print(f"  Pre-computing PCA for {len(set(pca_dims))} unique dim(s): "
          f"{sorted(set(pca_dims))}")
    _pca_cache = {}
    for pd in sorted(set(pca_dims)):
        _pca_cache[pd] = _apply_pca_or_norm(l_feats_raw, u_feats_raw, pd)

    # ------------------------------------------------------------------
    # Build param grid
    # ------------------------------------------------------------------
    param_grid = [
        {'criterion': c, 'pca_dim': pd}
        for c, pd in itertools.product(crit_list, pca_dims)
    ]
    crit_names = {0: 'BIC', 1: 'MNDL'}
    print(f"  Tuning: {len(crit_list)} criterion × {len(pca_dims)} pca_dim(s) "
          f"= {len(param_grid)} candidates "
          f"(criterion={tuning_criterion})")

    # ------------------------------------------------------------------
    # pyclustering import check (fail loud if missing)
    # ------------------------------------------------------------------
    try:
        from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
    except ImportError as e:
        raise ImportError(
            f"pyclustering is required for xmeans k-estimation but is not "
            f"installed: {e}"
        )

    # ------------------------------------------------------------------
    # Define labelled_fn and full_fn (use PCA cache)
    # ------------------------------------------------------------------

    def _run_xmeans(X_list, criterion_val):
        """Run xmeans on a list-of-lists; return cluster label array."""
        initial_centers = kmeans_plusplus_initializer(
            X_list, 1, random_state=0
        ).initialize()
        xm = xmeans(
            X_list,
            initial_centers=initial_centers,
            kmax=kmax,
            criterion=criterion_val,
            ccore=True,
        )
        xm.process()
        return np.array(xm.predict(X_list), dtype=np.int64)

    def labelled_fn(params):
        """XMeans on labelled-only features → k_hat_l."""
        Xp_l, _Xp_u, _ = _pca_cache[params['pca_dim']]
        preds_l = _run_xmeans(Xp_l.tolist(), params['criterion'])
        return int(len(np.unique(preds_l)))

    def full_fn(params):
        """XMeans on X_all → k_hat, preds, X_all, pca, fitted_model."""
        Xp_l, Xp_u, pca = _pca_cache[params['pca_dim']]
        X_all  = np.vstack([Xp_l, Xp_u])
        preds  = _run_xmeans(X_all.tolist(), params['criterion'])
        k_hat  = int(len(np.unique(preds)))
        c_name = crit_names.get(params['criterion'], str(params['criterion']))
        print(f"    xmeans criterion={c_name}, pca_dim={params['pca_dim']}: "
              f"k_hat={k_hat}")
        return k_hat, preds, X_all, pca, None

    # ------------------------------------------------------------------
    # Run selection
    # ------------------------------------------------------------------
    best = select_best_config(
        param_grid=param_grid,
        k_true=k_l,
        criterion=tuning_criterion,
        weight=tuning_weight,
        labelled_fn=labelled_fn,
        full_fn=full_fn,
        test_feats_raw=test_feats_raw,
        test_labels=test_labels,
        num_labeled_classes=args.num_labeled_classes,
        skip_on_error=True,
    )

    k_hat   = best.k_hat
    k_novel = max(1, k_hat - k_l)

    import json as _json
    print(f"Best params: {_json.dumps(best.params)}")
    _eval_and_report(best.preds, l_targets, n_l, k_l, 'xmeans')

    result = _build_result_from_best(best, l_targets, u_targets, all_mask)
    return k_hat, k_novel, result


# ---------------------------------------------------------------------------
# G-Means
# ---------------------------------------------------------------------------

def estimate_k_gmeans(train_loader, args, device=None, test_loader=None):
    """
    Estimate K using G-Means (pyclustering).

    Prints "Optimal K is X" for parsing by run_pipeline.py.

    Parameters
    ----------
    train_loader : DataLoader over a MergedDataset FeatureVectorDataset
    args         : namespace  (uses pca_dim, train_classes)
    device       : torch.device or None
    test_loader  : DataLoader for test split; required when
                   args.tuning_criterion is 'test_cluster' or 'combined'

    Optional args fields (new)
    --------------------------
    tuning_criterion   : 'labelled_k' (default) | 'test_cluster' | 'combined'
    tuning_weight      : float (default 0.5)
    tune_pca           : bool  (default False)
    pca_dim_candidates : str   (default str(pca_dim))

    Returns
    -------
    estimated_k_total : int
    estimated_C_u     : int
    result            : dict with keys all_feats, all_labels, mask_lab, preds, pca
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pca_dim           = int(getattr(args, 'pca_dim', 64))
    tuning_criterion  = getattr(args, 'tuning_criterion', CRITERION_LABELLED)
    tuning_weight     = float(getattr(args, 'tuning_weight', 0.5))
    tune_pca          = bool(getattr(args, 'tune_pca', False))
    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', str(pca_dim))

    pca_dims = (
        [int(x.strip()) for x in pca_dim_cands_str.split(',')]
        if tune_pca else [pca_dim]
    )

    print("GMeans k-estimation: collecting features...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask, n_l, k_l = \
        _collect_features_raw(train_loader, args, device)

    # ------------------------------------------------------------------
    # Pre-collect test features if needed
    # ------------------------------------------------------------------
    test_feats_raw = None
    test_labels    = None
    if tuning_criterion in (CRITERION_TEST, CRITERION_COMBINED):
        if test_loader is None:
            raise ValueError(
                "estimate_k_gmeans: test_loader must be provided when "
                f"tuning_criterion={tuning_criterion!r}. "
                "Pass test_loader from estimate_k.py."
            )
        print("  Pre-collecting test features for tuning criterion...")
        test_feats_raw, test_labels = collect_test_features_np(
            test_loader, pca=None, device=device
        )

    # ------------------------------------------------------------------
    # Pre-cache PCA transforms (one fit per unique pca_dim, not per candidate)
    # ------------------------------------------------------------------
    print(f"  Pre-computing PCA for {len(set(pca_dims))} unique dim(s): "
          f"{sorted(set(pca_dims))}")
    _pca_cache = {}
    for pd in sorted(set(pca_dims)):
        _pca_cache[pd] = _apply_pca_or_norm(l_feats_raw, u_feats_raw, pd)

    # ------------------------------------------------------------------
    # Build param grid (pca_dim only — no other tunable hyperparameter)
    # ------------------------------------------------------------------
    param_grid = [{'pca_dim': pd} for pd in pca_dims]
    print(f"  Tuning: {len(pca_dims)} pca_dim(s) = {len(param_grid)} candidates "
          f"(criterion={tuning_criterion})")

    # ------------------------------------------------------------------
    # pyclustering import check (fail loud if missing)
    # ------------------------------------------------------------------
    try:
        from pyclustering.cluster.gmeans import gmeans as pyclustering_gmeans
    except ImportError as e:
        raise ImportError(
            f"pyclustering is required for gmeans k-estimation but is not "
            f"installed: {e}"
        )

    # ------------------------------------------------------------------
    # Define labelled_fn and full_fn (use PCA cache)
    # ------------------------------------------------------------------

    def _run_gmeans(X_list):
        """Run gmeans on a list-of-lists; return cluster label array."""
        gm = pyclustering_gmeans(X_list, ccore=True)
        gm.process()
        return np.array(gm.predict(X_list), dtype=np.int64)

    def labelled_fn(params):
        """GMeans on labelled-only features → k_hat_l."""
        Xp_l, _Xp_u, _ = _pca_cache[params['pca_dim']]
        preds_l = _run_gmeans(Xp_l.tolist())
        return int(len(np.unique(preds_l)))

    def full_fn(params):
        """GMeans on X_all → k_hat, preds, X_all, pca, fitted_model."""
        Xp_l, Xp_u, pca = _pca_cache[params['pca_dim']]
        X_all = np.vstack([Xp_l, Xp_u])
        preds = _run_gmeans(X_all.tolist())
        k_hat = int(len(np.unique(preds)))
        print(f"    gmeans pca_dim={params['pca_dim']}: k_hat={k_hat}")
        return k_hat, preds, X_all, pca, None

    # ------------------------------------------------------------------
    # Run selection
    # ------------------------------------------------------------------
    best = select_best_config(
        param_grid=param_grid,
        k_true=k_l,
        criterion=tuning_criterion,
        weight=tuning_weight,
        labelled_fn=labelled_fn,
        full_fn=full_fn,
        test_feats_raw=test_feats_raw,
        test_labels=test_labels,
        num_labeled_classes=args.num_labeled_classes,
        skip_on_error=True,
    )

    k_hat   = best.k_hat
    k_novel = max(1, k_hat - k_l)

    import json as _json
    print(f"Best params: {_json.dumps(best.params)}")
    _eval_and_report(best.preds, l_targets, n_l, k_l, 'gmeans')

    result = _build_result_from_best(best, l_targets, u_targets, all_mask)
    return k_hat, k_novel, result
