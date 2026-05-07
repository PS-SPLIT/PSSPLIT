"""
estimate_k_tuning.py — Shared hyperparameter-tuning framework for K estimation.

Provides a consistent selection pipeline for agglo, spectral, xmeans, and gmeans.

Tuning criteria
---------------
labelled_k      : score = f(k_hat_l, k_true) where k_hat_l comes from running the
                  method on labelled features only.  Mirrors the original agglo
                  tune_on_labelled logic: a good hyperparameter setting should
                  recover the known number of classes from labelled data.

test_cluster    : score = clustering accuracy on the *old* (known) classes in the
                  test split, estimated via nearest-centroid prediction from training
                  cluster centroids.  Only old-class accuracy is used because novel
                  class labels are unknown at tuning time.

combined        : weighted combination of both:
                  weight * labelled_k_score + (1 - weight) * test_cluster_score

labelled_cluster: score = cluster_acc after clustering *labelled training* features at
                  the known k_true.  Unlike labelled_k it evaluates actual cluster
                  quality (not just k estimation).  Unlike test_cluster it needs no
                  held-out test data.  Used by estimate_k_brent_tuned as the default
                  Phase-1 hyperparameter selection criterion.
                  Note: evaluates on the same features used for fitting (standard for
                  transductive methods like spectral/agglo that have no predict step).

Caller contract
---------------
Each method must supply:

  labelled_fn(params) -> k_hat_l : int
      Runs the method on *labelled features only* and returns the number of
      clusters found.  Required for 'labelled_k' and 'combined' criteria.

  full_fn(params) -> (k_hat, preds, X_all, pca)
      Runs the method on the full training set (labelled + unlabelled), returns:
        k_hat  – estimated total K
        preds  – int64 cluster assignments shaped (N_l + N_u,) ordered
                 [labelled | unlabelled] matching X_all
        X_all  – float32 feature matrix (N_l+N_u, D), PCA-reduced if any
        pca    – fitted sklearn PCA object, or None
      Always called (even for 'labelled_k') to produce the final result dict.

Usage example (sketch)
----------------------
    from methods.estimate_k.estimate_k_tuning import (
        collect_test_features_np, select_best_config,
        CRITERION_LABELLED, CRITERION_TEST, CRITERION_COMBINED,
    )

    # Pre-collect raw test features once (no PCA yet)
    test_feats_raw, test_labels = collect_test_features_np(
        test_loader, pca=None, device=device
    )

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
    # best.k_hat, best.preds, best.X_all, best.pca, best.params
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize as sk_normalize

# ---------------------------------------------------------------------------
# Shared clustering helper
# ---------------------------------------------------------------------------

def _cluster_with_method(X, k, method, n_neighbors=10, return_model=False,
                          covariance_type='full', init_params='kmeans'):
    """
    Cluster X into k groups using the specified method.

    Parameters
    ----------
    X               : np.ndarray (N, D)
    k               : int  number of clusters
    method          : str  'kmeans' | 'spectral' | 'gmm' | 'agglo'
                           ('auto' is treated as 'kmeans')
    n_neighbors     : int  only used for 'spectral'
    return_model    : bool if True, returns (labels, fitted_model) instead of labels.
                           fitted_model is None for 'spectral' and 'agglo' (no
                           predict-on-new-data model — nearest centroid used downstream),
                           the fitted GaussianMixture for 'gmm', and the fitted KMeans
                           for 'kmeans'/'auto'.
    covariance_type : str  GMM covariance type: 'full' | 'tied' | 'diag' | 'spherical'
                           Only used when method='gmm'. (default: 'full')
    init_params     : str  GMM initialization strategy: 'kmeans' | 'k-means++' |
                           'random' | 'random_from_data'. Only used when method='gmm'.
                           (default: 'kmeans')

    Returns
    -------
    labels : np.ndarray (N,) int64                          if return_model=False
    (labels, model) : Tuple[np.ndarray, Any]                if return_model=True
    """
    if method == 'spectral':
        from sklearn.cluster import SpectralClustering
        n_nb = min(n_neighbors, max(2, X.shape[0] - 1))
        sc = SpectralClustering(
            n_clusters=k,
            affinity='nearest_neighbors',
            n_neighbors=n_nb,
            assign_labels='kmeans',
            random_state=0,
        )
        labels = sc.fit_predict(X).astype(np.int64)
        model = None
    elif method == 'gmm':
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                             init_params=init_params, random_state=0)
        gm.fit(X)
        labels = gm.predict(X).astype(np.int64)
        model = gm
    elif method == 'agglo':
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = ac.fit_predict(X).astype(np.int64)
        model = None   # no predict() — nearest centroid used for test assignment downstream
    else:  # 'kmeans' or 'auto'
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(X).astype(np.int64)
        model = km
    return (labels, model) if return_model else labels


# ---------------------------------------------------------------------------
# Criterion constants
# ---------------------------------------------------------------------------

CRITERION_LABELLED         = 'labelled_k'
CRITERION_TEST             = 'test_cluster'
CRITERION_COMBINED         = 'combined'
CRITERION_LABELLED_CLUSTER = 'labelled_cluster'

_VALID_CRITERIA = (
    CRITERION_LABELLED,
    CRITERION_TEST,
    CRITERION_COMBINED,
    CRITERION_LABELLED_CLUSTER,
)


# ---------------------------------------------------------------------------
# Test feature collection
# ---------------------------------------------------------------------------

def collect_test_features_np(
    test_loader,
    pca,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect raw test features from a DataLoader, L2-normalise, optionally apply
    PCA, and L2-normalise again.

    Parameters
    ----------
    test_loader : DataLoader yielding (feats, label, idx) 3-tuples
    pca         : fitted sklearn PCA object, or None
    device      : torch.device

    Returns
    -------
    test_feats  : (N_test, D) float32  — L2-normed, PCA-applied if pca given
    test_labels : (N_test,)   int64
    """
    feats_list, labels_list = [], []
    for batch in test_loader:
        feats, label = batch[0], batch[1]
        feats = F.normalize(feats.float().to(device), dim=-1)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(label.cpu().numpy())

    raw_feats  = np.concatenate(feats_list,  axis=0).astype(np.float32)
    raw_labels = np.concatenate(labels_list, axis=0).astype(np.int64)

    if pca is not None:
        pca_feats  = pca.transform(raw_feats)
        test_feats = sk_normalize(pca_feats, norm='l2', axis=1).astype(np.float32)
    else:
        test_feats = raw_feats

    return test_feats, raw_labels


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def labelled_k_score(k_hat: int, k_true: int) -> float:
    """
    Smooth score measuring how close k_hat is to k_true.

    Returns 1.0 for an exact match; 0.5 if off by 1; 0.333 if off by 2; etc.
    Using 1/(1+|Δ|) instead of binary 0/1 lets the grid search distinguish
    between candidates that are all wrong but at varying distances.
    """
    return 1.0 / (1.0 + abs(k_hat - k_true))


def labelled_cluster_score(
    l_feats:     np.ndarray,
    l_targets:   np.ndarray,
    k_true:      int,
    cluster_fn,
) -> float:
    """
    Score = cluster_acc after clustering labelled training features at k_true.

    Used as Phase-1 criterion in estimate_k_brent_tuned: given best-params,
    cluster only the labelled training data at the known k, and measure how
    well the known class structure is recovered.

    Parameters
    ----------
    l_feats    : (N_l, D) float32  labelled training features (PCA-applied if any)
    l_targets  : (N_l,)   int      ground-truth class indices
    k_true     : int               number of labelled classes (= n_clusters to fit)
    cluster_fn : callable (X, k) -> preds np.ndarray (N,)
                 A closure that encapsulates method + params; receives the
                 features and n_clusters, returns int64 cluster assignments.

    Returns
    -------
    float in [0, 1] — higher is better
    """
    from project_utils.cluster_utils import cluster_acc
    preds = cluster_fn(l_feats, k_true)
    return float(cluster_acc(l_targets.astype(int), preds.astype(int)))


def predict_nearest_centroid_np(
    train_feats: np.ndarray,
    train_preds: np.ndarray,
    test_feats:  np.ndarray,
) -> np.ndarray:
    """
    Assign each test point to the nearest training cluster centroid.

    Parameters
    ----------
    train_feats : (N_train, D)
    train_preds : (N_train,)  int
    test_feats  : (N_test, D)

    Returns
    -------
    (N_test,) int64 — predicted cluster IDs
    """
    unique_clusters = np.unique(train_preds)
    centroids = np.array([
        train_feats[train_preds == c].mean(axis=0)
        for c in unique_clusters
    ], dtype=np.float32)
    # Squared L2 distances: (N_test, n_clusters)
    diffs = test_feats[:, None, :] - centroids[None, :, :]  # (N_test, K, D)
    sq_dists = (diffs ** 2).sum(axis=-1)                    # (N_test, K)
    return unique_clusters[np.argmin(sq_dists, axis=1)].astype(np.int64)


def test_cluster_score(
    X_all:               np.ndarray,
    preds:               np.ndarray,
    test_feats_raw:      np.ndarray,
    test_labels:         np.ndarray,
    num_labeled_classes: int,
    pca=None,
    fitted_model=None,
) -> float:
    """
    Evaluate clustering quality on the **old (known) classes** in the test split.

    Novel class labels are unknown at tuning time so only old-class accuracy is
    used as the validation signal.

    Parameters
    ----------
    X_all               : (N_train, D)  training features (PCA-reduced if any)
    preds               : (N_train,)    cluster assignments on training data
    test_feats_raw      : (N_test, D)   raw L2-normalised test features (no PCA)
    test_labels         : (N_test,)     ground-truth test class IDs
    num_labeled_classes : int           threshold: labels < this are "old"
    pca                 : optional fitted PCA from this candidate's full_fn
    fitted_model        : optional fitted sklearn model (e.g. GaussianMixture).
                          When provided and is a GaussianMixture, its predict()
                          is used instead of nearest-centroid assignment.

    Returns
    -------
    float — acc_old from split_cluster_acc_v2 (accuracy on old classes only)
    """
    from sklearn.mixture import GaussianMixture
    from project_utils.cluster_and_log_utils import split_cluster_acc_v2

    # Apply same PCA (if any) to raw test features
    if pca is not None:
        pca_feats  = pca.transform(test_feats_raw)
        test_feats = sk_normalize(pca_feats, norm='l2', axis=1).astype(np.float32)
    else:
        test_feats = test_feats_raw

    if isinstance(fitted_model, GaussianMixture):
        test_preds = fitted_model.predict(test_feats).astype(np.int64)
    else:
        test_preds = predict_nearest_centroid_np(X_all, preds, test_feats)

    mask_old = test_labels < num_labeled_classes
    if mask_old.sum() == 0:
        # No old-class test samples — cannot evaluate; return 0.0
        return 0.0

    _, acc_old, _ = split_cluster_acc_v2(
        test_labels.astype(int), test_preds.astype(int), mask_old
    )
    return float(acc_old)


def compute_selection_score(
    l_score:   float,
    t_score:   float,
    criterion: str,
    weight:    float,
    lc_score:  float = 0.0,
) -> float:
    """
    Combine criterion scores into a single selection scalar.

    Parameters
    ----------
    l_score   : output of labelled_k_score (used by 'labelled_k' and 'combined')
    t_score   : output of test_cluster_score (used by 'test_cluster' and 'combined')
    criterion : 'labelled_k' | 'test_cluster' | 'combined' | 'labelled_cluster'
    weight    : weight for l_score in 'combined' (test gets 1-weight)
    lc_score  : output of labelled_cluster_score (used by 'labelled_cluster')

    Raises
    ------
    ValueError  if criterion is not one of the valid options
    """
    if criterion == CRITERION_LABELLED:
        return l_score
    elif criterion == CRITERION_TEST:
        return t_score
    elif criterion == CRITERION_COMBINED:
        return weight * l_score + (1.0 - weight) * t_score
    elif criterion == CRITERION_LABELLED_CLUSTER:
        return lc_score
    else:
        raise ValueError(
            f"Unknown tuning criterion: {criterion!r}. "
            f"Expected one of {_VALID_CRITERIA}."
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    """
    Holds everything about the best-selected candidate configuration.

    Attributes
    ----------
    params       : hyperparameter dict for this candidate
    k_hat        : estimated total K (from full_fn)
    preds        : cluster assignments on X_all, shape (N_l + N_u,)
                   ordered [labelled | unlabelled] matching X_all
    X_all        : feature matrix used for clustering (N_l+N_u, D)
    pca          : fitted PCA or None (apply to test features for consistency)
    fitted_model : fitted sklearn clustering model (GaussianMixture, KMeans) or
                   None; use for predict-on-new-data instead of nearest centroid
    labelled_k_sc: score from labelled_k_score (0.0 if criterion skips it)
    test_acc     : score from test_cluster_score (0.0 if criterion skips it)
    score        : final selection score from compute_selection_score
    """
    params:        Dict[str, Any]
    k_hat:         int
    preds:         np.ndarray
    X_all:         np.ndarray
    pca:           Any
    fitted_model:  Any
    labelled_k_sc: float
    test_acc:      float
    score:         float


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------

def select_best_config(
    param_grid:          List[Dict[str, Any]],
    k_true:              int,
    criterion:           str,
    weight:              float,
    labelled_fn:         Optional[Callable[[Dict], int]],
    full_fn:             Callable[[Dict], Tuple[int, np.ndarray, np.ndarray, Any]],
    test_feats_raw:      Optional[np.ndarray],
    test_labels:         Optional[np.ndarray],
    num_labeled_classes: int,
    skip_on_error:       bool = False,
) -> TuningResult:
    """
    Iterate over param_grid and return the TuningResult with the highest score.

    Parameters
    ----------
    param_grid          : list of dicts, one per candidate configuration
    k_true              : true number of known classes (= num_labeled_classes)
    criterion           : 'labelled_k' | 'test_cluster' | 'combined'
    weight              : weight for labelled_k in combined criterion
    labelled_fn         : params -> k_hat_l; required when criterion includes
                          'labelled_k'
    full_fn             : params -> (k_hat, preds, X_all, pca, fitted_model);
                          always called to produce the final result.
                          fitted_model is the fitted sklearn clustering object
                          (GaussianMixture, KMeans) or None for methods that
                          lack a predict-on-new-data model (e.g. spectral/agglo)
    test_feats_raw      : raw L2-norm test features (no PCA);
                          required when criterion includes 'test_cluster'
    test_labels         : test ground-truth labels (same length as test_feats_raw)
    num_labeled_classes : number of known classes
    skip_on_error       : if True, log a warning and skip candidates where
                          full_fn raises an exception; if ALL candidates fail
                          a RuntimeError is raised regardless

    Returns
    -------
    TuningResult for the best-scoring candidate

    Raises
    ------
    ValueError   if param_grid is empty, criterion is invalid, or required
                 arguments are missing
    RuntimeError if skip_on_error=True and every candidate fails
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if criterion not in _VALID_CRITERIA:
        raise ValueError(
            f"Unknown tuning criterion: {criterion!r}. "
            f"Expected one of {_VALID_CRITERIA}."
        )
    if len(param_grid) == 0:
        raise ValueError(
            "param_grid is empty; cannot select best configuration. "
            "Check that candidate lists (linkages, n_neighbors, etc.) are non-empty."
        )
    if criterion in (CRITERION_LABELLED, CRITERION_COMBINED) and labelled_fn is None:
        raise ValueError(
            f"labelled_fn must be provided when criterion={criterion!r}. "
            "Provide a function that runs the method on labelled features."
        )
    if criterion in (CRITERION_TEST, CRITERION_COMBINED) and test_feats_raw is None:
        raise ValueError(
            f"test_feats_raw must be provided when criterion={criterion!r}. "
            "Pass test_loader to the estimation function so test features can be "
            "pre-collected before the tuning loop."
        )
    if criterion == CRITERION_LABELLED_CLUSTER:
        raise ValueError(
            "criterion='labelled_cluster' is not supported by select_best_config. "
            "It is used exclusively by estimate_k_brent_tuned which has its own "
            "Phase-1 loop. Use 'labelled_k', 'test_cluster', or 'combined' with "
            "this function."
        )

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    results: List[TuningResult] = []

    for i, params in enumerate(param_grid):
        # --- labelled_k score (only when needed) ---
        l_score = 0.0
        if criterion in (CRITERION_LABELLED, CRITERION_COMBINED):
            k_hat_l = labelled_fn(params)
            l_score = labelled_k_score(k_hat_l, k_true)

        # --- full clustering ---
        try:
            k_hat, preds, X_all, pca, fitted_model = full_fn(params)
        except Exception as exc:
            if skip_on_error:
                print(
                    f"  [Tuning] WARNING: candidate {i+1}/{len(param_grid)} "
                    f"params={params} failed — skipping. Error: {exc}"
                )
                continue
            raise

        # --- test cluster score (only when needed) ---
        t_score = 0.0
        if criterion in (CRITERION_TEST, CRITERION_COMBINED):
            t_score = test_cluster_score(
                X_all, preds,
                test_feats_raw, test_labels,
                num_labeled_classes,
                pca=pca,
                fitted_model=fitted_model,
            )

        score = compute_selection_score(l_score, t_score, criterion, weight)
        results.append(TuningResult(
            params=params,
            k_hat=k_hat,
            preds=preds,
            X_all=X_all,
            pca=pca,
            fitted_model=fitted_model,
            labelled_k_sc=l_score,
            test_acc=t_score,
            score=score,
        ))

    if not results:
        raise RuntimeError(
            "All candidates in param_grid failed during tuning. "
            "This likely means pyclustering is unavailable or all hyperparameter "
            "combinations produced errors. Check the warnings above."
        )

    best = max(results, key=lambda r: r.score)
    print(
        f"  [Tuning] Best params={best.params}, k_hat={best.k_hat}, "
        f"l_score={best.labelled_k_sc:.4f}, t_acc_old={best.test_acc:.4f}, "
        f"score={best.score:.4f}  ({len(results)}/{len(param_grid)} candidates evaluated)"
    )
    return best
