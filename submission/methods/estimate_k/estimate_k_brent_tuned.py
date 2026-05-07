"""
estimate_k_brent_tuned.py — Two-phase K estimation: tune hyperparameters on
labelled data, then search for K with Brent's algorithm on full data.

Phase 1 — Hyperparameter tuning (k is FIXED = k_true, labelled data only)
--------------------------------------------------------------------------
Cluster only the labelled training features at n_clusters = k_true (the
known number of labelled classes).  Score each hyperparameter config with
cluster_acc(l_targets, preds) — a direct measure of how well the config
captures the known class structure.  k is *not* being searched here; it is
fixed to the known value.  Only hyperparameters (PCA dim, n_neighbors,
linkage, covariance_type, etc.) vary.

Phase 2 — K search (k is VARIABLE, all data)
---------------------------------------------
Using all features (labelled + unlabelled), with the best hyperparameters
found in Phase 1 held fixed, run Brent's algorithm over k in
[k_true, max_classes].  For each candidate k all features are clustered and
scored by labelled-subset cluster_acc.  The k that maximises this score is
the estimated total K (k_true + k_novel, where k_novel was hidden in the
unlabelled pool).

Supported clustering methods (--final_cluster_method)
-----------------------------------------------------
spectral : SpectralClustering with precomputed affinity built once (Phase 2).
           Phase-1 param grid: n_neighbors × pca_dim × assign_labels.
           Affinity type controlled by --spectral_affinity (all types supported).
agglo    : AgglomerativeClustering with n_clusters=k.
           Phase-1 param grid: linkage × pca_dim.
gmm      : GaussianMixture.  Phase-1 param grid: covariance_type × init_params × pca_dim.
kmeans   : KMeans.  Phase-1 param grid: pca_dim only.
auto     : treated as 'kmeans' (no extra params to tune).

Phase-1 criterion (--tuning_criterion, default 'labelled_cluster')
-------------------------------------------------------------------
labelled_cluster (default): cluster_acc on labelled training features at k_true.
test_cluster               : old-class ACC on test split (requires test_loader).

Phase-2 criterion is always Brent with labelled-subset cluster_acc.
"""

import itertools
import json as _json

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from methods.estimate_k.estimate_k_agglo import (
    apply_pca_and_norm,
    collect_and_split_features,
)
from methods.estimate_k.estimate_k_spectral import _build_affinity
from methods.estimate_k.estimate_k_tuning import (
    CRITERION_LABELLED_CLUSTER,
    CRITERION_TEST,
    _cluster_with_method,
    collect_test_features_np,
    test_cluster_score,
)
from project_utils.cluster_utils import cluster_acc


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def _apply_pca_candidate(l_feats_raw, u_feats_raw, pca_dim):
    """Apply PCA if pca_dim > 0, else return L2-normalised copies."""
    if pca_dim > 0:
        return apply_pca_and_norm(l_feats_raw, u_feats_raw, pca_dim)
    Xp_l = normalize(l_feats_raw, norm='l2', axis=1).astype(np.float32)
    Xp_u = normalize(u_feats_raw, norm='l2', axis=1).astype(np.float32)
    return Xp_l, Xp_u, None


# ---------------------------------------------------------------------------
# Phase-1 clustering helpers
# ---------------------------------------------------------------------------

def _phase1_cluster_spectral(Xp_l, k_true, params, affinity_type, degree, coef0):
    """Cluster labelled-only features at k_true with SpectralClustering."""
    n_nb = min(params.get('n_neighbors', 10), max(2, Xp_l.shape[0] - 1))
    A_l = _build_affinity(Xp_l, affinity_type,
                          n_neighbors=n_nb,
                          gamma=params.get('gamma', None),
                          degree=degree, coef0=coef0)
    A_dense = A_l.toarray() if hasattr(A_l, 'toarray') else np.asarray(A_l)
    sc = SpectralClustering(
        n_clusters=k_true,
        affinity='precomputed',
        assign_labels=params.get('assign_labels', 'kmeans'),
        random_state=0,
    )
    return sc.fit_predict(A_dense).astype(np.int64)


def _phase1_cluster_agglo(Xp_l, k_true, params, agglo_metric):
    """Cluster labelled-only features at k_true with AgglomerativeClustering."""
    linkage = params.get('linkage', 'ward')
    ac = AgglomerativeClustering(
        n_clusters=k_true,
        linkage=linkage,
        metric=agglo_metric if linkage != 'ward' else 'euclidean',
    )
    return ac.fit_predict(Xp_l).astype(np.int64)


def _phase1_cluster_gmm(Xp_l, k_true, params):
    """Cluster labelled-only features at k_true with GaussianMixture."""
    gm = GaussianMixture(
        n_components=k_true,
        covariance_type=params.get('covariance_type', 'full'),
        init_params=params.get('init_params', 'kmeans'),
        random_state=0,
    )
    gm.fit(Xp_l)
    # predict() returns component indices (0..k-1) — correct for cluster_acc.
    return gm.predict(Xp_l).astype(np.int64)


def _phase1_cluster_kmeans(Xp_l, k_true):
    """Cluster labelled-only features at k_true with KMeans."""
    return _cluster_with_method(Xp_l, k_true, 'kmeans').astype(np.int64)


# ---------------------------------------------------------------------------
# Phase-2 clustering helpers
# ---------------------------------------------------------------------------

def _phase2_cluster(all_feats, k, method, best_params, prebuilt_affinity=None,
                    agglo_metric='euclidean', return_model=False):
    """
    Cluster all_feats (labelled + unlabelled, [l|u] order) into k groups
    using the best params from Phase 1.

    Parameters
    ----------
    all_feats          : (N, D) float32
    k                  : int  number of clusters
    method             : 'spectral' | 'agglo' | 'gmm' | 'kmeans'
    best_params        : dict from Phase-1 selection
    prebuilt_affinity  : dense (N, N) affinity matrix; required when method='spectral'
    agglo_metric       : str  distance metric for agglo (ignored for ward)
    return_model       : bool  if True returns (preds, fitted_model) else preds

    Returns
    -------
    preds : np.ndarray (N,) int64  [or (preds, model) if return_model]
    """
    if method == 'spectral':
        if prebuilt_affinity is None:
            raise RuntimeError(
                "_phase2_cluster: prebuilt_affinity must be provided for spectral."
            )
        sc = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            assign_labels=best_params.get('assign_labels', 'kmeans'),
            random_state=0,
        )
        preds = sc.fit_predict(prebuilt_affinity).astype(np.int64)
        model = None

    elif method == 'agglo':
        linkage = best_params.get('linkage', 'ward')
        ac = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=agglo_metric if linkage != 'ward' else 'euclidean',
        )
        preds = ac.fit_predict(all_feats).astype(np.int64)
        model = None

    elif method == 'gmm':
        gm = GaussianMixture(
            n_components=k,
            covariance_type=best_params.get('covariance_type', 'full'),
            init_params=best_params.get('init_params', 'kmeans'),
            random_state=0,
        )
        gm.fit(all_feats)
        preds = gm.predict(all_feats).astype(np.int64)
        model = gm

    else:  # 'kmeans' or 'auto'
        preds, model = _cluster_with_method(all_feats, k, 'kmeans', return_model=True)
        preds = preds.astype(np.int64)

    return (preds, model) if return_model else preds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_k_brent_tuned(train_loader, args, device=None, test_loader=None):
    """
    Two-phase K estimation: tune hyperparameters on labelled data (Phase 1),
    then search for K with Brent's algorithm on full data (Phase 2).

    Prints "Optimal K is X" for parsing by run_pipeline.py.

    Parameters
    ----------
    train_loader : DataLoader over a MergedDataset FeatureVectorDataset
    args         : namespace — see module docstring for recognised fields
    device       : torch.device or None
    test_loader  : DataLoader for test split; required when
                   args.tuning_criterion == 'test_cluster'

    Optional args fields
    --------------------
    final_cluster_method            : 'spectral' | 'agglo' | 'gmm' | 'kmeans' | 'auto'
    tuning_criterion                : 'labelled_cluster' (default) | 'test_cluster'
    pca_dim                         : int   (default 64)
    tune_pca                        : bool  (default False)
    pca_dim_candidates              : str   (default str(pca_dim))
    max_classes                     : int   (default 1000)  — upper bound for Phase-2 search
    num_labeled_classes             : int   — k_true; set by get_class_splits
    train_classes                   : list  — set by get_class_splits
    -- spectral-specific --
    spectral_n_neighbors_candidates : str   (default '10')
    spectral_affinity               : str   (default 'knn_euclidean')
    spectral_gamma                  : float (default None → 1/n_features)
    spectral_degree                 : float (default 3.0)
    spectral_coef0                  : float (default 1.0)
    spectral_assign_labels          : str   (default 'kmeans')
    -- agglo-specific --
    agglo_linkages                  : str   (default 'ward,complete,average')
    agglo_metric                    : str   (default 'euclidean')
    -- gmm-specific --
    gmm_covariance_type_candidates  : str   (default 'full,diag')
    gmm_init_params_candidates      : str   (default 'kmeans')

    Returns
    -------
    k_hat   : int
    k_novel : int
    result  : dict with keys all_feats, all_labels, mask_lab, preds, pca, fitted_model
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Parse args
    # ------------------------------------------------------------------
    cluster_method    = getattr(args, 'final_cluster_method', 'auto')
    if cluster_method == 'auto':
        cluster_method = 'kmeans'

    tuning_criterion  = getattr(args, 'tuning_criterion', CRITERION_LABELLED_CLUSTER)
    pca_dim           = int(getattr(args, 'pca_dim', 64))
    tune_pca          = bool(getattr(args, 'tune_pca', False))
    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', str(pca_dim))
    k_true            = int(getattr(args, 'num_labeled_classes', len(args.train_classes)))
    k_max             = int(getattr(args, 'max_classes', 1000))

    pca_dims = (
        [int(x.strip()) for x in pca_dim_cands_str.split(',')]
        if tune_pca else [pca_dim]
    )

    # Spectral params
    n_neighbors       = int(getattr(args, 'spectral_n_neighbors', 10))
    nn_cands_str      = getattr(args, 'spectral_n_neighbors_candidates', str(n_neighbors))
    affinity_type     = getattr(args, 'spectral_affinity', 'knn_euclidean')
    gamma             = getattr(args, 'spectral_gamma', None)
    degree            = float(getattr(args, 'spectral_degree', 3.0))
    coef0             = float(getattr(args, 'spectral_coef0', 1.0))
    assign_labels_arg = getattr(args, 'spectral_assign_labels', 'kmeans')
    assign_labels_candidates = ['kmeans', 'discretize', 'cluster_qr']
    if assign_labels_arg and assign_labels_arg != 'kmeans':
        assign_labels_candidates = [assign_labels_arg]

    # Agglo params
    agglo_linkages_str = getattr(args, 'agglo_linkages', 'ward,complete,average')
    agglo_metric       = getattr(args, 'agglo_metric', 'euclidean')
    linkages = [s.strip() for s in agglo_linkages_str.split(',')]
    if agglo_metric == 'cosine' and 'ward' in linkages:
        linkages = [l for l in linkages if l != 'ward']
        print("  [brent_tuned] Warning: 'ward' linkage is incompatible with "
              "cosine metric — removing 'ward' from linkage candidates.")

    # GMM params
    gmm_cov_str    = getattr(args, 'gmm_covariance_type_candidates', 'full,diag')
    gmm_init_str   = getattr(args, 'gmm_init_params_candidates', 'kmeans')
    gmm_cov_types  = [s.strip() for s in gmm_cov_str.split(',')]
    gmm_init_types = [s.strip() for s in gmm_init_str.split(',')]

    nn_list = [int(x.strip()) for x in nn_cands_str.split(',')]

    # ------------------------------------------------------------------
    # 1. Collect and L2-normalise features
    # ------------------------------------------------------------------
    print(f"BrentTuned: collecting features (method={cluster_method})...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask = \
        collect_and_split_features(train_loader, device)
    l_feats_raw = normalize(l_feats_raw, norm='l2', axis=1).astype(np.float32)
    u_feats_raw = normalize(u_feats_raw, norm='l2', axis=1).astype(np.float32)
    n_l = int(all_mask.sum())
    print(f"  labelled={n_l}, unlabelled={len(u_feats_raw)}, "
          f"dim={l_feats_raw.shape[1]}, k_true={k_true}, k_max={k_max}")

    # ------------------------------------------------------------------
    # 2. Pre-collect test features if needed for Phase-1 criterion
    # ------------------------------------------------------------------
    test_feats_raw_np = None
    test_labels_np    = None
    if tuning_criterion == CRITERION_TEST:
        if test_loader is None:
            raise ValueError(
                "estimate_k_brent_tuned: test_loader must be provided when "
                f"tuning_criterion={tuning_criterion!r}. "
                "Pass test_loader from estimate_k.py."
            )
        print("  Pre-collecting test features for Phase-1 test_cluster criterion...")
        test_feats_raw_np, test_labels_np = collect_test_features_np(
            test_loader, pca=None, device=device
        )

    # ------------------------------------------------------------------
    # 3. Pre-cache PCA transforms (fit on labelled only, once per unique pca_dim)
    # ------------------------------------------------------------------
    print(f"  Pre-computing PCA for {len(set(pca_dims))} unique dim(s): "
          f"{sorted(set(pca_dims))}")
    _pca_cache = {}
    for pd in sorted(set(pca_dims)):
        _pca_cache[pd] = _apply_pca_candidate(l_feats_raw, u_feats_raw, pd)

    # ------------------------------------------------------------------
    # 4. Build Phase-1 param grid
    # ------------------------------------------------------------------
    if cluster_method == 'spectral':
        param_grid = [
            {'n_neighbors': nn, 'pca_dim': pd, 'gamma': gamma,
             'assign_labels': al}
            for nn, pd, al in itertools.product(nn_list, pca_dims,
                                                assign_labels_candidates)
        ]
        print(f"  Phase-1 grid (spectral): {len(nn_list)} n_neighbors × "
              f"{len(pca_dims)} pca_dim(s) × "
              f"{len(assign_labels_candidates)} assign_labels "
              f"= {len(param_grid)} candidates")
    elif cluster_method == 'agglo':
        param_grid = [
            {'linkage': lnk, 'pca_dim': pd}
            for lnk, pd in itertools.product(linkages, pca_dims)
        ]
        print(f"  Phase-1 grid (agglo): {len(linkages)} linkages × "
              f"{len(pca_dims)} pca_dim(s) = {len(param_grid)} candidates")
    elif cluster_method == 'gmm':
        param_grid = [
            {'covariance_type': ct, 'init_params': ip, 'pca_dim': pd}
            for ct, ip, pd in itertools.product(gmm_cov_types, gmm_init_types,
                                                pca_dims)
        ]
        print(f"  Phase-1 grid (gmm): {len(gmm_cov_types)} cov_types × "
              f"{len(gmm_init_types)} init_params × "
              f"{len(pca_dims)} pca_dim(s) = {len(param_grid)} candidates")
    else:  # kmeans
        param_grid = [{'pca_dim': pd} for pd in pca_dims]
        print(f"  Phase-1 grid (kmeans): {len(pca_dims)} pca_dim(s) "
              f"= {len(param_grid)} candidates")

    # ------------------------------------------------------------------
    # 5. Phase 1: score each param config by clustering labelled data at k_true
    # ------------------------------------------------------------------
    print(f"\n--- Phase 1: Hyperparameter tuning (k fixed = {k_true}, "
          f"criterion={tuning_criterion}) ---")

    best_params = None
    best_score  = -1.0

    for i, params in enumerate(param_grid):
        Xp_l = _pca_cache[params['pca_dim']][0]
        try:
            if cluster_method == 'spectral':
                preds_l = _phase1_cluster_spectral(
                    Xp_l, k_true, params, affinity_type, degree, coef0)
            elif cluster_method == 'agglo':
                preds_l = _phase1_cluster_agglo(Xp_l, k_true, params, agglo_metric)
            elif cluster_method == 'gmm':
                preds_l = _phase1_cluster_gmm(Xp_l, k_true, params)
            else:
                preds_l = _phase1_cluster_kmeans(Xp_l, k_true)
        except Exception as exc:
            print(f"  [Phase 1] WARNING: candidate {i+1}/{len(param_grid)} "
                  f"params={params} failed — skipping. Error: {exc}")
            continue

        if tuning_criterion == CRITERION_LABELLED_CLUSTER:
            score = float(cluster_acc(l_targets.astype(int), preds_l.astype(int)))
        elif tuning_criterion == CRITERION_TEST:
            # Cluster full data at k_true with this config, score on test
            Xp_l_pd, Xp_u_pd, pca_pd = _pca_cache[params['pca_dim']]
            X_all_pd = np.vstack([Xp_l_pd, Xp_u_pd])
            mask_lab_lu = np.array([True] * n_l + [False] * len(u_feats_raw))
            try:
                if cluster_method == 'spectral':
                    n_nb = min(params.get('n_neighbors', n_neighbors),
                               max(2, X_all_pd.shape[0] - 1))
                    A_pd = _build_affinity(X_all_pd, affinity_type,
                                          n_neighbors=n_nb,
                                          gamma=params.get('gamma', gamma),
                                          degree=degree, coef0=coef0)
                    A_dense_pd = A_pd.toarray() if hasattr(A_pd, 'toarray') \
                        else np.asarray(A_pd)
                    sc_pd = SpectralClustering(
                        n_clusters=k_true, affinity='precomputed',
                        assign_labels=params.get('assign_labels', 'kmeans'),
                        random_state=0,
                    )
                    preds_all = sc_pd.fit_predict(A_dense_pd).astype(np.int64)
                    fitted_pd = None
                else:
                    preds_all, fitted_pd = _phase2_cluster(
                        X_all_pd, k_true, cluster_method, params,
                        agglo_metric=agglo_metric, return_model=True)
            except Exception as exc2:
                print(f"  [Phase 1/test] WARNING: candidate {i+1} full-cluster "
                      f"failed — skipping. Error: {exc2}")
                continue
            score = test_cluster_score(
                X_all_pd, preds_all,
                test_feats_raw_np, test_labels_np,
                args.num_labeled_classes,
                pca=pca_pd,
                fitted_model=fitted_pd,
            )
        else:
            raise ValueError(
                f"estimate_k_brent_tuned: unsupported tuning_criterion="
                f"{tuning_criterion!r} for Phase 1. "
                "Use 'labelled_cluster' or 'test_cluster'."
            )

        print(f"  [{i+1}/{len(param_grid)}] params={params} → score={score:.4f}")
        if score > best_score:
            best_score  = score
            best_params = params

    if best_params is None:
        raise RuntimeError(
            "estimate_k_brent_tuned: all Phase-1 candidates failed. "
            "Check warnings above."
        )

    print(f"\n  Phase-1 best: params={best_params}, score={best_score:.4f}")

    # ------------------------------------------------------------------
    # 6. Phase 2: Brent search on full data with best params
    # ------------------------------------------------------------------
    print(f"\n--- Phase 2: Brent K search (k_min={k_true}, k_max={k_max}) ---")

    # Apply best PCA to full features
    Xp_l_best, _Xp_u_best_unused, pca_best = _pca_cache[best_params['pca_dim']]
    if pca_best is not None:
        from sklearn.preprocessing import normalize as sk_normalize
        Xp_u_best = pca_best.transform(u_feats_raw)
        Xp_u_best = sk_normalize(Xp_u_best, norm='l2', axis=1).astype(np.float32)
    else:
        Xp_u_best = normalize(u_feats_raw, norm='l2', axis=1).astype(np.float32)

    all_feats_lu = np.vstack([Xp_l_best, Xp_u_best])  # [l | u] order
    all_labels_lu = np.concatenate([l_targets.astype(np.int64),
                                    u_targets.astype(np.int64)])
    mask_lab_lu = np.array([True] * n_l + [False] * len(u_feats_raw), dtype=bool)

    # For spectral: pre-build affinity matrix ONCE (most expensive part)
    prebuilt_affinity = None
    if cluster_method == 'spectral':
        n_nb_best = min(best_params.get('n_neighbors', n_neighbors),
                        max(2, all_feats_lu.shape[0] - 1))
        print(f"  Building affinity matrix once "
              f"(affinity={affinity_type}, n_neighbors={n_nb_best}, "
              f"pca_dim={best_params['pca_dim']})...")
        A_all = _build_affinity(all_feats_lu, affinity_type,
                                n_neighbors=n_nb_best,
                                gamma=best_params.get('gamma', gamma),
                                degree=degree, coef0=coef0)
        prebuilt_affinity = A_all.toarray() if hasattr(A_all, 'toarray') \
            else np.asarray(A_all)

    def _neg_score(K):
        k = max(k_true, min(int(round(K)), k_max))
        try:
            preds = _phase2_cluster(
                all_feats_lu, k, cluster_method, best_params,
                prebuilt_affinity=prebuilt_affinity,
                agglo_metric=agglo_metric,
            )
        except Exception as exc:
            print(f"  [Phase 2] k={k} failed: {exc} — returning 0 score")
            return 0.0
        acc = float(cluster_acc(
            all_labels_lu[mask_lab_lu].astype(int),
            preds[mask_lab_lu].astype(int),
        ))
        print(f"  k={k}: labelled_acc={acc:.4f}")
        return -acc

    res = minimize_scalar(
        _neg_score,
        bounds=(k_true, k_max),
        method='bounded',
        options={'disp': True},
    )
    k_hat = max(k_true + 1, int(round(res.x)))
    print(f"\nOptimal K is {k_hat}")

    # ------------------------------------------------------------------
    # 7. Final clustering at k_hat with best params
    # ------------------------------------------------------------------
    preds_lu, fitted_model = _phase2_cluster(
        all_feats_lu, k_hat, cluster_method, best_params,
        prebuilt_affinity=prebuilt_affinity,
        agglo_metric=agglo_metric,
        return_model=True,
    )

    print(f"Best params: {_json.dumps(best_params)}")

    k_novel = max(1, k_hat - k_true)

    # ------------------------------------------------------------------
    # 8. Scatter [l|u] ordering back to original loader ordering (all_mask)
    # ------------------------------------------------------------------
    feat_dim = all_feats_lu.shape[1]

    Xp_full = np.empty((len(all_mask), feat_dim), dtype=np.float32)
    Xp_full[all_mask]  = all_feats_lu[:n_l]
    Xp_full[~all_mask] = all_feats_lu[n_l:]

    labels_full = np.empty(len(all_mask), dtype=np.int64)
    labels_full[all_mask]  = l_targets.astype(np.int64)
    labels_full[~all_mask] = u_targets.astype(np.int64)

    preds_full = np.empty(len(all_mask), dtype=np.int64)
    preds_full[all_mask]  = preds_lu[:n_l]
    preds_full[~all_mask] = preds_lu[n_l:]

    result = dict(
        all_feats=Xp_full,
        all_labels=labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=pca_best,
        fitted_model=fitted_model,
    )
    return k_hat, k_novel, result
