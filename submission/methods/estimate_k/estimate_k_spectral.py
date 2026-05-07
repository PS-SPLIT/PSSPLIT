"""
estimate_k_spectral.py — Spectral eigengap k estimation method.

Builds an affinity graph over the combined labelled + unlabelled features,
computes the normalised graph Laplacian, and selects K via the largest
eigengap in the smallest Kmax+2 eigenvalues.

Affinity graph types (--spectral_affinity)
------------------------------------------
knn_euclidean  : Binary kNN connectivity graph using Euclidean distances (default).
                 Uses sklearn's kneighbors_graph — identical to sklearn's own
                 SpectralClustering(affinity='nearest_neighbors') internally.
knn_cosine     : Same but using cosine distances for the kNN graph.  Useful when
                 features are L2-normalised and directional structure matters.
rbf            : Full pairwise RBF kernel — exp(-gamma * ||x-y||^2).  O(N^2) memory;
                 not recommended for N > 20,000.  Tune gamma via
                 --spectral_gamma / --spectral_gamma_candidates.

The following sklearn pairwise kernels are also supported (all O(N^2) memory):
cosine         : Full pairwise cosine similarity (zero diagonal zeroed out).
polynomial     : (gamma * x·y + coef0)^degree.  Tune via --spectral_gamma,
                 --spectral_degree, --spectral_coef0.
sigmoid        : tanh(gamma * x·y + coef0).  May produce negative values; zeroed out.
laplacian      : exp(-gamma * ||x-y||_1).  L1-distance variant of RBF.

Note: 'chi2' and 'additive_chi2' kernels are NOT supported — they require non-negative
inputs, but L2-normalised DINOv2 embeddings can be negative.

Assign-labels method (--spectral_assign_labels)
------------------------------------------------
kmeans     : default — robust, matches sklearn SpectralClustering default.
discretize : graph-partitioning-based rounding; sometimes better on non-convex clusters.
cluster_qr : QR decomposition based; fastest of the three.

Tunable hyperparameters
-----------------------
n_neighbors : number of nearest neighbours for kNN affinity graphs.
              Controlled via --spectral_n_neighbors (single value) or
              --spectral_n_neighbors_candidates (comma-separated list).
              Ignored when --spectral_affinity rbf.
gamma       : RBF kernel bandwidth.  Controlled via --spectral_gamma (single float)
              or --spectral_gamma_candidates (comma-sep; 'auto' = 1/n_features).
              Only used when --spectral_affinity rbf.
pca_dim     : PCA dimensionality before building the affinity graph.
              Controlled via --pca_dim (single value) or, when --tune_pca is
              set, --pca_dim_candidates (comma-separated list).

Tuning criterion
----------------
Controlled by --tuning_criterion (default: 'labelled_k'):

  labelled_k   : run eigengap on *labelled* features only; score = how close
                 k_hat is to k_true.  Checks whether the affinity config can
                 recover known class structure (reasonable proxy for novel
                 class detection quality).
                 NOTE: eigengap on labelled-only does not see novel class
                 structure, so this is an indirect proxy.

  test_cluster : run eigengap on full X_all; cluster X_all; score = old-class
                 ACC on test split via nearest-centroid.

  combined     : weighted combination of both scores.
"""

import itertools

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from methods.estimate_k.estimate_k_agglo import (
    collect_and_split_features,
    apply_pca_and_norm,
)
from methods.estimate_k.estimate_k_tuning import (
    CRITERION_LABELLED, CRITERION_TEST, CRITERION_COMBINED,
    collect_test_features_np,
    select_best_config,
)


# ---------------------------------------------------------------------------
# Affinity constructors
# ---------------------------------------------------------------------------

def _knn_affinity(X, n_neighbors=10, metric='euclidean'):
    """
    Build a symmetric binary kNN affinity matrix (0/1 weights).

    Uses sklearn's kneighbors_graph — same as SpectralClustering(affinity='nearest_neighbors')
    internally. Supports metric='euclidean' or 'cosine'.

    Parameters
    ----------
    metric : 'euclidean' or 'cosine' (or any metric accepted by kneighbors_graph)
    """
    n_neighbors = max(2, min(n_neighbors, X.shape[0] - 1))
    A = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric,
                         mode='connectivity', include_self=False)
    A = 0.5 * (A + A.T)
    return A


def _rbf_affinity(X, gamma=None):
    """
    Full pairwise RBF affinity matrix.

    A(i,j) = exp(-gamma * ||x_i - x_j||^2),  self-affinities zeroed out.
    gamma=None → 1 / n_features  (sklearn convention).

    WARNING: O(N^2) memory — not suitable for N > ~20,000.
    """
    N = X.shape[0]
    if N > 20_000:
        import warnings
        warnings.warn(
            f"_rbf_affinity: N={N} > 20,000 — building a full {N}x{N} affinity "
            "matrix will require significant memory. Consider using knn_euclidean "
            "or knn_cosine instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    A = rbf_kernel(X, gamma=gamma)
    np.fill_diagonal(A, 0.0)
    return csr_matrix(A)


_SKLEARN_KERNEL_AFFINITIES = ('cosine', 'polynomial', 'sigmoid', 'laplacian')

_ALL_AFFINITY_TYPES = (
    'knn_euclidean', 'knn_cosine', 'rbf',
    'cosine', 'polynomial', 'sigmoid', 'laplacian',
)


def _sklearn_kernel_affinity(X, kernel_type, gamma=None, degree=3.0, coef0=1.0):
    """
    Full pairwise affinity via sklearn pairwise_kernels.

    Kernels: 'cosine' | 'polynomial' | 'sigmoid' | 'laplacian'

    All are O(N^2) memory — not suitable for N > ~20,000.
    Negative values (can occur for 'sigmoid') are zeroed out before returning.

    Parameters
    ----------
    kernel_type : str — one of _SKLEARN_KERNEL_AFFINITIES
    gamma       : float or None — kernel coefficient; None → 1/n_features
    degree      : float — degree for 'polynomial' kernel (ignored by others)
    coef0       : float — zero coefficient for 'polynomial' and 'sigmoid' (ignored by others)
    """
    from sklearn.metrics.pairwise import pairwise_kernels

    N = X.shape[0]
    if N > 20_000:
        import warnings
        warnings.warn(
            f"_sklearn_kernel_affinity: N={N} > 20,000 — building a full {N}×{N} "
            f"affinity matrix with kernel='{kernel_type}' will require significant memory. "
            "Consider using knn_euclidean or knn_cosine instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Build only the kwargs each kernel accepts:
    #   polynomial : gamma, degree, coef0
    #   sigmoid    : gamma, coef0
    #   laplacian  : gamma
    #   cosine     : (none — pure normalised dot product)
    if kernel_type == 'polynomial':
        kwds = {'gamma': gamma, 'degree': degree, 'coef0': coef0}
    elif kernel_type == 'sigmoid':
        kwds = {'gamma': gamma, 'coef0': coef0}
    elif kernel_type == 'laplacian':
        kwds = {'gamma': gamma}
    else:  # 'cosine'
        kwds = {}

    A = pairwise_kernels(X, metric=kernel_type, **kwds)
    np.fill_diagonal(A, 0.0)
    # Zero out any negative values (can occur for sigmoid)
    np.clip(A, 0.0, None, out=A)
    return csr_matrix(A)


def _build_affinity(X, affinity_type, n_neighbors=10, gamma=None,
                    degree=3.0, coef0=1.0):
    """
    Dispatch to the correct affinity constructor.

    Parameters
    ----------
    affinity_type : one of _ALL_AFFINITY_TYPES
    n_neighbors   : used for knn_* types
    gamma         : used for rbf, cosine, polynomial, sigmoid, laplacian
    degree        : used for polynomial (ignored by others)
    coef0         : used for polynomial and sigmoid (ignored by others)
    """
    if affinity_type == 'knn_euclidean':
        return _knn_affinity(X, n_neighbors=n_neighbors, metric='euclidean')
    elif affinity_type == 'knn_cosine':
        return _knn_affinity(X, n_neighbors=n_neighbors, metric='cosine')
    elif affinity_type == 'rbf':
        return _rbf_affinity(X, gamma=gamma)
    elif affinity_type in _SKLEARN_KERNEL_AFFINITIES:
        return _sklearn_kernel_affinity(X, affinity_type, gamma=gamma,
                                        degree=degree, coef0=coef0)
    else:
        raise ValueError(
            f"Unknown spectral_affinity={affinity_type!r}. "
            f"Expected one of: {_ALL_AFFINITY_TYPES}."
        )


# ---------------------------------------------------------------------------
# Eigengap helpers
# ---------------------------------------------------------------------------

def eigengap_select_K_from_affinity(A, Kmax=20):
    """
    Select K using the largest eigengap of the normalised graph Laplacian,
    given a pre-built affinity matrix A (csr_matrix or dense ndarray).

    Returns
    -------
    k_hat : int  (1 ≤ k_hat ≤ Kmax)
    """
    N = A.shape[0]
    if N < 3:
        return 1

    L = csgraph_laplacian(A, normed=True)

    k_eigs = min(Kmax + 2, N - 1)
    try:
        vals = np.sort(
            eigsh(L, k=k_eigs, which='SM', return_eigenvectors=False)
        )
    except ArpackNoConvergence as exc:
        # Use whatever eigenvalues converged; fall back to k=1 if too few.
        vals = np.sort(exc.eigenvalues) if len(exc.eigenvalues) >= 2 else None
        if vals is None:
            return 1

    gaps = np.diff(vals[: Kmax + 1])
    if gaps.size == 0:
        return 1

    k_hat = int(np.argmax(gaps) + 1)
    return max(1, min(k_hat, Kmax))


def eigengap_select_K(X, Kmax=20, n_neighbors=10):
    """
    Backward-compatible wrapper: build binary kNN Euclidean affinity then run eigengap.

    Parameters
    ----------
    X          : (N, D) float32 feature matrix
    Kmax       : upper bound for k
    n_neighbors: kNN graph size
    """
    if X.shape[0] < 3:
        return 1
    A = _knn_affinity(X, n_neighbors=n_neighbors, metric='euclidean')
    return eigengap_select_K_from_affinity(A, Kmax=Kmax)


# ---------------------------------------------------------------------------
# Gamma candidate parser
# ---------------------------------------------------------------------------

def _parse_gamma_candidates(cands_str, gamma_default, n_features):
    """
    Parse a comma-separated string of gamma candidates.

    'auto' expands to 1/n_features.
    Returns a list of floats.
    """
    tokens = [t.strip() for t in cands_str.split(',')]
    result = []
    for t in tokens:
        if t == 'auto':
            result.append(1.0 / n_features)
        else:
            result.append(float(t))
    if not result:
        result = [gamma_default if gamma_default is not None else 1.0 / n_features]
    return result


# ---------------------------------------------------------------------------
# Internal PCA helper
# ---------------------------------------------------------------------------

def _apply_pca_candidate(l_feats_raw, u_feats_raw, pca_dim):
    """Apply PCA (or just L2-norm) for a given pca_dim candidate."""
    if pca_dim > 0:
        return apply_pca_and_norm(l_feats_raw, u_feats_raw, pca_dim)
    else:
        Xp_l = normalize(l_feats_raw, norm='l2', axis=1).astype(np.float32)
        Xp_u = normalize(u_feats_raw, norm='l2', axis=1).astype(np.float32)
        return Xp_l, Xp_u, None


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_k_spectral(train_loader, args, device=None, test_loader=None):
    """
    Estimate K using the spectral eigengap method.

    Prints "Optimal K is X" for parsing by run_pipeline.py.

    Parameters
    ----------
    train_loader : DataLoader over a MergedDataset FeatureVectorDataset
    args         : namespace  (uses pca_dim, max_classes, spectral_n_neighbors,
                               train_classes, num_labeled_classes)
    device       : torch.device or None
    test_loader  : DataLoader for test split; required when
                   args.tuning_criterion is 'test_cluster' or 'combined'

    Optional args fields
    --------------------
    tuning_criterion                : 'labelled_k' (default) | 'test_cluster' | 'combined'
    tuning_weight                   : float  (default 0.5)
    tune_pca                        : bool   (default False)
    pca_dim_candidates              : str    (default str(pca_dim))
    spectral_n_neighbors_candidates : str    (default str(spectral_n_neighbors))
    spectral_affinity               : one of _ALL_AFFINITY_TYPES (default 'knn_euclidean')
    spectral_gamma                  : float  (default None → 1/n_features)
    spectral_gamma_candidates       : str    (default 'auto'; non-knn types)
    spectral_degree                 : float  (default 3.0; polynomial only)
    spectral_coef0                  : float  (default 1.0; polynomial/sigmoid only)
    spectral_assign_labels          : 'kmeans' (default) | 'discretize' | 'cluster_qr'

    Returns
    -------
    estimated_k_total : int
    estimated_C_u     : int
    result            : dict with keys all_feats, all_labels, mask_lab, preds, pca,
                        fitted_model
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pca_dim      = int(getattr(args, 'pca_dim', 64))
    kmax         = int(getattr(args, 'max_classes', 50))
    n_neighbors  = int(getattr(args, 'spectral_n_neighbors', 10))
    k_l          = len(args.train_classes)

    tuning_criterion  = getattr(args, 'tuning_criterion', CRITERION_LABELLED)
    tuning_weight     = float(getattr(args, 'tuning_weight', 0.5))
    tune_pca          = bool(getattr(args, 'tune_pca', False))
    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', str(pca_dim))
    nn_cands_str      = getattr(args, 'spectral_n_neighbors_candidates',
                                str(n_neighbors))

    # New options
    affinity_type   = getattr(args, 'spectral_affinity', 'knn_euclidean')
    gamma           = getattr(args, 'spectral_gamma', None)
    gamma_cands_str = getattr(args, 'spectral_gamma_candidates', 'auto')
    degree          = float(getattr(args, 'spectral_degree', 3.0))
    coef0           = float(getattr(args, 'spectral_coef0', 1.0))
    assign_labels   = getattr(args, 'spectral_assign_labels', 'kmeans')

    pca_dims    = (
        [int(x.strip()) for x in pca_dim_cands_str.split(',')]
        if tune_pca else [pca_dim]
    )
    nn_list     = [int(x.strip()) for x in nn_cands_str.split(',')]

    # ------------------------------------------------------------------
    # 1. Load raw features
    # ------------------------------------------------------------------
    print("Spectral eigengap k-estimation: collecting features...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask = \
        collect_and_split_features(train_loader, device)
    l_feats_raw = normalize(l_feats_raw, norm='l2', axis=1).astype(np.float32)
    u_feats_raw = normalize(u_feats_raw, norm='l2', axis=1).astype(np.float32)
    print(f"  labelled={len(l_feats_raw)}, unlabelled={len(u_feats_raw)}, "
          f"dim={l_feats_raw.shape[1]}, affinity={affinity_type}, "
          f"assign_labels={assign_labels}")

    # ------------------------------------------------------------------
    # 2. Pre-collect test features if needed
    # ------------------------------------------------------------------
    test_feats_raw = None
    test_labels    = None
    if tuning_criterion in (CRITERION_TEST, CRITERION_COMBINED):
        if test_loader is None:
            raise ValueError(
                "estimate_k_spectral: test_loader must be provided when "
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
    # 4. Build param grid (branches on affinity type)
    # ------------------------------------------------------------------
    if affinity_type in ('knn_euclidean', 'knn_cosine'):
        param_grid = [
            {'n_neighbors': nn, 'pca_dim': pd, 'gamma': None}
            for nn, pd in itertools.product(nn_list, pca_dims)
        ]
        print(f"  Tuning: {len(nn_list)} n_neighbors × {len(pca_dims)} pca_dim(s) "
              f"= {len(param_grid)} candidates "
              f"(criterion={tuning_criterion}, k_l={k_l})")
    else:  # rbf and sklearn pairwise kernels
        n_features = _pca_cache[pca_dims[0]][0].shape[1]
        gammas = _parse_gamma_candidates(gamma_cands_str, gamma, n_features)
        param_grid = [
            {'n_neighbors': n_neighbors, 'pca_dim': pd, 'gamma': g}
            for g, pd in itertools.product(gammas, pca_dims)
        ]
        print(f"  Tuning ({affinity_type}): {len(gammas)} gamma(s) × {len(pca_dims)} pca_dim(s) "
              f"= {len(param_grid)} candidates "
              f"(criterion={tuning_criterion}, k_l={k_l})")

    # ------------------------------------------------------------------
    # 5. Define labelled_fn and full_fn closures (use PCA cache)
    # ------------------------------------------------------------------

    def labelled_fn(params):
        """
        Eigengap on labelled-only features → k_hat_l.

        This is a proxy: a good config should recover the known class
        structure from labelled data.  It does not see novel class structure
        (limitation noted in module docstring).
        """
        Xp_l, _Xp_u, _ = _pca_cache[params['pca_dim']]
        n_nb = min(params['n_neighbors'] or n_neighbors,
                   max(2, Xp_l.shape[0] - 1))
        A_l = _build_affinity(Xp_l, affinity_type,
                              n_neighbors=n_nb, gamma=params['gamma'],
                              degree=degree, coef0=coef0)
        return eigengap_select_K_from_affinity(A_l, Kmax=kmax)

    def full_fn(params):
        """Eigengap on X_all → k_hat; SpectralClustering → preds."""
        Xp_l, Xp_u, pca = _pca_cache[params['pca_dim']]
        X_all = np.vstack([Xp_l, Xp_u])
        n_nb = min(params['n_neighbors'] or n_neighbors,
                   max(2, X_all.shape[0] - 1))

        A_all = _build_affinity(X_all, affinity_type,
                                n_neighbors=n_nb, gamma=params['gamma'],
                                degree=degree, coef0=coef0)
        k_hat_raw = eigengap_select_K_from_affinity(A_all, Kmax=kmax)
        # Apply k_l+1 floor inside full_fn so preds are consistent with k_hat
        k_hat = max(k_hat_raw, k_l + 1)

        gamma_str = f"{params['gamma']:.4g}" if params['gamma'] is not None else "None"
        print(f"    eigengap K={k_hat_raw} → k_hat={k_hat} "
              f"(affinity={affinity_type}, n_neighbors={n_nb}, "
              f"pca_dim={params['pca_dim']}, gamma={gamma_str})")

        # Final clustering — call SpectralClustering directly so assign_labels
        # is respected (cannot go through _cluster_with_method for this).
        A_dense = A_all.toarray() if hasattr(A_all, 'toarray') else np.asarray(A_all)
        sc = SpectralClustering(
            n_clusters=k_hat,
            affinity='precomputed',
            assign_labels=assign_labels,
            random_state=0,
        )
        preds = sc.fit_predict(A_dense).astype(np.int64)
        # SpectralClustering has no .predict() → fitted_model=None
        return k_hat, preds, X_all, pca, None

    # ------------------------------------------------------------------
    # 6. Run selection
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

    import json as _json
    print(f"Best params: {_json.dumps(best.params)}")

    k_hat   = best.k_hat
    k_novel = k_hat - k_l

    print(f"  Final k_hat={k_hat}, k_novel={k_novel}")
    print(f"Optimal K is {k_hat}")

    # ------------------------------------------------------------------
    # 7. Build result dict in original loader order
    # ------------------------------------------------------------------
    n_l      = int(all_mask.sum())
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

    result = dict(
        all_feats=Xp_full,
        all_labels=labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=best.pca,
        fitted_model=best.fitted_model,
    )
    return k_hat, k_novel, result


# ---------------------------------------------------------------------------
# Fixed-K variant: tune affinity/assign_labels/n_neighbors/pca_dim at oracle K
# ---------------------------------------------------------------------------

def estimate_k_spectral_fixed_k(train_loader, args, device=None, test_loader=None, k=None):
    """
    Spectral clustering with a *fixed* K (oracle K known).

    Instead of using eigengap to select K, we pass n_clusters=k directly to
    SpectralClustering and tune over affinity × assign_labels × n_neighbors × pca_dim.

    The tuning criterion is always test_cluster (no labelled_fn needed).

    Parameters
    ----------
    k : int  the fixed number of clusters (total = labeled + unlabeled)

    Returns
    -------
    k : int  (same as input)
    k_novel : int
    result  : dict with keys all_feats, all_labels, mask_lab, preds, pca, fitted_model
    """
    import json as _json
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if k is None:
        raise ValueError("estimate_k_spectral_fixed_k: k must be provided")

    pca_dim           = int(getattr(args, 'pca_dim', 64))
    n_neighbors       = int(getattr(args, 'spectral_n_neighbors', 10))
    k_l               = len(args.train_classes)
    k_novel           = max(1, k - k_l)

    tune_pca          = bool(getattr(args, 'tune_pca', False))
    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', str(pca_dim))
    nn_cands_str      = getattr(args, 'spectral_n_neighbors_candidates', str(n_neighbors))
    affinity_type     = getattr(args, 'spectral_affinity', 'knn_euclidean')
    gamma             = getattr(args, 'spectral_gamma', None)
    gamma_cands_str   = getattr(args, 'spectral_gamma_candidates', 'auto')
    degree            = float(getattr(args, 'spectral_degree', 3.0))
    coef0             = float(getattr(args, 'spectral_coef0', 1.0))
    tuning_weight     = float(getattr(args, 'tuning_weight', 0.5))

    # For fixed-K we tune over all assign_labels options by default
    assign_labels_candidates = ['kmeans', 'discretize', 'cluster_qr']
    # Allow override via a single --spectral_assign_labels (if user wants just one)
    assign_labels_arg = getattr(args, 'spectral_assign_labels', None)
    if assign_labels_arg and assign_labels_arg != 'kmeans':
        # User specified a non-default single value — honour it
        assign_labels_candidates = [assign_labels_arg]

    pca_dims = (
        [int(x.strip()) for x in pca_dim_cands_str.split(',')]
        if tune_pca else [pca_dim]
    )
    nn_list = [int(x.strip()) for x in nn_cands_str.split(',')]

    # ------------------------------------------------------------------
    # 1. Load raw features
    # ------------------------------------------------------------------
    print(f"Spectral fixed-K clustering (k={k}): collecting features...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask = \
        collect_and_split_features(train_loader, device)
    l_feats_raw = normalize(l_feats_raw, norm='l2', axis=1).astype(np.float32)
    u_feats_raw = normalize(u_feats_raw, norm='l2', axis=1).astype(np.float32)
    print(f"  labelled={len(l_feats_raw)}, unlabelled={len(u_feats_raw)}, "
          f"dim={l_feats_raw.shape[1]}, affinity={affinity_type}")

    # ------------------------------------------------------------------
    # 2. Always use test_cluster criterion — collect test features
    # ------------------------------------------------------------------
    if test_loader is None:
        raise ValueError(
            "estimate_k_spectral_fixed_k: test_loader must be provided. "
            "Fixed-K tuning always uses test_cluster criterion."
        )
    print("  Pre-collecting test features for test_cluster criterion...")
    test_feats_raw, test_labels = collect_test_features_np(
        test_loader, pca=None, device=device
    )

    # ------------------------------------------------------------------
    # 3. Pre-cache PCA transforms
    # ------------------------------------------------------------------
    print(f"  Pre-computing PCA for {len(set(pca_dims))} unique dim(s): "
          f"{sorted(set(pca_dims))}")
    _pca_cache = {}
    for pd in sorted(set(pca_dims)):
        _pca_cache[pd] = _apply_pca_candidate(l_feats_raw, u_feats_raw, pd)

    # ------------------------------------------------------------------
    # 4. Build param grid
    # ------------------------------------------------------------------
    if affinity_type in ('knn_euclidean', 'knn_cosine'):
        param_grid = [
            {'n_neighbors': nn, 'pca_dim': pd, 'gamma': None, 'assign_labels': al}
            for nn, pd, al in itertools.product(nn_list, pca_dims, assign_labels_candidates)
        ]
        print(f"  Fixed-K tuning: {len(nn_list)} n_neighbors × {len(pca_dims)} pca_dim(s) × "
              f"{len(assign_labels_candidates)} assign_labels = {len(param_grid)} candidates "
              f"(k={k}, affinity={affinity_type}, criterion=test_cluster)")
    else:  # rbf and sklearn pairwise kernels
        n_features = _pca_cache[pca_dims[0]][0].shape[1]
        gammas = _parse_gamma_candidates(gamma_cands_str, gamma, n_features)
        param_grid = [
            {'n_neighbors': n_neighbors, 'pca_dim': pd, 'gamma': g, 'assign_labels': al}
            for g, pd, al in itertools.product(gammas, pca_dims, assign_labels_candidates)
        ]
        print(f"  Fixed-K tuning ({affinity_type}): {len(gammas)} gamma(s) × "
              f"{len(pca_dims)} pca_dim(s) × "
              f"{len(assign_labels_candidates)} assign_labels = {len(param_grid)} candidates "
              f"(k={k}, criterion=test_cluster)")

    # ------------------------------------------------------------------
    # 5. Define full_fn (no labelled_fn — test_cluster criterion only)
    # ------------------------------------------------------------------

    def full_fn(params):
        """SpectralClustering with fixed n_clusters=k, tuning over other params."""
        Xp_l, Xp_u, pca = _pca_cache[params['pca_dim']]
        X_all = np.vstack([Xp_l, Xp_u])
        n_nb = min(params['n_neighbors'] or n_neighbors,
                   max(2, X_all.shape[0] - 1))

        A_all = _build_affinity(X_all, affinity_type,
                                n_neighbors=n_nb, gamma=params['gamma'],
                                degree=degree, coef0=coef0)
        A_dense = A_all.toarray() if hasattr(A_all, 'toarray') else np.asarray(A_all)

        sc = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            assign_labels=params['assign_labels'],
            random_state=0,
        )
        preds = sc.fit_predict(A_dense).astype(np.int64)

        gamma_str = f"{params['gamma']:.4g}" if params['gamma'] is not None else "None"
        print(f"    spectral fixed-K: affinity={affinity_type}, "
              f"n_neighbors={n_nb}, pca_dim={params['pca_dim']}, "
              f"assign_labels={params['assign_labels']}, gamma={gamma_str}")
        return k, preds, X_all, pca, None

    # ------------------------------------------------------------------
    # 6. Run selection (test_cluster criterion, labelled_fn=None)
    # ------------------------------------------------------------------
    best = select_best_config(
        param_grid=param_grid,
        k_true=k_l,
        criterion=CRITERION_TEST,
        weight=tuning_weight,
        labelled_fn=None,
        full_fn=full_fn,
        test_feats_raw=test_feats_raw,
        test_labels=test_labels,
        num_labeled_classes=args.num_labeled_classes,
        skip_on_error=True,
    )

    print(f"Best params: {_json.dumps(best.params)}")
    print(f"  fixed k={k}, k_novel={k_novel}")
    print(f"Optimal K is {k}")

    # ------------------------------------------------------------------
    # 7. Build result dict in original loader order
    # ------------------------------------------------------------------
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

    result = dict(
        all_feats=Xp_full,
        all_labels=labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=best.pca,
        fitted_model=best.fitted_model,
    )
    return k, k_novel, result
