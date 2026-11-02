"""
estimate_k_hier_tuning.py — Tuning-based per-coarse K estimation for hierarchical GCD.

Three tuning strategies:

  A1 (agglo)    : Tune AgglomerativeClustering distance threshold on known coarses.
                  Ward linkage, no PCA. Best threshold applied to each unknown coarse.

  A2 (spectral) : Tune minimum eigengap threshold on known coarses.
                  Pre-computes affinity + eigenvalues per known coarse once, then
                  loops over gap_threshold candidates cheaply.

  A3 (xmeans)   : Tune PCA dim before xmeans (MNDL criterion) on known coarses.
                  Best pca_dim applied to each unknown coarse.

In all cases, hyperparameters are tuned on known coarses (where fine-class labels
are available for labelled_k_score), then the best config is applied to each
unknown coarse independently.

Output:
  args.k_results_path               — JSON {str(coarse_id): int(k_hat)}
  args.k_results_path (metrics)     — comprehensive metrics JSON via save_hier_metrics
  stdout: "K estimation complete: {args.k_results_path}"

Run standalone:
    python -m methods.estimate_k_hier.estimate_k_hier_tuning \\
        --dataset_name aves_small \\
        --coarse_labels_path data/coarse_labels/aves_small.json \\
        --k_results_path /tmp/aves_small_k_tuning.json \\
        --estimate_k_method agglo
"""

import argparse
import json
import os
import warnings as _warnings

import numpy as np
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
import torch

from config import feature_extract_dir
from data.data_utils import build_coarse_of, load_coarse_map, oracle_k_per_coarse
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from methods.estimate_k.estimate_k_spectral import _knn_affinity
from methods.estimate_k.estimate_k_tuning import labelled_k_score
from methods.estimate_k_hier.estimate_k_hier import load_all_features
from methods.estimate_k_hier.hier_eval_utils import (
    eval_per_coarse, aggregate_metrics, save_hier_metrics, print_aggregate_summary,
    CLUSTER_METHODS,
)
from project_utils.cluster_utils import cluster_acc

# Module-level random seed propagated to sklearn / pyclustering random_state
# args. Overwritten from args.seed by main(). 0 reproduces historical behaviour.
_RANDOM_STATE = 0
from project_utils.general_utils import str2bool

# pyclustering monkey-patch (same as estimate_k_xgmeans.py)
if not hasattr(np, 'warnings'):
    np.warnings = _warnings


# ---------------------------------------------------------------------------
# Shared: per-coarse feature lists
# ---------------------------------------------------------------------------

def _build_coarse_structures(all_feats, all_labels, all_mask, coarse_of):
    """
    Partition samples by coarse ID into known / unknown coarses.

    Returns
    -------
    label_to_coarse : dict {int(label) -> int(coarse_id)}
    known_coarses   : set of int coarse IDs where all fine classes are labeled
    unknown_coarses : set of int coarse IDs where all fine classes are unlabeled
    known_coarse_feats_list : list of (feats, k_true) for known coarses (sorted by coarse_id)
    oracle_k        : dict {int(coarse_id): int(k_true)} for unknown coarses
    """
    unique_labels = np.unique(all_labels)
    label_to_coarse = {int(lbl): int(coarse_of[lbl]) for lbl in unique_labels}

    known_coarses, unknown_coarses = set(), set()
    for lbl in unique_labels:
        c = label_to_coarse[int(lbl)]
        if all_mask[all_labels == lbl].any():
            known_coarses.add(c)
        else:
            unknown_coarses.add(c)

    mixed = known_coarses & unknown_coarses
    if mixed:
        raise ValueError(
            f"Coarse categories {mixed} contain both known and unknown fine classes. "
            "This violates the hierarchical split constraint. "
            "Use --hier_split_type 5050 (or similar) to get coarse-respecting splits."
        )

    # Build per-coarse feature list for known coarses
    known_coarse_feats_list = []
    for c in sorted(known_coarses):
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        c_feats = all_feats[c_mask].astype(np.float32)
        c_labels = all_labels[c_mask]
        k_true = int(len(np.unique(c_labels)))
        known_coarse_feats_list.append((c_feats, k_true))

    # Oracle K for unknown coarses
    oracle_k = {}
    for c in unknown_coarses:
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        oracle_k[c] = int(len(np.unique(all_labels[c_mask])))

    return label_to_coarse, known_coarses, unknown_coarses, known_coarse_feats_list, oracle_k


# ---------------------------------------------------------------------------
# Method A1: Agglomerative threshold tuning
# ---------------------------------------------------------------------------

def estimate_k_agglo(feats: np.ndarray, threshold: float) -> int:
    """
    AgglomerativeClustering with distance_threshold; return number of clusters found.

    Uses Ward linkage, Euclidean metric, no PCA.
    Guard: if len(feats) < 2, return 1.
    """
    if len(feats) < 2:
        return 1
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage='ward',
        metric='euclidean',
    )
    labels = model.fit_predict(feats)
    return int(len(np.unique(labels)))


def tune_agglo_threshold(
    known_coarse_feats_list: list,
    threshold_grid: list,
) -> float:
    """
    Return the threshold from threshold_grid that maximises mean labelled_k_score
    on known coarses.

    Parameters
    ----------
    known_coarse_feats_list : list of (feats [N,D] float32, k_true int)
    threshold_grid          : list of float thresholds to try

    Returns
    -------
    best_threshold : float
    """
    if not threshold_grid:
        raise ValueError("tune_agglo_threshold: threshold_grid is empty.")
    if not known_coarse_feats_list:
        raise ValueError("tune_agglo_threshold: known_coarse_feats_list is empty.")

    best_threshold = threshold_grid[0]
    best_score = -1.0

    for t in threshold_grid:
        scores = []
        for feats, k_true in known_coarse_feats_list:
            try:
                k_hat = estimate_k_agglo(feats, t)
                scores.append(float(labelled_k_score(k_hat, k_true)))
            except Exception as exc:
                print(f"  WARNING: agglo failed for threshold={t:.4f}: {exc}")
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        print(f"  agglo threshold={t:.4f}: mean_labelled_k_score={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_threshold = t

    print(f"Best agglo threshold={best_threshold:.4f} (score={best_score:.4f})")
    return best_threshold


def run_agglo(
    all_feats: np.ndarray,
    all_labels: np.ndarray,
    label_to_coarse: dict,
    unknown_coarses: set,
    known_coarse_feats_list: list,
    args,
) -> tuple:
    """
    Full A1 pipeline: tune threshold on known coarses, apply to unknown coarses.

    Returns ({str(coarse_id): int(k_hat)}, {int(coarse_id): np.ndarray preds}).
    The second dict contains the actual cluster assignments produced by the tuned
    threshold-based AgglomerativeClustering (the direct tuning output).
    """
    n_grid = int(getattr(args, 'agglo_n_grid', 25))
    lo = float(getattr(args, 'agglo_lo', 0.0))
    hi = float(getattr(args, 'agglo_hi', 2.0))
    threshold_grid = np.linspace(lo, hi, n_grid).tolist()

    print(f"\nTuning agglo threshold on {len(known_coarse_feats_list)} known coarses "
          f"({n_grid} candidates in [{lo:.3f}, {hi:.3f}])...")
    best_t = tune_agglo_threshold(known_coarse_feats_list, threshold_grid)

    print(f"\nApplying best threshold={best_t:.4f} to {len(unknown_coarses)} unknown coarses...")
    k_per_coarse = {}
    direct_preds = {}
    for c in sorted(unknown_coarses):
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        c_feats = all_feats[c_mask].astype(np.float32)
        if len(c_feats) < 2:
            k_per_coarse[str(c)] = 1
            direct_preds[c] = np.zeros(len(c_feats), dtype=np.int64)
            continue
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=best_t,
            linkage='ward', metric='euclidean',
        )
        preds = model.fit_predict(c_feats).astype(np.int64)
        k_hat = max(1, int(len(np.unique(preds))))
        print(f"  Coarse {c}: k_hat={k_hat}")
        k_per_coarse[str(c)] = k_hat
        direct_preds[c] = preds

    return k_per_coarse, direct_preds


# ---------------------------------------------------------------------------
# Method A2: Spectral gap threshold tuning
# ---------------------------------------------------------------------------

def eigengap_with_threshold(eigenvalues: np.ndarray, kmax: int, gap_threshold: float) -> int:
    """
    Return k by first-gap-threshold rule.

    gaps = np.diff(eigenvalues[:kmax+1])
    For each gap i, if gap >= gap_threshold: return i + 1
    Return 1 if no gap exceeds threshold.
    """
    if len(eigenvalues) < 2:
        return 1
    gaps = np.diff(eigenvalues[:kmax + 1])
    for i, g in enumerate(gaps):
        if g >= gap_threshold:
            return i + 1
    return 1


def _compute_eigenvalues_for_coarse(feats: np.ndarray, kmax: int, n_neighbors: int):
    """
    Build kNN affinity + normalized Laplacian and compute kmax+2 smallest eigenvalues.

    Returns (vals, A):
      vals : float32 array of sorted eigenvalues (may be shorter than kmax+2 if
             ArpackNoConvergence occurs). Falls back to [0.0, 1.0] if < 2 converge.
      A    : sparse kNN affinity matrix (can be reused for spectral clustering).
             Returns None when N < 3 (fallback path).
    """
    N = len(feats)
    if N < 3:
        return np.array([0.0, 1.0], dtype=np.float32), None

    A = _knn_affinity(feats, n_neighbors=n_neighbors, metric='euclidean')
    L = csgraph_laplacian(A, normed=True)
    k_eigs = min(kmax + 2, N - 1)
    try:
        vals = np.sort(eigsh(L, k=k_eigs, which='SM', return_eigenvectors=False))
    except ArpackNoConvergence as exc:
        if len(exc.eigenvalues) >= 2:
            vals = np.sort(exc.eigenvalues)
        else:
            print(f"  WARNING: ArpackNoConvergence with < 2 eigenvalues "
                  f"(N={N}). Falling back to [0.0, 1.0].")
            vals = np.array([0.0, 1.0], dtype=np.float32)
    return vals.astype(np.float32), A


def tune_spectral_gap_threshold(
    known_coarse_eigenvals: list,  # list of (eigenvalues, k_true)
    gap_threshold_grid: list,
    kmax: int,
) -> float:
    """
    Return the gap_threshold that maximises mean labelled_k_score on known coarses.

    Eigenvalues are pre-computed once outside this function; only the cheap
    eigengap_with_threshold() call runs per candidate.
    """
    if not gap_threshold_grid:
        raise ValueError("tune_spectral_gap_threshold: gap_threshold_grid is empty.")
    if not known_coarse_eigenvals:
        raise ValueError("tune_spectral_gap_threshold: known_coarse_eigenvals is empty.")

    best_threshold = gap_threshold_grid[0]
    best_score = -1.0

    for g_min in gap_threshold_grid:
        scores = []
        for vals, k_true in known_coarse_eigenvals:
            k_hat = eigengap_with_threshold(vals, kmax=kmax, gap_threshold=g_min)
            scores.append(float(labelled_k_score(k_hat, k_true)))
        mean_score = float(np.mean(scores))
        print(f"  spectral gap_threshold={g_min:.5f}: mean_labelled_k_score={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_threshold = g_min

    print(f"Best spectral gap_threshold={best_threshold:.5f} (score={best_score:.4f})")
    return best_threshold


def run_spectral(
    all_feats: np.ndarray,
    all_labels: np.ndarray,
    label_to_coarse: dict,
    known_coarse_feats_list: list,
    unknown_coarses: set,
    args,
) -> tuple:
    """
    Full A2 pipeline: pre-compute eigenvalues for known coarses, tune gap_threshold,
    then apply to each unknown coarse.

    Returns ({str(coarse_id): int(k_hat)}, {int(coarse_id): np.ndarray preds}).
    The second dict contains SpectralClustering predictions at the estimated k_hat
    using the same kNN affinity matrix built during eigengap computation.
    """
    n_grid = int(getattr(args, 'spectral_gap_n_grid', 30))
    lo = float(getattr(args, 'spectral_gap_lo', 0.001))
    hi = float(getattr(args, 'spectral_gap_hi', 1.0))
    n_neighbors = int(getattr(args, 'spectral_n_neighbors', 10))

    # kmax derived from max k_true across known coarses
    kmax = max((k_true for _, k_true in known_coarse_feats_list), default=10)
    print(f"\nSpectral: kmax={kmax} (from max known coarse k_true)")

    gap_threshold_grid = np.linspace(lo, hi, n_grid).tolist()

    # Pre-compute eigenvalues for known coarses ONCE (affinity discarded — not needed for tuning)
    print(f"Pre-computing eigenvalues for {len(known_coarse_feats_list)} known coarses "
          f"(n_neighbors={n_neighbors})...")
    known_coarse_eigenvals = []
    for feats, k_true in known_coarse_feats_list:
        vals, _ = _compute_eigenvalues_for_coarse(feats, kmax=kmax, n_neighbors=n_neighbors)
        known_coarse_eigenvals.append((vals, k_true))
    print(f"  Done. Eigenvalue arrays computed.")

    # Tune gap threshold cheaply
    print(f"\nTuning spectral gap_threshold on {len(known_coarse_eigenvals)} known coarses "
          f"({n_grid} candidates in [{lo:.5f}, {hi:.3f}])...")
    best_g = tune_spectral_gap_threshold(known_coarse_eigenvals, gap_threshold_grid, kmax=kmax)

    # Apply to unknown coarses; also run spectral clustering at k_hat for direct eval
    print(f"\nApplying best gap_threshold={best_g:.5f} to {len(unknown_coarses)} unknown coarses...")
    k_per_coarse = {}
    direct_preds = {}
    for c in sorted(unknown_coarses):
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        c_feats = all_feats[c_mask].astype(np.float32)
        # Build affinity and eigenvalues together (A reused for direct clustering below)
        vals, A = _compute_eigenvalues_for_coarse(c_feats, kmax=kmax, n_neighbors=n_neighbors)
        k_hat = max(1, eigengap_with_threshold(vals, kmax=kmax, gap_threshold=best_g))
        print(f"  Coarse {c}: k_hat={k_hat}")
        k_per_coarse[str(c)] = k_hat

        # Run spectral clustering at k_hat using the same affinity (direct output)
        if A is not None and k_hat >= 2 and len(c_feats) >= k_hat:
            try:
                sc = SpectralClustering(
                    n_clusters=k_hat, affinity='precomputed',
                    assign_labels='cluster_qr', random_state=_RANDOM_STATE,
                )
                direct_preds[c] = sc.fit_predict(A.toarray()).astype(np.int64)
            except Exception as exc:
                print(f"  WARNING: direct spectral clustering failed for coarse {c}: {exc}")
                direct_preds[c] = None
        else:
            direct_preds[c] = np.zeros(len(c_feats), dtype=np.int64)

    return k_per_coarse, direct_preds


# ---------------------------------------------------------------------------
# Method A3: Xmeans PCA dim tuning
# ---------------------------------------------------------------------------

def _apply_pca_to_feats(feats: np.ndarray, pca_dim: int):
    """Apply PCA to feats if pca_dim > 0 and pca_dim < D, else return feats as-is."""
    if pca_dim > 0 and pca_dim < feats.shape[1] and len(feats) > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=_RANDOM_STATE)
        reduced = pca.fit_transform(feats)
        return normalize(reduced, norm='l2', axis=1).astype(np.float32)
    return feats.astype(np.float32)


def estimate_k_xmeans_hier(feats: np.ndarray, kmax: int, pca_dim: int = 0) -> int:
    """
    Run xmeans (pyclustering) with MNDL criterion (criterion=1). Return k_hat.

    Parameters
    ----------
    feats   : (N, D) float32 features (already L2-normed)
    kmax    : upper bound on K
    pca_dim : if > 0, apply PCA to pca_dim dims before clustering

    Returns
    -------
    k_hat : int

    Raises
    ------
    ImportError if pyclustering is not installed.
    """
    try:
        from pyclustering.cluster.xmeans import xmeans
        from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
    except ImportError as exc:
        raise ImportError(
            f"pyclustering is required for xmeans K estimation but is not installed: {exc}"
        ) from exc

    if len(feats) < 2:
        return 1

    X = _apply_pca_to_feats(feats, pca_dim)
    X_list = X.tolist()

    initial_centers = kmeans_plusplus_initializer(X_list, 1, random_state=_RANDOM_STATE).initialize()
    xm = xmeans(
        X_list,
        initial_centers=initial_centers,
        kmax=kmax,
        criterion=1,   # MNDL
        ccore=True,
    )
    xm.process()
    preds = np.array(xm.predict(X_list), dtype=np.int64)
    return int(len(np.unique(preds)))


def tune_xmeans_pca(
    known_coarse_feats_list: list,
    pca_dims: list,
    kmax: int,
) -> int:
    """
    Return the pca_dim that maximises mean labelled_k_score on known coarses.

    Parameters
    ----------
    known_coarse_feats_list : list of (feats [N,D] float32, k_true int)
    pca_dims                : list of int (0 = skip PCA)
    kmax                    : upper bound on K for xmeans

    Returns
    -------
    best_pca_dim : int
    """
    if not pca_dims:
        raise ValueError("tune_xmeans_pca: pca_dims is empty.")
    if not known_coarse_feats_list:
        raise ValueError("tune_xmeans_pca: known_coarse_feats_list is empty.")

    best_pca_dim = pca_dims[0]
    best_score = -1.0

    for pd in pca_dims:
        scores = []
        for feats, k_true in known_coarse_feats_list:
            try:
                k_hat = estimate_k_xmeans_hier(feats, kmax=kmax, pca_dim=pd)
                scores.append(float(labelled_k_score(k_hat, k_true)))
            except Exception as exc:
                print(f"  WARNING: xmeans failed for pca_dim={pd}: {exc}")
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        print(f"  xmeans pca_dim={pd}: mean_labelled_k_score={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_pca_dim = pd

    print(f"Best xmeans pca_dim={best_pca_dim} (score={best_score:.4f})")
    return best_pca_dim


def run_xmeans(
    all_feats: np.ndarray,
    all_labels: np.ndarray,
    label_to_coarse: dict,
    known_coarse_feats_list: list,
    unknown_coarses: set,
    args,
) -> tuple:
    """
    Full A3 pipeline: tune PCA dim on known coarses, apply xmeans to unknown coarses.

    Returns ({str(coarse_id): int(k_hat)}, {int(coarse_id): np.ndarray preds}).
    The second dict contains the actual xmeans cluster assignments (direct tuning output).
    """
    try:
        from pyclustering.cluster.xmeans import xmeans as _xmeans
        from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer as _kpp
    except ImportError as exc:
        raise ImportError(
            f"pyclustering is required for xmeans K estimation but is not installed: {exc}"
        ) from exc

    pca_dim_cands_str = getattr(args, 'pca_dim_candidates', '0,32,64,128')
    pca_dims = [int(x.strip()) for x in pca_dim_cands_str.split(',')]

    # kmax derived from max k_true across known coarses
    kmax = max((k_true for _, k_true in known_coarse_feats_list), default=10)
    print(f"\nXmeans: kmax={kmax} (from max known coarse k_true)")

    print(f"\nTuning xmeans pca_dim on {len(known_coarse_feats_list)} known coarses "
          f"(candidates: {pca_dims})...")
    best_pd = tune_xmeans_pca(known_coarse_feats_list, pca_dims, kmax=kmax)

    print(f"\nApplying best pca_dim={best_pd} to {len(unknown_coarses)} unknown coarses...")
    k_per_coarse = {}
    direct_preds = {}
    for c in sorted(unknown_coarses):
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        c_feats = all_feats[c_mask].astype(np.float32)
        try:
            X = _apply_pca_to_feats(c_feats, best_pd)
            X_list = X.tolist()
            initial_centers = _kpp(X_list, 1, random_state=_RANDOM_STATE).initialize()
            xm = _xmeans(X_list, initial_centers=initial_centers,
                         kmax=kmax, criterion=1, ccore=True)
            xm.process()
            preds = np.array(xm.predict(X_list), dtype=np.int64)
            k_hat = max(1, int(len(np.unique(preds))))
            direct_preds[c] = preds
        except Exception as exc:
            print(f"  WARNING: xmeans failed for coarse {c}: {exc}. Falling back to k=1.")
            k_hat = 1
            direct_preds[c] = None
        print(f"  Coarse {c}: k_hat={k_hat}")
        k_per_coarse[str(c)] = k_hat

    return k_per_coarse, direct_preds


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Tuning-based per-coarse K estimation for hierarchical GCD.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / features
    parser.add_argument('--dataset_name',      type=str, required=True)
    parser.add_argument('--model_name',        type=str, default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=1.0)
    parser.add_argument('--use_ssb_splits',    type=str2bool, default=True)
    parser.add_argument('--hier_split_type',   type=str, default=None)
    parser.add_argument('--num_novel_classes', type=int, default=None)
    parser.add_argument('--root_dir',          type=str, default=feature_extract_dir,
                        help='Root directory containing pre-extracted feature files')
    parser.add_argument('--batch_size',        type=int, default=256)
    parser.add_argument('--num_workers',       type=int, default=4)
    # Coarse labels
    parser.add_argument('--coarse_labels_path', type=str, required=True,
                        help='Path to JSON mapping fine_class_idx -> coarse_class_idx')
    parser.add_argument('--k_results_path',     type=str, required=True,
                        help='Output path for per-coarse K JSON {coarse_id: k}')
    # Method selection
    parser.add_argument('--estimate_k_method',  type=str, default='agglo',
                        choices=['agglo', 'spectral', 'xmeans'],
                        help='Tuning-based K estimation method')
    # Eval phase (clustering methods to evaluate with)
    parser.add_argument('--cluster_methods',    type=str,
                        default='kmeans,agglo,spectral,gmm_tied',
                        help='Comma-separated list of clustering methods for eval phase')
    # A1 (agglo) args
    parser.add_argument('--agglo_n_grid',   type=int,   default=25,
                        help='Number of threshold candidates in the grid')
    parser.add_argument('--agglo_lo',       type=float, default=0.0,
                        help='Lower bound for threshold grid')
    parser.add_argument('--agglo_hi',       type=float, default=2.0,
                        help='Upper bound for threshold grid')
    # A2 (spectral) args
    parser.add_argument('--spectral_gap_n_grid',  type=int,   default=30,
                        help='Number of gap_threshold candidates')
    parser.add_argument('--spectral_gap_lo',      type=float, default=0.001,
                        help='Lower bound for gap_threshold grid')
    parser.add_argument('--spectral_gap_hi',      type=float, default=1.0,
                        help='Upper bound for gap_threshold grid')
    parser.add_argument('--spectral_n_neighbors', type=int,   default=10,
                        help='Number of neighbors for kNN affinity graph')
    # A3 (xmeans) args
    parser.add_argument('--pca_dim_candidates',   type=str, default='0,32,64,128',
                        help='Comma-separated PCA dims to try (0 = skip PCA)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed propagated to KMeans / Agglo / Spectral '
                             '/ xmeans / PCA `random_state` (default 0 reproduces '
                             'historical behaviour).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    global _RANDOM_STATE
    _RANDOM_STATE = int(args.seed)
    print(f'Tuning K estimation random_state = {_RANDOM_STATE}')

    # Class splits
    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # Coarse map and coarse_of array
    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    # Load features
    feature_root = os.path.join(args.root_dir,
                                f'{args.model_name}_{args.dataset_name}')
    print(f'Loading features from {feature_root} ...')
    train_dataset, _, _, _ = get_datasets(
        args.dataset_name, train_transform=None, test_transform=None, args=args
    )
    all_feats, all_labels, all_mask = load_all_features(
        train_dataset, feature_root, args.batch_size, args.num_workers, device
    )
    # Ensure L2-normalised (load_all_features already normalises, but re-normalise for safety)
    all_feats = normalize(all_feats, norm='l2', axis=1).astype(np.float32)
    print(f'Loaded {len(all_feats)} feature vectors (dim={all_feats.shape[1]})')
    print(f'  Labeled: {all_mask.sum()} | Unlabeled: {(~all_mask).sum()}')

    # Build coarse structures
    (label_to_coarse, known_coarses, unknown_coarses,
     known_coarse_feats_list, oracle_k) = _build_coarse_structures(
        all_feats, all_labels, all_mask, coarse_of
    )
    print(f'Known coarses: {len(known_coarses)} | Unknown coarses: {len(unknown_coarses)}')
    if not unknown_coarses:
        raise RuntimeError(
            "No unknown coarses found. All fine classes appear to be in train_classes. "
            "Check --hier_split_type and the coarse labels JSON."
        )
    if not known_coarses:
        raise RuntimeError(
            "No known coarses found. Cannot tune hyperparameters without known coarses "
            "with labeled fine-class supervision."
        )

    # Dispatch to tuning method
    method = args.estimate_k_method
    print(f'\nRunning tuning-based K estimation: method={method}')

    if method == 'agglo':
        k_per_coarse, direct_preds = run_agglo(
            all_feats, all_labels, label_to_coarse,
            unknown_coarses, known_coarse_feats_list, args,
        )
    elif method == 'spectral':
        k_per_coarse, direct_preds = run_spectral(
            all_feats, all_labels, label_to_coarse,
            known_coarse_feats_list, unknown_coarses, args,
        )
    elif method == 'xmeans':
        k_per_coarse, direct_preds = run_xmeans(
            all_feats, all_labels, label_to_coarse,
            known_coarse_feats_list, unknown_coarses, args,
        )
    else:
        raise ValueError(
            f"Unknown estimate_k_method={method!r}. Expected one of: agglo, spectral, xmeans."
        )

    total_k_unknown = sum(k_per_coarse.values())
    print(f'\nPer-coarse K estimates: {k_per_coarse}')
    print(f'Total unknown clusters: {total_k_unknown}')

    # Save K results JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.k_results_path)), exist_ok=True)
    with open(args.k_results_path, 'w') as f:
        json.dump(k_per_coarse, f, indent=2)

    # Evaluation phase — re-cluster at k_hat using multiple methods
    cluster_methods_list = [m.strip() for m in args.cluster_methods.split(',')]

    print(f'\nRunning eval phase with clustering methods: {cluster_methods_list}')
    per_coarse_metrics = eval_per_coarse(
        all_feats=all_feats,
        all_labels=all_labels,
        coarse_of=coarse_of,
        k_per_coarse=k_per_coarse,
        oracle_k=oracle_k,
        methods=cluster_methods_list,
    )

    # Direct metrics — ACC/AMI from the clustering actually produced by the tuning step
    print('\nComputing direct tuning metrics...')
    coarse_to_fine_classes = {}
    for label_idx, coarse_id in enumerate(coarse_of.tolist()):
        coarse_to_fine_classes.setdefault(int(coarse_id), set()).add(label_idx)

    for c, preds in direct_preds.items():
        if c not in per_coarse_metrics:
            continue
        if preds is None:
            per_coarse_metrics[c]['tuning_direct_acc'] = float('nan')
            per_coarse_metrics[c]['tuning_direct_ami'] = float('nan')
            continue
        fine_classes = coarse_to_fine_classes.get(c, set())
        mask_c = np.array([int(l) in fine_classes for l in all_labels], dtype=bool)
        c_labels = all_labels[mask_c]
        if len(preds) != len(c_labels) or len(c_labels) < 2:
            per_coarse_metrics[c]['tuning_direct_acc'] = float('nan')
            per_coarse_metrics[c]['tuning_direct_ami'] = float('nan')
            continue
        try:
            acc = float(cluster_acc(c_labels.astype(int), preds.astype(int)))
            ami = float(adjusted_mutual_info_score(c_labels, preds))
        except Exception as exc:
            print(f"  WARNING: direct metric computation failed for coarse {c}: {exc}")
            acc, ami = float('nan'), float('nan')
        per_coarse_metrics[c]['tuning_direct_acc'] = acc
        per_coarse_metrics[c]['tuning_direct_ami'] = ami

    aggregate = aggregate_metrics(per_coarse_metrics, methods=cluster_methods_list)

    # Add direct metric aggregates manually
    def _nanmean(vals):
        vals = [v for v in vals if v == v and v is not None]
        return float(np.mean(vals)) if vals else float('nan')

    all_entries = list(per_coarse_metrics.values())
    multi_entries = [e for e in all_entries if e.get('k_true', 1) > 1]
    for split_name, entries in [('all', all_entries), ('multi_class', multi_entries)]:
        aggregate[split_name]['tuning_direct_acc'] = _nanmean(
            [e.get('tuning_direct_acc', float('nan')) for e in entries])
        aggregate[split_name]['tuning_direct_ami'] = _nanmean(
            [e.get('tuning_direct_ami', float('nan')) for e in entries])

    print('\nAggregate metrics:')
    print_aggregate_summary(aggregate, methods=cluster_methods_list)
    # Print direct stats
    for split_name in ('all', 'multi_class'):
        d = aggregate.get(split_name, {})
        tag = 'All' if split_name == 'all' else 'Multi-class'
        print(f"  [{tag}] tuning_direct: "
              f"acc={d.get('tuning_direct_acc', float('nan')):.3f}  "
              f"ami={d.get('tuning_direct_ami', float('nan')):.3f}")

    # Save metrics JSON
    metrics_path = args.k_results_path.replace('.json', '_metrics.json')
    save_hier_metrics(
        k_per_coarse=k_per_coarse,
        per_coarse=per_coarse_metrics,
        aggregate=aggregate,
        method_name=f'tuning_{method}',
        dataset_name=args.dataset_name,
        output_path=metrics_path,
    )

    print(f'\nK estimation complete: {args.k_results_path}')


if __name__ == '__main__':
    main()
