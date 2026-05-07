"""
Semi-Supervised Spectral Clustering
=====================================
Builds a kNN affinity graph on all (labeled + unlabeled) features, computes
the k smallest eigenvectors of the normalized Laplacian to get a spectral
embedding, then runs SemiSupKMeans.fit_mix in that embedded space.

The supervision signal anchors the first ``n_labeled_classes`` cluster centers
to the labeled-class means in eigenvector space (exactly as fit_mix_once does
for standard semi-supervised k-means), while novel cluster centers and
unlabeled assignments are optimized by EM.

Ordering convention (matches run_pipeline.py evaluate_features):
  all_feats = np.vstack([l_feats, u_feats])   → [labeled | unlabeled]
  all_preds returned in the same [labeled | unlabeled] order
  centers returned in original feature space (for nearest-centroid test inference)
"""

import numpy as np
import torch
from typing import Tuple

from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from sklearn.neighbors import kneighbors_graph

from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans


def semi_sup_spectral_fit(
    l_feats: np.ndarray,
    l_targets: np.ndarray,
    u_feats: np.ndarray,
    n_clusters: int,
    n_neighbors: int = 20,
    metric: str = 'cosine',
    n_init: int = 10,
    device=None,
    cluster_size=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit semi-supervised spectral clustering and return cluster assignments.

    Parameters
    ----------
    l_feats : (N_l, D) float32 — labeled features (L2-normalised)
    l_targets : (N_l,) int — raw class indices for labeled samples
    u_feats : (N_u, D) float32 — unlabeled features (L2-normalised)
    n_clusters : int — total number of clusters K (labeled + novel)
    n_neighbors : int — number of neighbors for kNN affinity graph
    metric : str — distance metric for kNN graph ('cosine' or 'euclidean')
    n_init : int — number of SemiSupKMeans restarts in eigenvector space
    device : torch.device — device for SemiSupKMeans (defaults to cuda:0 or cpu)

    Returns
    -------
    all_preds : (N_l + N_u,) int — cluster assignments in [labeled | unlabeled] order
    centers : (K, D) float32 — cluster means in **original** feature space
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_l = len(l_targets)
    n_u = len(u_feats)
    N = n_l + n_u

    # ------------------------------------------------------------------
    # Step 1: Build kNN affinity graph on all features
    # all_feats is in [labeled | unlabeled] order
    # ------------------------------------------------------------------
    all_feats = np.vstack([l_feats, u_feats])

    n_nb = max(2, min(n_neighbors, N - 1))
    print(f'  [semi_spectral] Building kNN affinity graph '
          f'(N={N}, n_neighbors={n_nb}, metric={metric})...')

    A = kneighbors_graph(all_feats, n_neighbors=n_nb, metric=metric,
                         mode='connectivity', include_self=False)
    A = 0.5 * (A + A.T)   # symmetrize to undirected graph

    # ------------------------------------------------------------------
    # Step 2: Compute normalized Laplacian eigenvectors
    # Request n_clusters+1 smallest eigenvalues; skip the first (trivial
    # zero eigenvector of a connected graph) and keep the next n_clusters.
    # ------------------------------------------------------------------
    print(f'  [semi_spectral] Computing {n_clusters} eigenvectors...')
    L = csgraph_laplacian(A, normed=True)

    k_eigs = min(n_clusters + 1, N - 1)
    # Use shift-invert (sigma≈0) with 'LM' for the smallest eigenvalues of the
    # normalized Laplacian — much more numerically stable than which='SM' alone,
    # especially for large K.  Fixed v0 makes the result deterministic.
    v0 = np.ones(N) / np.sqrt(N)
    try:
        vals, vecs = eigsh(L, k=k_eigs, sigma=1e-7, which='LM',
                           tol=1e-6, maxiter=20 * N, v0=v0)
    except ArpackNoConvergence as exc:
        n_conv = len(exc.eigenvalues)
        raise RuntimeError(
            f"[semi_spectral] eigsh failed to converge: only {n_conv}/{k_eigs} "
            f"eigenvalues converged. Try reducing n_neighbors or n_clusters, "
            f"or check that the affinity graph is connected."
        ) from exc

    # Sort by eigenvalue ascending and drop the first (smallest ≈ 0)
    order = np.argsort(vals)
    vecs = vecs[:, order[1:n_clusters + 1]]   # (N, n_clusters)

    if vecs.shape[1] < n_clusters:
        raise RuntimeError(
            f"[semi_spectral] Only {vecs.shape[1]} eigenvectors available "
            f"(requested {n_clusters}). Reduce n_clusters or check graph connectivity."
        )

    # ------------------------------------------------------------------
    # Step 3: Row-normalize the spectral embedding (standard practice)
    # ------------------------------------------------------------------
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    vecs = vecs.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4: Split into labeled / unlabeled embeddings
    # Rows 0..N_l-1 = labeled (matches vstack ordering above)
    # ------------------------------------------------------------------
    l_eig = vecs[:n_l]   # (N_l, n_clusters)
    u_eig = vecs[n_l:]   # (N_u, n_clusters)

    # ------------------------------------------------------------------
    # Step 5: Run SemiSupKMeans in eigenvector space
    # fit_mix(u_feats, l_feats, l_targets) — unlabeled first in signature,
    # but internally concatenates [l | u] and returns labels_ in [l | u] order.
    # ------------------------------------------------------------------
    print(f'  [semi_spectral] Running SemiSupKMeans in eigenvector space '
          f'(K={n_clusters}, n_init={n_init})...')

    kmeans = SemiSupKMeans(
        k=n_clusters, tolerance=1e-4, max_iterations=10,
        init='k-means++', n_init=n_init, random_state=None,
        n_jobs=None, pairwise_batch_size=1024, mode=None,
        protos=None, cluster_size=cluster_size, is_code=False,
    )
    l_eig_t = torch.from_numpy(l_eig).to(device)
    u_eig_t = torch.from_numpy(u_eig).to(device)
    lt_t    = torch.from_numpy(l_targets.astype(np.int64)).to(device)

    kmeans.fit_mix(u_eig_t, l_eig_t, lt_t)
    all_preds = kmeans.labels_.cpu().numpy()   # (N_l + N_u,) in [l | u] order

    # ------------------------------------------------------------------
    # Step 6: Compute cluster centers in original feature space
    # (used for nearest-centroid test inference in evaluate_features)
    # ------------------------------------------------------------------
    centers = np.array([
        all_feats[all_preds == i].mean(axis=0) if (all_preds == i).any()
        else np.zeros(all_feats.shape[1], dtype=np.float32)
        for i in range(n_clusters)
    ])

    return all_preds, centers
