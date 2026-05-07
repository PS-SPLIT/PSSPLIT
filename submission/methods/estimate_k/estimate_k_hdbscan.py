"""
estimate_k_hdbscan.py — HDBSCAN-based K estimation with Optuna tuning.
Uses sklearn.cluster.HDBSCAN (requires scikit-learn >= 1.3).

Algorithm:
    1. Load labelled and unlabelled training features from disk.
    2. Optionally apply PCA (controlled by args.pca_dim).
    3. Use Optuna (TPE sampler) to tune HDBSCAN hyperparameters by maximising
       labelled_k_score on the labelled features — the same criterion used by
       agglo/spectral labelled_fn.
       The labelled set is subsampled to hdbscan_max_labelled_samples (default
       2000) for speed; the full labelled set is used in the final run.
    4. Run HDBSCAN with best params on all features to estimate total K.
       Noise points (label -1) are remapped to a fresh cluster ID so that no
       point is discarded.
    5. k_hat is floored at num_labeled_classes + 1 — if HDBSCAN estimates fewer
       total clusters than there are known classes, the estimate is clearly wrong.

Fixed hyperparameter search space (not user-configurable):
    min_cluster_size          : [5, 10, 15, 20, 25, 30, 40, 50, 60]
    min_samples               : [None, 1, 5, 10, 15, 20, 30, 40, 50, 60]
    cluster_selection_epsilon : [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    cluster_selection_method  : ['eom', 'leaf']
    alpha                     : [1.0, 1.2]
    allow_single_cluster      : [True, False]

Number of Optuna trials is controlled by args.hdbscan_n_trials (default 120).
Labelled subsample cap is args.hdbscan_max_labelled_samples (default 2000).
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.cluster import HDBSCAN

from methods.estimate_k.estimate_k_agglo import (
    collect_and_split_features,
    apply_pca_and_norm,
)
from methods.estimate_k.estimate_k_tuning import labelled_k_score


# ---------------------------------------------------------------------------
# Fixed search space
# ---------------------------------------------------------------------------

_HDBSCAN_GRID = dict(
    min_cluster_size=[5, 10, 15, 20, 25, 30, 40, 50, 60],
    min_samples=[None, 1, 5, 10, 15, 20, 30, 40, 50, 60],
    cluster_selection_epsilon=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9],
    cluster_selection_method=["eom", "leaf"],
    alpha=[1.0, 1.2],
    allow_single_cluster=[True, False],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remap_noise(labels: np.ndarray) -> np.ndarray:
    """
    Replace all -1 (noise) labels with a single new cluster ID (max_label + 1).
    If there is no noise the array is returned unchanged.
    Always returns a copy.
    """
    labels = labels.copy()
    noise_mask = labels == -1
    if noise_mask.any():
        new_id = int(labels[~noise_mask].max()) + 1 if (~noise_mask).any() else 0
        labels[noise_mask] = new_id
    return labels


def _run_hdbscan(X: np.ndarray, mcs: int, ms, eps: float,
                 csm: str, alpha: float, asc: bool) -> np.ndarray:
    """Fit HDBSCAN and return integer labels (noise NOT remapped)."""
    model = HDBSCAN(
        min_cluster_size=int(mcs),
        min_samples=None if ms is None else int(ms),
        cluster_selection_epsilon=float(eps),
        cluster_selection_method=str(csm),
        alpha=float(alpha),
        allow_single_cluster=bool(asc),
    )
    return model.fit_predict(X)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_k_hdbscan(train_loader, args, device=None, test_loader=None):
    """
    Estimate K using HDBSCAN with Optuna hyperparameter tuning.

    Parameters
    ----------
    train_loader : DataLoader  — merged (labelled + unlabelled) training loader
    args         : Namespace   — must have: num_labeled_classes, pca_dim,
                                 hdbscan_n_trials (default 120),
                                 hdbscan_max_labelled_samples (default 2000)
    device       : torch.device or None
    test_loader  : DataLoader or None  — unused; accepted for API consistency

    Returns
    -------
    estimated_k_total : int  (>= num_labeled_classes + 1)
    estimated_C_u     : int
    result            : dict with keys all_feats, all_labels, mask_lab,
                                       preds, pca, fitted_model
    """
    import torch
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "estimate_k_hdbscan requires optuna. "
            "Install with: pip install optuna"
        ) from e

    pca_dim          = int(getattr(args, 'pca_dim', 64))
    n_trials         = int(getattr(args, 'hdbscan_n_trials', 120))
    max_lab_samples  = int(getattr(args, 'hdbscan_max_labelled_samples', 2000))
    k_true           = int(args.num_labeled_classes)

    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    print("HDBSCAN k-estimation: collecting features...")
    l_feats_raw, l_targets, u_feats_raw, u_targets, all_mask = \
        collect_and_split_features(train_loader, device)
    l_feats_raw = l_feats_raw.astype(np.float32)
    u_feats_raw = u_feats_raw.astype(np.float32)
    print(f"  labelled={len(l_feats_raw)}, unlabelled={len(u_feats_raw)}, "
          f"dim={l_feats_raw.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Apply PCA (fit on labelled, transform both)
    # ------------------------------------------------------------------
    Xp_l, Xp_u, pca = apply_pca_and_norm(l_feats_raw, u_feats_raw, pca_dim)

    # ------------------------------------------------------------------
    # 3. Subsample labelled features for fast Optuna tuning
    # ------------------------------------------------------------------
    n_l = len(Xp_l)
    if n_l > max_lab_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_l, size=max_lab_samples, replace=False)
        Xp_l_tune = Xp_l[idx]
        print(f"  Subsampled labelled set for tuning: {n_l} → {max_lab_samples}")
    else:
        Xp_l_tune = Xp_l

    # ------------------------------------------------------------------
    # 4. Optuna: tune HDBSCAN params on (subsampled) labelled features
    # ------------------------------------------------------------------
    print(f"  Tuning HDBSCAN with Optuna ({n_trials} trials) on labelled features...")
    print(f"  Criterion: labelled_k_score (target k={k_true})")

    def objective(trial):
        mcs = trial.suggest_categorical("min_cluster_size",
                                        _HDBSCAN_GRID["min_cluster_size"])
        ms  = trial.suggest_categorical("min_samples",
                                        _HDBSCAN_GRID["min_samples"])
        eps = trial.suggest_categorical("cluster_selection_epsilon",
                                        _HDBSCAN_GRID["cluster_selection_epsilon"])
        csm = trial.suggest_categorical("cluster_selection_method",
                                        _HDBSCAN_GRID["cluster_selection_method"])
        a   = trial.suggest_categorical("alpha",
                                        _HDBSCAN_GRID["alpha"])
        asc = trial.suggest_categorical("allow_single_cluster",
                                        _HDBSCAN_GRID["allow_single_cluster"])
        try:
            labels = _run_hdbscan(Xp_l_tune, mcs, ms, eps, csm, a, asc)
            labels = _remap_noise(labels)
            k_hat_l = len(np.unique(labels))
            return labelled_k_score(k_hat_l, k_true)
        except Exception:
            return 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*force_all_finite.*")
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*A single label.*")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score  = study.best_value
    print(f"  Best Optuna score: {best_score:.4f}  params: {best_params}")

    # ------------------------------------------------------------------
    # 5. Run HDBSCAN with best params on ALL features
    # ------------------------------------------------------------------
    X_all = np.vstack([Xp_l, Xp_u])  # [labelled | unlabelled] ordering

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*force_all_finite.*")
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*A single label.*")
        labels_all = _run_hdbscan(
            X_all,
            mcs=best_params["min_cluster_size"],
            ms=best_params["min_samples"],
            eps=best_params["cluster_selection_epsilon"],
            csm=best_params["cluster_selection_method"],
            alpha=best_params["alpha"],
            asc=best_params["allow_single_cluster"],
        )

    n_noise = int((labels_all == -1).sum())
    labels_all = _remap_noise(labels_all)
    k_hat_raw = len(np.unique(labels_all))

    # Floor: HDBSCAN cannot produce fewer total clusters than known classes + 1
    k_hat = max(k_hat_raw, k_true + 1)
    if k_hat != k_hat_raw:
        print(f"  WARNING: HDBSCAN estimated k={k_hat_raw} < num_labeled+1={k_true+1}; "
              f"flooring to {k_hat}")

    k_novel = k_hat - k_true

    print(f"  noise points remapped to own cluster: {n_noise}")
    print(f"  estimated_k_total={k_hat}, estimated_C_u={k_novel}")
    print(f"Optimal K is {k_hat}")

    # ------------------------------------------------------------------
    # 6. Build result dict in original loader ordering
    #    X_all is shaped (N_l+N_u, D): first N_l rows = labelled.
    #    Scatter back into all_mask ordering so run_evaluation can use
    #    mask_lab to index correctly.
    # ------------------------------------------------------------------
    n_l      = int(all_mask.sum())
    feat_dim = X_all.shape[1]

    Xp_full = np.empty((len(all_mask), feat_dim), dtype=np.float32)
    Xp_full[all_mask]  = X_all[:n_l]
    Xp_full[~all_mask] = X_all[n_l:]

    all_labels_full = np.empty(len(all_mask), dtype=np.int64)
    all_labels_full[all_mask]  = l_targets.astype(np.int64)
    all_labels_full[~all_mask] = u_targets.astype(np.int64)

    preds_full = np.empty(len(all_mask), dtype=np.int64)
    preds_full[all_mask]  = labels_all[:n_l].astype(np.int64)
    preds_full[~all_mask] = labels_all[n_l:].astype(np.int64)

    result = dict(
        all_feats=Xp_full,
        all_labels=all_labels_full,
        mask_lab=all_mask,
        preds=preds_full,
        pca=pca,
        fitted_model=None,  # HDBSCAN has no sklearn-compatible predict for new data
    )
    return k_hat, k_novel, result
