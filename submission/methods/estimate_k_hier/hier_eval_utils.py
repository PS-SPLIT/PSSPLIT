"""
hier_eval_utils.py — Shared utilities for hierarchical per-coarse K estimation evaluation.

Provides:
1. cluster_coarse()      — cluster features for a single coarse group
2. eval_per_coarse()     — run clustering per unknown coarse, compute ACC/AMI/K-accuracy
3. aggregate_metrics()   — compute means over all coarses and multi-class coarses (k>1)
4. save_hier_metrics()   — write comprehensive JSON results file

Used by estimate_k_hier_tuning.py, estimate_k_hier_brent.py, and
estimate_k_hier_transformer.py.
"""

import os
import json

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from methods.estimate_k.estimate_k_tuning import labelled_k_score
from project_utils.cluster_utils import cluster_acc


CLUSTER_METHODS = ['kmeans', 'agglo', 'spectral', 'gmm_tied']


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def cluster_coarse(feats: np.ndarray, k: int, method: str = 'kmeans',
                   n_neighbors: int = 10, random_state: int = 0) -> np.ndarray:
    """
    Cluster (N, D) features into k groups.

    Returns int64 label array (0..k-1).
    k <= 1      → all zeros (degenerate / single-cluster case).
    k > N       → cap k at N (cannot have more clusters than samples).

    Methods
    -------
    kmeans   : sklearn KMeans(n_init=10)
    agglo    : AgglomerativeClustering(linkage='ward')
    spectral : kNN connectivity graph (n_neighbors), assign_labels='cluster_qr'
    gmm_tied : GaussianMixture(covariance_type='tied', n_init=3)
    """
    N = len(feats)
    if N == 0:
        return np.zeros(0, dtype=np.int64)
    if k <= 1:
        return np.zeros(N, dtype=np.int64)
    k = min(k, N)

    if method == 'kmeans':
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        return model.fit_predict(feats).astype(np.int64)

    elif method == 'agglo':
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        return model.fit_predict(feats).astype(np.int64)

    elif method == 'spectral':
        n_nb = max(2, min(n_neighbors, N - 1))
        A = kneighbors_graph(feats, n_neighbors=n_nb,
                             mode='connectivity', include_self=False)
        A = 0.5 * (A + A.T)
        model = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            assign_labels='cluster_qr',
            random_state=random_state,
        )
        return model.fit_predict(A.toarray()).astype(np.int64)

    elif method == 'gmm_tied':
        model = GaussianMixture(
            n_components=k,
            covariance_type='tied',
            n_init=3,
            random_state=random_state,
        )
        model.fit(feats)
        return model.predict(feats).astype(np.int64)

    else:
        raise ValueError(
            f"Unknown cluster method {method!r}. "
            f"Expected one of: {CLUSTER_METHODS}."
        )


# ---------------------------------------------------------------------------
# Per-coarse evaluation
# ---------------------------------------------------------------------------

def eval_per_coarse(
    all_feats: np.ndarray,
    all_labels: np.ndarray,
    coarse_of: np.ndarray,
    k_per_coarse: dict,
    oracle_k: dict,
    methods: list = None,
    alt_feats: np.ndarray = None,
    alt_name: str = 'transformer',
) -> dict:
    """
    Run clustering per unknown coarse and compute ACC / AMI / K-accuracy.

    Parameters
    ----------
    all_feats   : (N, D) float32 — backbone features in original all_mask ordering
    all_labels  : (N,) int64    — remapped fine-class labels (0..num_classes-1)
    coarse_of   : (num_classes,) int64 — coarse_of[remapped_label] = coarse_id
    k_per_coarse: {str(coarse_id): int k_hat}
    oracle_k    : {int(coarse_id): int k_true}
    methods     : subset of CLUSTER_METHODS (None = all)
    alt_feats   : optional (N, D) second feature set (e.g., transformer features)
    alt_name    : prefix for alt_feats metric keys (default 'transformer')

    Returns
    -------
    per_coarse : dict keyed by int(coarse_id), each value is a dict:
        {
            'k_true': int, 'k_hat': int, 'k_score': float,
            'kmeans_acc': float, 'kmeans_ami': float,
            'agglo_acc':  float, 'agglo_ami':  float,
            'spectral_acc': float, 'spectral_ami': float,
            'gmm_tied_acc': float, 'gmm_tied_ami': float,
            # If alt_feats provided, also:
            '{alt_name}_kmeans_acc': float, ... etc.
        }

    Skips coarses where mask_c.sum() < 2.
    Per-(method, coarse) exceptions are caught, set to NaN, and warned.
    """
    if methods is None:
        methods = CLUSTER_METHODS

    # Build a map: coarse_id → set of fine class indices (0-based remapped labels)
    coarse_to_fine = {}
    for label_idx, coarse_id in enumerate(coarse_of.tolist()):
        coarse_to_fine.setdefault(int(coarse_id), set()).add(label_idx)

    per_coarse = {}

    for coarse_str, k_hat in k_per_coarse.items():
        c = int(coarse_str)
        k_hat = int(k_hat)
        k_true = int(oracle_k.get(c, k_hat))
        k_score = float(labelled_k_score(k_hat, k_true))

        fine_classes = coarse_to_fine.get(c, set())
        # mask: samples whose label belongs to this coarse
        mask_c = np.array([int(l) in fine_classes for l in all_labels], dtype=bool)
        if mask_c.sum() < 2:
            continue

        c_feats = all_feats[mask_c].astype(np.float32)
        c_labels = all_labels[mask_c]

        entry = {'k_true': k_true, 'k_hat': k_hat, 'k_score': k_score}

        for method in methods:
            for feat_set, prefix in [(c_feats, ''), (alt_feats[mask_c] if alt_feats is not None else None, alt_name + '_')]:
                if feat_set is None:
                    continue
                try:
                    preds = cluster_coarse(feat_set, k_hat, method=method)
                    acc = float(cluster_acc(c_labels.astype(int), preds.astype(int)))
                    ami = float(adjusted_mutual_info_score(c_labels, preds))
                except Exception as exc:
                    print(f"  WARNING: {prefix}{method} clustering failed for coarse {c}: {exc}")
                    acc = float('nan')
                    ami = float('nan')
                entry[f'{prefix}{method}_acc'] = acc
                entry[f'{prefix}{method}_ami'] = ami

        per_coarse[c] = entry

    return per_coarse


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(
    per_coarse: dict,
    methods: list = None,
    alt_name: str = None,
) -> dict:
    """
    Compute mean metrics over:
    - 'all'         : all coarses in per_coarse
    - 'multi_class' : coarses with k_true > 1 only

    NaN values are excluded from means.  Returns NaN for empty sets.

    Parameters
    ----------
    per_coarse : output of eval_per_coarse()
    methods    : subset of CLUSTER_METHODS (None = all)
    alt_name   : if set, also aggregate '{alt_name}_{method}_acc/ami' columns
    """
    if methods is None:
        methods = CLUSTER_METHODS

    def _nanmean(vals):
        vals = [v for v in vals if v == v and v is not None]  # exclude NaN/None
        return float(np.mean(vals)) if vals else float('nan')

    all_entries = list(per_coarse.values())
    multi_entries = [e for e in all_entries if e.get('k_true', 1) > 1]

    result = {}
    for split_name, entries in [('all', all_entries), ('multi_class', multi_entries)]:
        d = {
            'n_coarses': len(entries),
            'k_accuracy': _nanmean([e.get('k_score', float('nan')) for e in entries]),
        }
        for method in methods:
            d[f'{method}_acc'] = _nanmean([e.get(f'{method}_acc', float('nan')) for e in entries])
            d[f'{method}_ami'] = _nanmean([e.get(f'{method}_ami', float('nan')) for e in entries])
            if alt_name:
                d[f'{alt_name}_{method}_acc'] = _nanmean(
                    [e.get(f'{alt_name}_{method}_acc', float('nan')) for e in entries])
                d[f'{alt_name}_{method}_ami'] = _nanmean(
                    [e.get(f'{alt_name}_{method}_ami', float('nan')) for e in entries])
        result[split_name] = d

    return result


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def compute_precision_curve(
    confidence_per_coarse: dict,
    oracle_k: dict,
    n_thresholds: int = 21,
) -> dict:
    """
    Compute a precision-recall curve for predicting 'this coarse has k > 1'.

    Parameters
    ----------
    confidence_per_coarse : {str or int coarse_id: float in [0, 1]}
        Confidence score that a coarse has multiple fine classes.
        For the transformer: p_split.
        For brent: normalised score margin (best_kgt1_score - score_k1).
    oracle_k : {int coarse_id: int k_true}
    n_thresholds : number of evenly-spaced threshold points in [0, 1]

    Returns
    -------
    dict with keys:
        thresholds         : list of floats
        precision          : precision at each threshold (None when n_predictions=0)
        recall             : recall at each threshold
        n_predictions      : number of coarses with confidence >= threshold
        n_true_positives   : number of truly-multiclass among those
        n_total_positives  : total number of k_true > 1 coarses
        average_precision  : area under PR curve (trapezoidal; None if not computable)
        topk_precision     : {str(k): precision} for k in [1,2,3,5,10,20,'all']
        ranked_predictions : list of {coarse_id, confidence, k_hat_is_split, k_true, correct}
                             sorted by confidence descending (all coarses)
    """
    records = []
    for c_str, conf in confidence_per_coarse.items():
        c = int(c_str)
        k_true = oracle_k.get(c, 1)
        records.append({
            "coarse_id": str(c_str),
            "confidence": float(conf),
            "k_true": int(k_true),
            "is_positive": bool(k_true > 1),
        })

    # Sort by confidence descending for ranked summaries
    records_sorted = sorted(records, key=lambda r: -r["confidence"])
    n_total_positives = sum(1 for r in records if r["is_positive"])

    thresholds = list(np.linspace(0.0, 1.0, n_thresholds))
    precision_vals, recall_vals, n_pred_vals, n_tp_vals = [], [], [], []

    for t in thresholds:
        predicted = [r for r in records if r["confidence"] >= t]
        n_pred = len(predicted)
        n_tp = sum(1 for r in predicted if r["is_positive"])
        prec = n_tp / n_pred if n_pred > 0 else None
        rec = n_tp / n_total_positives if n_total_positives > 0 else None
        precision_vals.append(prec)
        recall_vals.append(rec)
        n_pred_vals.append(n_pred)
        n_tp_vals.append(n_tp)

    # Average precision (trapezoidal under the non-None points of PR curve)
    valid_pairs = [(r, p) for r, p in zip(recall_vals, precision_vals)
                   if r is not None and p is not None]
    if len(valid_pairs) >= 2:
        valid_sorted = sorted(dict(valid_pairs).items())  # deduplicate by recall
        rs = [x[0] for x in valid_sorted]
        ps = [x[1] for x in valid_sorted]
        ap = float(np.trapz(ps, rs)) if rs[-1] > rs[0] else None
    else:
        ap = None

    # Top-K precision (ranked by confidence, including all coarses)
    topk_precision = {}
    for k in [1, 2, 3, 5, 10, 20, len(records)]:
        top = records_sorted[:k]
        n_tp_k = sum(1 for r in top if r["is_positive"])
        topk_precision[str(k)] = round(n_tp_k / len(top), 4) if top else None

    # Ranked list for saving
    ranked = [
        {
            "rank": i + 1,
            "coarse_id": r["coarse_id"],
            "confidence": round(r["confidence"], 4),
            "k_true": r["k_true"],
            "correct": r["is_positive"],
        }
        for i, r in enumerate(records_sorted)
    ]

    return {
        "thresholds": [round(t, 4) for t in thresholds],
        "precision": precision_vals,
        "recall": recall_vals,
        "n_predictions": n_pred_vals,
        "n_true_positives": n_tp_vals,
        "n_total_positives": int(n_total_positives),
        "average_precision": round(ap, 4) if ap is not None else None,
        "topk_precision": topk_precision,
        "ranked_predictions": ranked,
    }


def print_precision_summary(precision_curve: dict, method_name: str = '') -> None:
    """Print a human-readable top-K precision summary."""
    header = f"Split-prediction confidence ranking ({method_name})" if method_name else \
             "Split-prediction confidence ranking"
    print(f"\n  {header}")
    ranked = precision_curve.get("ranked_predictions", [])
    n_total_pos = precision_curve.get("n_total_positives", 0)
    n_total = len(ranked)
    print(f"  (total coarses: {n_total} | truly multi-class: {n_total_pos})")
    print(f"  {'Rank':<6} {'Coarse':<8} {'Conf':>6}  {'k_true':>6}  {'Correct':>7}")
    for r in ranked[:10]:
        tick = '✓' if r['correct'] else '✗'
        print(f"  {r['rank']:<6} {r['coarse_id']:<8} {r['confidence']:>6.3f}  "
              f"{r['k_true']:>6}  {tick:>7}")
    if n_total > 10:
        print(f"  ... ({n_total - 10} more)")
    print(f"  Top-K precision:")
    for k_str, prec in precision_curve.get("topk_precision", {}).items():
        k_int = int(k_str)
        top = ranked[:k_int]
        n_tp = sum(1 for r in top if r["correct"])
        prec_str = f"{prec:.3f}" if prec is not None else "—"
        print(f"    Top-{k_str:<5}: {n_tp}/{len(top)}  = {prec_str}")
    ap = precision_curve.get("average_precision")
    if ap is not None:
        print(f"  Average precision (PR-AUC): {ap:.4f}")


def save_hier_metrics(
    k_per_coarse: dict,
    per_coarse: dict,
    aggregate: dict,
    method_name: str,
    dataset_name: str,
    output_path: str,
    extra_data: dict = None,
) -> None:
    """
    Save comprehensive per-coarse K estimation and clustering metrics to JSON.

    Parameters
    ----------
    k_per_coarse : {str(coarse_id): int k_hat}
    per_coarse   : {int(coarse_id): {...}} — output of eval_per_coarse()
    aggregate    : output of aggregate_metrics()
    method_name  : e.g. 'tuning_agglo', 'brent_kmeans', 'transformer'
    dataset_name : e.g. 'aves_small'
    output_path  : full path to write JSON

    Creates parent directories if they do not exist.
    Prints 'Metrics saved to {output_path}'.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert int keys in per_coarse to strings for JSON serialisation
    per_coarse_serialisable = {}
    for coarse_id, entry in per_coarse.items():
        ser = {}
        for k, v in entry.items():
            if isinstance(v, float) and v != v:  # NaN
                ser[k] = None
            elif isinstance(v, (np.integer,)):
                ser[k] = int(v)
            elif isinstance(v, (np.floating,)):
                ser[k] = float(v)
            else:
                ser[k] = v
        per_coarse_serialisable[str(coarse_id)] = ser

    results = {
        'method': method_name,
        'dataset': dataset_name,
        'k_per_coarse': {str(k): int(v) for k, v in k_per_coarse.items()},
        'per_coarse': per_coarse_serialisable,
        'aggregate': aggregate,
    }
    if extra_data:
        results.update(extra_data)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Metrics saved to {output_path}")


# ---------------------------------------------------------------------------
# Convenience: print aggregate summary to stdout
# ---------------------------------------------------------------------------

def print_aggregate_summary(aggregate: dict, methods: list = None, alt_name: str = None) -> None:
    """Print a concise summary table of aggregate metrics."""
    if methods is None:
        methods = CLUSTER_METHODS
    for split_name in ('all', 'multi_class'):
        d = aggregate.get(split_name, {})
        n = d.get('n_coarses', 0)
        k_acc = d.get('k_accuracy', float('nan'))
        tag = 'All coarses' if split_name == 'all' else f'Multi-class (k>1)'
        print(f"  [{tag} | n={n}] k_acc={k_acc:.3f}")
        for method in methods:
            acc = d.get(f'{method}_acc', float('nan'))
            ami = d.get(f'{method}_ami', float('nan'))
            print(f"    {method:<12}: acc={acc:.3f}  ami={ami:.3f}")
            if alt_name and f'{alt_name}_{method}_acc' in d:
                acc_alt = d.get(f'{alt_name}_{method}_acc', float('nan'))
                ami_alt = d.get(f'{alt_name}_{method}_ami', float('nan'))
                print(f"    {alt_name}_{method}: acc={acc_alt:.3f}  ami={ami_alt:.3f}")
