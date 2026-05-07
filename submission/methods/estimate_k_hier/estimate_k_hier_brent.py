"""
estimate_k_hier_brent.py — Probe-based grid-search per-coarse K estimation.

For each unknown coarse, selects ~4 relevant known coarses as a "probe" pool.
Combines probe features with the unknown coarse features, then grid-searches over
k_unknown (1..max_k_unknown), scoring by clustering accuracy on the probe portion.
The k_unknown with best probe accuracy is the estimate.

Taxonomy metadata (for aves_small / aves_full) is used to select taxonomically
similar known coarses as probes (via ancestor overlap and alignment_id matching).
For other datasets, probe coarses are chosen randomly.

Output format:
  - JSON: {str(coarse_id): k_i}  (same as all other estimate_k_hier methods)
  - Metrics JSON: k_results_path.replace('.json', '_metrics.json')
  - Prints "K estimation complete: {k_results_path}" (parsed by run_pipeline_hier.py)

Run standalone:
    python -m methods.estimate_k_hier.estimate_k_hier_brent \\
        --dataset_name aves_small \\
        --coarse_labels_path data/coarse_labels/aves_small.json \\
        --k_results_path /tmp/aves_small_k_brent.json \\
        --cluster_method kmeans
"""

import argparse
import json
import os

import numpy as np
from sklearn.preprocessing import normalize

from config import feature_extract_dir
from data.data_utils import build_coarse_of, load_coarse_map, oracle_k_per_coarse
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from methods.estimate_k_hier.estimate_k_hier import load_all_features
from methods.estimate_k_hier.hier_eval_utils import (
    CLUSTER_METHODS,
    aggregate_metrics,
    cluster_coarse,
    compute_precision_curve,
    eval_per_coarse,
    print_aggregate_summary,
    print_precision_summary,
    save_hier_metrics,
)
from project_utils.cluster_utils import cluster_acc
from project_utils.general_utils import str2bool


# ---------------------------------------------------------------------------
# Taxonomy metadata loading
# ---------------------------------------------------------------------------

def load_taxonomy_metadata(dataset_name: str, args) -> tuple:
    """
    Returns (coarse_to_ancestors, coarse_to_alignment, coarse_to_should_split)
    or (None, None, None) for datasets without taxonomy metadata.

    Only non-None for: aves_small, aves_full.

    coarse_to_ancestors   : {coarse_idx: set of str ancestor IDs}
    coarse_to_alignment   : {coarse_idx: int alignment_id}
    coarse_to_should_split: {coarse_idx: bool}

    The coarse_idx is built identically to how aves_small.json / aves_full.json
    was built: sorted(unique split_ids) → enumerate → split_to_coarse_idx.
    """
    import pandas as pd

    csv_map = {}
    try:
        from config import aves_small_splits_csv
        csv_map['aves_small'] = aves_small_splits_csv
    except ImportError:
        pass
    try:
        from config import aves_full_splits_csv
        csv_map['aves_full'] = aves_full_splits_csv
    except ImportError:
        pass

    csv_path = csv_map.get(dataset_name)
    if csv_path is None:
        return None, None, None

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Taxonomy CSV not found at {csv_path!r}. "
            "Check config.py paths."
        )

    df = pd.read_csv(csv_path)

    # Build the same split_id → coarse_idx mapping used when generating the JSON.
    all_split_ids = sorted(df['split_id'].unique())
    split_to_coarse_idx = {int(sid): i for i, sid in enumerate(all_split_ids)}

    coarse_to_ancestors: dict = {}
    coarse_to_alignment: dict = {}
    coarse_to_should_split: dict = {}

    for _, row in df.iterrows():
        c_idx = split_to_coarse_idx[int(row['split_id'])]
        # ancestor_ids is stored as a comma-separated string of taxon IDs
        anc_raw = str(row.get('ancestor_ids', ''))
        coarse_to_ancestors[c_idx] = set(anc_raw.split(',')) if anc_raw.strip() else set()
        coarse_to_alignment[c_idx] = int(row['alignment_id'])
        coarse_to_should_split[c_idx] = bool(row.get('should_split', True))

    return coarse_to_ancestors, coarse_to_alignment, coarse_to_should_split


# ---------------------------------------------------------------------------
# Probe coarse selection
# ---------------------------------------------------------------------------

def select_probe_coarses(
    unknown_c: int,
    known_coarses: set,
    coarse_to_ancestors,      # may be None
    coarse_to_alignment,      # may be None
    coarse_to_should_split,   # may be None
    n_related: int = 2,
    n_total: int = 4,
    rng: np.random.Generator = None,
) -> set:
    """
    Select up to n_total known probe coarses for unknown coarse c.

    For aves datasets (coarse_to_ancestors is not None):
      1. Aligned coarses: known coarses sharing alignment_id with c.
         (These are the "control" species for the same alignment event.)
      2. Taxonomically similar non-aligned known coarses with should_split=False:
         score = |ancestors_c ∩ ancestors_k|; take top n_related.
      Together these form the probe set (may be < n_total if not enough candidates).

    For other datasets:
      Random n_total from known_coarses (seeded via rng).

    Returns set of coarse indices (size ≤ n_total).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if coarse_to_ancestors is None:
        # Non-aves: random selection
        known_list = sorted(known_coarses)
        n_pick = min(n_total, len(known_list))
        chosen = rng.choice(known_list, size=n_pick, replace=False)
        return set(int(c) for c in chosen)

    # Aves path: use taxonomy
    c_alignment = coarse_to_alignment.get(unknown_c, -1)
    c_ancestors = coarse_to_ancestors.get(unknown_c, set())

    # Step 1: aligned known coarses (same alignment_id)
    aligned = set()
    for k in known_coarses:
        if coarse_to_alignment.get(k, -2) == c_alignment:
            aligned.add(k)

    # Step 2: taxonomically related but non-aligned known coarses with should_split=False
    #         (should_split=False means the control species is not genuinely split;
    #          these make better probes for k_unknown=1 baseline)
    candidates = []
    for k in known_coarses:
        if k in aligned:
            continue
        if coarse_to_should_split.get(k, True):
            continue  # prefer should_split=False candidates
        k_ancestors = coarse_to_ancestors.get(k, set())
        score = len(c_ancestors & k_ancestors)
        candidates.append((score, k))

    # Sort descending by overlap score; break ties by coarse index (deterministic)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    related = set(k for _, k in candidates[:n_related])

    probe_set = aligned | related

    # If we still need more, fill randomly from remaining known coarses
    remaining = sorted(known_coarses - probe_set)
    n_still_needed = max(0, n_total - len(probe_set))
    if n_still_needed > 0 and remaining:
        n_pick = min(n_still_needed, len(remaining))
        extra = rng.choice(remaining, size=n_pick, replace=False)
        probe_set.update(int(c) for c in extra)

    return probe_set


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search_k(
    probe_feats: np.ndarray,
    probe_labels: np.ndarray,
    unknown_feats: np.ndarray,
    max_k_unknown: int,
    cluster_method: str,
) -> tuple:
    """
    Grid search over k_unknown = 1..max_k_unknown.

    For each candidate k_unknown:
      k_total = k_probe + k_unknown
      all_feats = vstack([probe_feats, unknown_feats])
      preds = cluster_coarse(all_feats, k_total, method=cluster_method)
      acc = cluster_acc(probe_labels, preds[:N_probe])

    Returns (k_hat, split_margin) where:
      k_hat        : k_unknown with the best probe accuracy
      split_margin : best_score_for_k>1 - score_for_k=1
                     Positive → k>1 outperforms the k=1 baseline (confident split)
                     Negative → k=1 had the best score (confident no-split)
                     None     → only k=1 could be evaluated (max_k_unknown=1)

    Edge cases:
    - If probe is empty, cannot score → raise RuntimeError (caller should handle).
    - If unknown_feats is empty, returns (1, None).
    """
    N_probe = len(probe_feats)
    if N_probe == 0:
        raise RuntimeError(
            "grid_search_k: probe_feats is empty — cannot score any k_unknown. "
            "Ensure probe coarses have features."
        )

    if len(unknown_feats) == 0:
        return 1, None

    k_probe = int(len(np.unique(probe_labels)))
    all_feats_combined = np.vstack([probe_feats, unknown_feats]).astype(np.float32)
    N_total = len(all_feats_combined)

    best_k = 1
    best_acc = -1.0
    score_k1 = None          # score at k_unknown = 1
    best_kgt1_score = -1.0   # best score at k_unknown >= 2

    for k_unknown in range(1, max_k_unknown + 1):
        k_total = k_probe + k_unknown
        if k_total > N_total:
            break

        preds = cluster_coarse(all_feats_combined, k_total, method=cluster_method)
        probe_preds = preds[:N_probe]

        try:
            acc = float(cluster_acc(probe_labels.astype(int), probe_preds.astype(int)))
        except Exception:
            acc = 0.0

        if k_unknown == 1:
            score_k1 = acc
        elif acc > best_kgt1_score:
            best_kgt1_score = acc

        if acc > best_acc:
            best_acc = acc
            best_k = k_unknown

    # Confidence that k > 1: margin of best k>1 score over k=1 baseline
    if score_k1 is None:
        split_margin = None
    elif best_kgt1_score < 0.0:
        # max_k_unknown == 1: no k>1 candidate was evaluated
        split_margin = None
    else:
        split_margin = best_kgt1_score - score_k1

    return best_k, split_margin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe-based grid-search per-coarse K estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / features
    parser.add_argument('--dataset_name',      type=str,    default='aves_small')
    parser.add_argument('--model_name',        type=str,    default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float,  default=1.0)
    parser.add_argument('--use_ssb_splits',    type=str2bool, default=True)
    parser.add_argument('--hier_split_type',   type=str,    default=None)
    parser.add_argument('--num_novel_classes', type=int,    default=None)
    parser.add_argument('--root_dir',          type=str,    default=feature_extract_dir)
    parser.add_argument('--batch_size',        type=int,    default=256)
    parser.add_argument('--num_workers',       type=int,    default=4)
    # Coarse labels
    parser.add_argument('--coarse_labels_path', type=str,  required=True,
                        help='Path to JSON mapping fine_class_idx -> coarse_class_idx')
    parser.add_argument('--k_results_path',    type=str,   required=True,
                        help='Output path for per-coarse K JSON {coarse_id: k}')
    # Grid search
    parser.add_argument('--cluster_method',    type=str,   default='kmeans',
                        choices=CLUSTER_METHODS,
                        help='Clustering method used during the K grid search')
    parser.add_argument('--cluster_methods',   type=str,
                        default='kmeans,agglo,spectral,gmm_tied',
                        help='Comma-separated clustering methods for eval phase')
    # Probe selection
    parser.add_argument('--n_probe_total',     type=int,   default=4,
                        help='Target number of probe coarses per unknown coarse')
    parser.add_argument('--n_probe_related',   type=int,   default=2,
                        help='Max taxonomically related non-aligned probe coarses')
    # Seed
    parser.add_argument('--seed',              type=int,   default=0)
    return parser.parse_args()


def main():
    import torch
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- Class splits ---
    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # --- Coarse map ---
    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    # --- Load pre-extracted backbone features ---
    save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
    print(f'Loading features from {save_dir} ...')
    train_dataset, _, _, _ = get_datasets(
        args.dataset_name, train_transform=None, test_transform=None, args=args
    )
    all_feats, all_labels, all_mask = load_all_features(
        train_dataset, save_dir, args.batch_size, args.num_workers, device
    )
    print(f'Loaded {len(all_feats)} feature vectors (dim={all_feats.shape[1]})')
    print(f'  Labeled: {all_mask.sum()} | Unlabeled: {(~all_mask).sum()}')

    # L2-normalize
    all_feats = normalize(all_feats, norm='l2', axis=1).astype(np.float32)

    # --- Build coarse label lookup ---
    unique_labels   = np.unique(all_labels)
    label_to_coarse = {int(lbl): int(coarse_of[lbl]) for lbl in unique_labels}

    known_coarses: set   = set()
    unknown_coarses: set = set()
    for lbl in np.unique(all_labels[all_mask]):
        known_coarses.add(label_to_coarse[int(lbl)])
    for lbl in np.unique(all_labels[~all_mask]):
        unknown_coarses.add(label_to_coarse[int(lbl)])

    mixed = known_coarses & unknown_coarses
    if mixed:
        raise ValueError(
            f"Coarse constraint violated: coarses {mixed} appear in both labeled and "
            "unlabeled splits. Use --hier_split_type to get coarse-respecting splits."
        )

    print(f'Known coarses: {len(known_coarses)} | Unknown coarses: {len(unknown_coarses)}')

    # --- Oracle K for unknown coarses ---
    oracle_k: dict = {}
    for c in unknown_coarses:
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        oracle_k[c] = int(len(np.unique(all_labels[c_mask])))

    # --- max_k from ALL known coarses ---
    max_k_unknown = 1
    for c in known_coarses:
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        k_c = int(len(np.unique(all_labels[c_mask])))
        if k_c > max_k_unknown:
            max_k_unknown = k_c
    print(f'max_k_unknown (from known coarses): {max_k_unknown}')

    # --- Load taxonomy metadata (aves datasets only) ---
    coarse_to_ancestors, coarse_to_alignment, coarse_to_should_split = \
        load_taxonomy_metadata(args.dataset_name, args)

    if coarse_to_ancestors is not None:
        print('Taxonomy metadata loaded.')
    else:
        print('No taxonomy metadata — probe coarses will be selected randomly.')

    rng = np.random.default_rng(args.seed)

    # --- Per-unknown-coarse grid search ---
    k_per_coarse: dict = {}
    raw_margins: dict = {}     # {str(c): float or None}  — unnormalised confidence
    eval_cluster_methods = [m.strip() for m in args.cluster_methods.split(',')
                            if m.strip() in CLUSTER_METHODS]

    for c in sorted(unknown_coarses):
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        c_feats  = all_feats[c_mask].astype(np.float32)
        c_labels = all_labels[c_mask]

        if len(c_feats) == 0:
            print(f'  WARNING: Unknown coarse {c} has no features — skipping (k_hat=1).')
            k_per_coarse[str(c)] = 1
            raw_margins[str(c)] = None
            continue

        # Select probe coarses
        probe_coarses = select_probe_coarses(
            unknown_c=c,
            known_coarses=known_coarses,
            coarse_to_ancestors=coarse_to_ancestors,
            coarse_to_alignment=coarse_to_alignment,
            coarse_to_should_split=coarse_to_should_split,
            n_related=args.n_probe_related,
            n_total=args.n_probe_total,
            rng=rng,
        )

        if not probe_coarses:
            print(f'  WARNING: No probe coarses selected for unknown coarse {c} — '
                  f'falling back to k_hat=1.')
            k_per_coarse[str(c)] = 1
            raw_margins[str(c)] = None
            continue

        # Build probe features with contiguous 0-indexed labels
        probe_feats_list:  list = []
        probe_labels_list: list = []
        label_offset = 0
        for pc in sorted(probe_coarses):
            pc_mask = np.array([label_to_coarse[int(l)] == pc for l in all_labels], dtype=bool)
            pc_feats  = all_feats[pc_mask].astype(np.float32)
            pc_labels = all_labels[pc_mask]
            if len(pc_feats) == 0:
                continue
            unique_fine = np.unique(pc_labels)
            remap = {int(old): label_offset + i for i, old in enumerate(unique_fine)}
            probe_labels_list.append(
                np.array([remap[int(l)] for l in pc_labels], dtype=np.int64)
            )
            probe_feats_list.append(pc_feats)
            label_offset += len(unique_fine)

        if not probe_feats_list:
            print(f'  WARNING: All probe coarses are empty for unknown coarse {c} — '
                  f'falling back to k_hat=1.')
            k_per_coarse[str(c)] = 1
            raw_margins[str(c)] = None
            continue

        probe_feats  = np.vstack(probe_feats_list)
        probe_labels = np.concatenate(probe_labels_list)

        oracle_k_c = oracle_k.get(c, 1)
        print(f'\n--- Unknown coarse {c} | oracle_k={oracle_k_c} | '
              f'n_unknown_samples={len(c_feats)} | '
              f'probe_coarses={sorted(probe_coarses)} | '
              f'n_probe_samples={len(probe_feats)} ---')

        try:
            k_hat_c, margin = grid_search_k(
                probe_feats=probe_feats,
                probe_labels=probe_labels,
                unknown_feats=c_feats,
                max_k_unknown=max_k_unknown,
                cluster_method=args.cluster_method,
            )
        except Exception as exc:
            print(f'  WARNING: grid_search_k failed ({exc}) — falling back to k_hat=1.')
            k_hat_c, margin = 1, None

        print(f'  k_hat={k_hat_c}  (oracle_k={oracle_k_c})'
              + (f'  split_margin={margin:.4f}' if margin is not None else ''))
        k_per_coarse[str(c)] = k_hat_c
        raw_margins[str(c)] = margin

    # --- Save k_results JSON ---
    os.makedirs(os.path.dirname(os.path.abspath(args.k_results_path)), exist_ok=True)
    with open(args.k_results_path, 'w') as f:
        json.dump(k_per_coarse, f, indent=2)

    total_k_unknown = sum(k_per_coarse.values())
    print(f'\nPer-coarse K: {k_per_coarse}')
    print(f'Total unknown clusters: {total_k_unknown}')

    # --- Normalise margins to [0, 1] for the confidence / precision curve ---
    # Margin = best_score_for_k>1 - score_k1 (positive → confident split).
    # Coarses with None margin (no k>1 candidate evaluated) get confidence 0.
    known_margins = [m for m in raw_margins.values() if m is not None]
    if known_margins:
        mn = float(min(known_margins))
        mx = float(max(known_margins))
        span = mx - mn if mx > mn else 1.0
        def _normalise(m):
            return (m - mn) / span if m is not None else 0.0
    else:
        def _normalise(m):
            return 0.0

    confidence_per_coarse = {
        sid: round(_normalise(raw_margins[sid]), 6)
        for sid in k_per_coarse
    }

    oracle_k_int = {int(c): k for c, k in oracle_k.items()}
    precision_curve = compute_precision_curve(
        confidence_per_coarse, oracle_k_int, n_thresholds=21
    )
    method_label = f'brent_{args.cluster_method}'
    print_precision_summary(precision_curve, method_name=method_label)

    confidence_scores_out = {
        sid: {
            "confidence": confidence_per_coarse[sid],
            "confidence_raw": raw_margins.get(sid),
            "k_hat": int(k_per_coarse[sid]),
            "k_true": int(oracle_k.get(int(sid), 1)),
        }
        for sid in k_per_coarse
    }

    # --- Evaluate clustering with k_per_coarse estimates ---
    print('\nRunning per-coarse clustering evaluation ...')
    per_coarse = eval_per_coarse(
        all_feats=all_feats,
        all_labels=all_labels,
        coarse_of=coarse_of,
        k_per_coarse=k_per_coarse,
        oracle_k=oracle_k_int,
        methods=eval_cluster_methods,
    )

    aggregate = aggregate_metrics(per_coarse, methods=eval_cluster_methods)
    print('\nAggregate metrics:')
    print_aggregate_summary(aggregate, methods=eval_cluster_methods)

    # --- Save metrics JSON ---
    metrics_path = args.k_results_path.replace('.json', '_metrics.json')
    method_name = f'brent_{args.cluster_method}'
    save_hier_metrics(
        k_per_coarse=k_per_coarse,
        per_coarse=per_coarse,
        aggregate=aggregate,
        method_name=method_name,
        dataset_name=args.dataset_name,
        output_path=metrics_path,
        extra_data={
            "confidence_scores": confidence_scores_out,
            "precision_curve": precision_curve,
        },
    )

    print(f'\nK estimation complete: {args.k_results_path}')


if __name__ == '__main__':
    main()
