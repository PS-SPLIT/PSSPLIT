"""
estimate_k_hier_baselines.py — Simple, uninformed lower-bound baselines for
per-coarse K estimation on the category-splitting task.

Requested by reviewers ("There should be a lower bound reported (e.g., a
random baseline) for the reported evaluations given the newness of the
task.") The aves_small / aves_full splitting task has, per unknown coarse,
k_true in {1, 2}: is this pair of aligned species actually one fine class or
two? Five baselines that use no learned signal whatsoever:

  always_split        : k_hat = 2 for every unknown coarse, then KMeans(k=2).
  always_split_random : k_hat = 2 for every unknown coarse (same as above),
                        but points are assigned to one of the 2 clusters
                        uniformly at random instead of running KMeans — i.e.
                        "we're sure it splits" with no feature signal at all.
  never_split         : k_hat = 1 for every unknown coarse — single cluster,
                        no clustering call at all.
  random_kmeans       : k_hat ~ Bernoulli(0.5) -> {1, 2} per coarse (fair
                        coin), then KMeans(k_hat) when k_hat=2.
  random_assign       : same coin flip; when k_hat=2, points are assigned to
                        one of the 2 clusters uniformly at random — no
                        clustering algorithm involved at all.

always_split / never_split are fully deterministic (features + KMeans with
fixed random_state, matching hier_eval_utils.cluster_coarse). always_split_random,
random_kmeans, and random_assign are stochastic, so results are Monte-Carlo
averaged over --n_repeats independent random draws and reported as
mean +/- std, exactly mirroring the multi-seed averaging already used
elsewhere in this project (build_combined_aves_method_tables.py).

Output format matches every other estimate_k_hier_*.py script:
  - JSON: {str(coarse_id): k_i}                    -> --k_results_path
          (for stochastic baselines this is one representative draw, seed 0)
  - Metrics JSON: k_results_path.replace('.json', '_metrics.json')
    'aggregate' holds the mean over repeats (drop-in compatible with the
    existing table-building scripts); 'aggregate_std' holds the std.
  - Prints "K estimation complete: {k_results_path}"

Run standalone:
    python -m methods.estimate_k_hier.estimate_k_hier_baselines \\
        --dataset_name aves_small \\
        --coarse_labels_path data/coarse_labels/aves_small.json \\
        --hier_split_type 5050 \\
        --baseline always_split \\
        --k_results_path /tmp/aves_small_always_split_k.json
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize

from config import feature_extract_dir
from data.data_utils import build_coarse_of, load_coarse_map
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from methods.estimate_k.estimate_k_tuning import labelled_k_score
from methods.estimate_k_hier.estimate_k_hier import load_all_features
from methods.estimate_k_hier.hier_eval_utils import cluster_coarse
from project_utils.cluster_utils import cluster_acc
from project_utils.general_utils import str2bool

BASELINES = ['always_split', 'always_split_random', 'never_split',
             'random_kmeans', 'random_assign']

DETERMINISTIC_BASELINES = {'always_split', 'never_split'}


# ---------------------------------------------------------------------------
# Per-coarse feature caching
# ---------------------------------------------------------------------------

def build_coarse_cache(all_feats, all_labels, coarse_of, unknown_coarses, oracle_k):
    """
    Precompute, for every unknown coarse: features, labels, k_true, and the
    (deterministic) KMeans(k=2) cluster assignment — so Monte-Carlo trials
    over the coin flip never have to refit KMeans.
    """
    coarse_to_fine = {}
    for label_idx, c in enumerate(coarse_of.tolist()):
        coarse_to_fine.setdefault(int(c), set()).add(label_idx)

    cache = {}
    for c in sorted(unknown_coarses):
        fine_classes = coarse_to_fine.get(c, set())
        mask_c = np.array([int(l) in fine_classes for l in all_labels], dtype=bool)
        if mask_c.sum() < 2:
            continue
        c_feats = all_feats[mask_c].astype(np.float32)
        c_labels = all_labels[mask_c]
        kmeans_k2_preds = cluster_coarse(c_feats, 2, method='kmeans')
        cache[c] = {
            'feats': c_feats,
            'labels': c_labels,
            'n': len(c_feats),
            'k_true': int(oracle_k.get(c, 1)),
            'kmeans_k2_preds': kmeans_k2_preds,
        }
    return cache


def _acc_ami(labels, preds):
    try:
        acc = float(cluster_acc(labels.astype(int), preds.astype(int)))
        ami = float(adjusted_mutual_info_score(labels, preds))
    except Exception:
        acc, ami = float('nan'), float('nan')
    return acc, ami


# ---------------------------------------------------------------------------
# Single-trial per-coarse evaluation for one baseline
# ---------------------------------------------------------------------------

def run_one_trial(cache, baseline, rng):
    """
    Draw one realisation of k_per_coarse (+ predictions) for `baseline` and
    return the per_coarse dict: {coarse_id: {k_true, k_hat, k_score,
    kmeans_acc, kmeans_ami}}.  Stored under the 'kmeans_' key regardless of
    baseline so it plugs directly into the existing table-building code,
    which looks metrics up by clustering-method name.
    """
    per_coarse = {}
    for c, entry in cache.items():
        k_true = entry['k_true']
        n = entry['n']

        if baseline in ('always_split', 'always_split_random'):
            k_hat = 2
        elif baseline == 'never_split':
            k_hat = 1
        elif baseline in ('random_kmeans', 'random_assign'):
            k_hat = 2 if rng.random() < 0.5 else 1
        else:
            raise ValueError(f'Unknown baseline {baseline!r}')

        if k_hat <= 1:
            preds = np.zeros(n, dtype=np.int64)
        elif baseline in ('always_split', 'random_kmeans'):
            preds = entry['kmeans_k2_preds']
        elif baseline in ('always_split_random', 'random_assign'):
            preds = rng.integers(0, 2, size=n)
        else:
            raise ValueError(f'Unknown baseline {baseline!r}')

        acc, ami = _acc_ami(entry['labels'], preds)
        per_coarse[c] = {
            'k_true': k_true,
            'k_hat': int(k_hat),
            'k_score': float(labelled_k_score(k_hat, k_true)),
            'kmeans_acc': acc,
            'kmeans_ami': ami,
        }
    return per_coarse


# ---------------------------------------------------------------------------
# Aggregation (mirrors hier_eval_utils.aggregate_metrics + the k-stats used
# by bash_scripts/build_combined_aves_method_tables.py)
# ---------------------------------------------------------------------------

def summarise_trial(per_coarse):
    """One trial's per_coarse dict -> flat dict of scalar summary stats."""
    all_entries = list(per_coarse.values())
    multi_entries = [e for e in all_entries if e['k_true'] > 1]

    def _nanmean(vals):
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float('nan')

    def _k_acc(entries):
        if not entries:
            return float('nan')
        return _nanmean([e['k_score'] for e in entries])

    def _split_prec(entries):
        splits = [(e['k_hat'], e['k_true']) for e in entries if e['k_hat'] > 1]
        if not splits:
            return float('nan')
        return sum(1 for _, kt in splits if kt > 1) / len(splits)

    total_k_hat = sum(e['k_hat'] for e in all_entries)
    total_k_true = sum(e['k_true'] for e in all_entries)

    return {
        'acc_all':     _nanmean([e['kmeans_acc'] for e in all_entries]),
        'acc_multi':   _nanmean([e['kmeans_acc'] for e in multi_entries]),
        'ami_all':     _nanmean([e['kmeans_ami'] for e in all_entries]),
        'ami_multi':   _nanmean([e['kmeans_ami'] for e in multi_entries]),
        'k_acc_all':   _k_acc(all_entries),
        'k_acc_multi': _k_acc(multi_entries),
        'k_diff':      total_k_hat - total_k_true,
        'split_prec':  _split_prec(all_entries),
    }


def mean_std(dicts):
    keys = dicts[0].keys()
    mean, std = {}, {}
    for k in keys:
        vals = [d[k] for d in dicts if d[k] == d[k]]  # drop NaN
        if not vals:
            mean[k], std[k] = None, None
            continue
        mean[k] = float(np.mean(vals))
        std[k] = float(np.std(vals)) if len(vals) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple lower-bound baselines for per-coarse K estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset_name',      type=str,    default='aves_small')
    parser.add_argument('--model_name',        type=str,    default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float,  default=1.0)
    parser.add_argument('--use_ssb_splits',    type=str2bool, default=True)
    parser.add_argument('--hier_split_type',   type=str,    default=None)
    parser.add_argument('--num_novel_classes', type=int,    default=None)
    parser.add_argument('--root_dir',          type=str,    default=feature_extract_dir)
    parser.add_argument('--batch_size',        type=int,    default=256)
    parser.add_argument('--num_workers',       type=int,    default=4)
    parser.add_argument('--coarse_labels_path', type=str,  required=True)
    parser.add_argument('--k_results_path',    type=str,   required=True)
    parser.add_argument('--baseline',          type=str,   required=True, choices=BASELINES)
    parser.add_argument('--n_repeats',         type=int,   default=200,
                        help='Monte-Carlo repeats for always_split_random / random_kmeans / '
                             'random_assign; ignored (=1) for always_split / never_split.')
    parser.add_argument('--seed',              type=int,   default=0)
    return parser.parse_args()


def main():
    import torch
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
    print(f'Loading features from {save_dir} ...')
    train_dataset, _, _, _ = get_datasets(
        args.dataset_name, train_transform=None, test_transform=None, args=args
    )
    all_feats, all_labels, all_mask = load_all_features(
        train_dataset, save_dir, args.batch_size, args.num_workers, device
    )
    print(f'Loaded {len(all_feats)} feature vectors (dim={all_feats.shape[1]})')
    all_feats = normalize(all_feats, norm='l2', axis=1).astype(np.float32)

    unique_labels   = np.unique(all_labels)
    label_to_coarse = {int(lbl): int(coarse_of[lbl]) for lbl in unique_labels}
    known_coarses, unknown_coarses = set(), set()
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

    oracle_k = {}
    for c in unknown_coarses:
        c_mask = np.array([label_to_coarse[int(l)] == c for l in all_labels], dtype=bool)
        oracle_k[c] = int(len(np.unique(all_labels[c_mask])))

    cache = build_coarse_cache(all_feats, all_labels, coarse_of, unknown_coarses, oracle_k)

    n_repeats = 1 if args.baseline in DETERMINISTIC_BASELINES else args.n_repeats
    rng = np.random.default_rng(args.seed)

    trial_summaries = []
    per_coarse_trial0 = None
    for t in range(n_repeats):
        per_coarse_t = run_one_trial(cache, args.baseline, rng)
        if t == 0:
            per_coarse_trial0 = per_coarse_t
        trial_summaries.append(summarise_trial(per_coarse_t))

    mean, std = mean_std(trial_summaries)

    print(f'\nBaseline: {args.baseline}  (n_repeats={n_repeats})')
    print(f"  Cl.Acc.   all={mean['acc_all']:.4f} +/- {std['acc_all']:.4f}   "
          f"multi={mean['acc_multi']:.4f} +/- {std['acc_multi']:.4f}")
    print(f"  AMI       all={mean['ami_all']:.4f} +/- {std['ami_all']:.4f}   "
          f"multi={mean['ami_multi']:.4f} +/- {std['ami_multi']:.4f}")
    print(f"  K Acc.    all={mean['k_acc_all']:.4f} +/- {std['k_acc_all']:.4f}   "
          f"multi={mean['k_acc_multi']:.4f} +/- {std['k_acc_multi']:.4f}")
    print(f"  Delta_sum_K = {mean['k_diff']:+.2f} +/- {std['k_diff']:.2f}")
    sp_mean = mean['split_prec']
    sp_std = std['split_prec']
    sp_str = f"{sp_mean:.4f} +/- {sp_std:.4f}" if sp_mean is not None else "--"
    print(f"  Split Prec. = {sp_str}")

    # --- Save representative k_per_coarse JSON (trial 0) ---
    k_per_coarse_repr = {str(c): int(entry['k_hat']) for c, entry in per_coarse_trial0.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.k_results_path)), exist_ok=True)
    with open(args.k_results_path, 'w') as f:
        json.dump(k_per_coarse_repr, f, indent=2)

    # --- Save metrics JSON ---
    # 'aggregate' mirrors hier_eval_utils.aggregate_metrics()'s shape so the
    # existing table-building scripts can read {method}_acc / {method}_ami
    # directly off the 'kmeans' key, exactly like every other baseline row.
    aggregate = {
        'all': {
            'n_coarses':  len(cache),
            'k_accuracy': mean['k_acc_all'],
            'kmeans_acc': mean['acc_all'],
            'kmeans_ami': mean['ami_all'],
        },
        'multi_class': {
            'n_coarses':  sum(1 for e in cache.values() if e['k_true'] > 1),
            'k_accuracy': mean['k_acc_multi'],
            'kmeans_acc': mean['acc_multi'],
            'kmeans_ami': mean['ami_multi'],
        },
    }
    per_coarse_serialisable = {
        str(c): {k: (None if isinstance(v, float) and v != v else v) for k, v in entry.items()}
        for c, entry in per_coarse_trial0.items()
    }

    results = {
        'method': args.baseline,
        'dataset': args.dataset_name,
        'n_repeats': n_repeats,
        'k_per_coarse': k_per_coarse_repr,
        'per_coarse': per_coarse_serialisable,
        'aggregate': aggregate,
        'summary_mean': mean,
        'summary_std': std,
    }

    metrics_path = args.k_results_path.replace('.json', '_metrics.json')
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Metrics saved to {metrics_path}')

    print(f'\nK estimation complete: {args.k_results_path}')


if __name__ == '__main__':
    main()
