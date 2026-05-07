import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from project_utils.cluster_utils import cluster_acc
from project_utils.general_utils import str2bool

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from data.get_datasets import get_datasets, get_class_splits, restrict_unlabeled_classes

from config import feature_extract_dir
from tqdm import tqdm

from scipy.optimize import minimize_scalar
from functools import partial

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Evaluation helpers (used when --evaluate is set)
# ---------------------------------------------------------------------------

def collect_train_features(train_loader, device):
    """
    Single pass over a MergedDataset train_loader; returns L2-normalised features,
    labels, and mask_lab (True = labelled instance).
    """
    feats_list, labels_list, mask_list = [], [], []
    for feats, label, _, mask_lab in train_loader:
        feats = F.normalize(feats.float().to(device), dim=-1)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(label.cpu().numpy())
        mask_list.append(mask_lab.cpu().bool().numpy())
    all_feats  = np.concatenate(feats_list,  axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_mask   = np.concatenate(mask_list,   axis=0).astype(bool)
    return all_feats, all_labels, all_mask


def collect_test_features(test_loader, device):
    """
    Single pass over a standard (non-Merged) test DataLoader.
    Returns L2-normalised features and labels.
    """
    feats_list, labels_list = [], []
    for feats, label, _ in test_loader:
        feats = F.normalize(feats.float().to(device), dim=-1)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(label.cpu().numpy())
    return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)


def predict_nearest_centroid(train_feats, train_preds, test_feats):
    """Assign each test point to the nearest training cluster centroid."""
    unique_clusters = np.unique(train_preds)
    centroids = np.array([
        train_feats[train_preds == i].mean(axis=0)
        for i in unique_clusters
    ])
    dists = np.linalg.norm(test_feats[:, None] - centroids[None, :], axis=-1)
    return unique_clusters[np.argmin(dists, axis=1)]


def evaluate_assignments(all_feats, all_labels, preds, args, split_name, subset_mask=None):
    """
    Compute and print split ACC + AMI for a clustering assignment.

    subset_mask : optional bool array to restrict evaluation to a subset
                  (e.g. ~mask_lab to evaluate only unlabelled instances).
    """
    from project_utils.cluster_and_log_utils import split_cluster_acc_v2, split_ami_score

    if subset_mask is not None:
        all_labels = all_labels[subset_mask]
        preds      = preds[subset_mask]

    mask_cls = (all_labels < args.num_labeled_classes)
    # Guard: need at least one sample in each subset for AMI
    if mask_cls.sum() == 0 or (~mask_cls).sum() == 0:
        print(f"[EVAL/{split_name}] skipped (no old or new class samples)")
        return

    acc_all, acc_old, acc_new = split_cluster_acc_v2(
        all_labels.astype(int), preds.astype(int), mask_cls)
    ami_all, ami_old, ami_new = split_ami_score(
        all_labels.astype(int), preds.astype(int), mask_cls)

    print(f"[EVAL/{split_name}] ACC all={acc_all:.4f} old={acc_old:.4f} new={acc_new:.4f}")
    print(f"[EVAL/{split_name}] AMI all={ami_all:.4f} old={ami_old:.4f} new={ami_new:.4f}")


def run_evaluation(result, test_loader, args, device):
    """
    Run all three evaluation splits (train_all, train_unlabeled, test) and print
    [EVAL/...] lines parseable by test_estimate_k_methods.py.
    """
    from sklearn.mixture import GaussianMixture

    all_feats    = result['all_feats']
    all_labels   = result['all_labels']
    mask_lab     = result['mask_lab']
    preds        = result['preds']
    pca          = result['pca']
    fitted_model = result.get('fitted_model')

    print("\n--- Clustering Evaluation ---")

    # Train-all
    evaluate_assignments(all_feats, all_labels, preds, args, 'train_all')

    # Train-unlabeled only (~mask_lab)
    evaluate_assignments(all_feats, all_labels, preds, args, 'train_unlabeled',
                         subset_mask=~mask_lab)

    # Test: collect features, optionally apply same PCA, predict assignments
    print("Collecting test features for evaluation...")
    test_feats_raw, test_labels = collect_test_features(test_loader, device)

    if pca is not None:
        from sklearn.preprocessing import normalize as sk_normalize
        test_feats_pca = pca.transform(test_feats_raw)
        test_feats = sk_normalize(test_feats_pca, norm='l2', axis=1).astype(np.float32)
    else:
        test_feats = test_feats_raw

    if isinstance(fitted_model, GaussianMixture):
        test_preds = fitted_model.predict(test_feats).astype(np.int64)
    else:
        test_preds = predict_nearest_centroid(all_feats, preds, test_feats)
    evaluate_assignments(test_feats, test_labels, test_preds, args, 'test')


def def_result_cluster(all_feats, all_labels, mask_lab, pca, k_hat, method, n_neighbors=10,
                        covariance_type='full', init_params='kmeans'):
    """Cluster pre-collected features with the given method and return a result dict."""
    from methods.estimate_k.estimate_k_tuning import _cluster_with_method
    print(f"  Fitting final clustering (method={method}, k={k_hat}) on all training features...")
    preds, fitted_model = _cluster_with_method(all_feats, k_hat, method,
                                               n_neighbors=n_neighbors, return_model=True,
                                               covariance_type=covariance_type,
                                               init_params=init_params)
    return dict(
        all_feats=all_feats,
        all_labels=all_labels.astype(np.int64),
        mask_lab=mask_lab,
        preds=preds,
        pca=pca,
        fitted_model=fitted_model,
    )


def _collect_and_pca(train_loader, args):
    """
    Collect all training features once and optionally apply PCA.

    Returns
    -------
    all_feats  : np.ndarray (N, D')  possibly PCA-reduced, float32
    all_labels : np.ndarray (N,)
    mask_lab   : np.ndarray (N,) bool
    pca        : fitted PCA or None
    """
    all_feats, all_labels, mask_lab = collect_train_features(train_loader, device)
    pca_dim = int(getattr(args, 'pca_dim', 0))
    pca = None
    if pca_dim > 0:
        from methods.estimate_k.estimate_k_agglo import apply_pca_and_norm
        l_pca, u_pca, pca = apply_pca_and_norm(
            all_feats[mask_lab], all_feats[~mask_lab], pca_dim
        )
        all_feats_new = np.empty(
            (len(all_feats), l_pca.shape[1]), dtype=np.float32
        )
        all_feats_new[mask_lab]  = l_pca
        all_feats_new[~mask_lab] = u_pca
        all_feats = all_feats_new
    return all_feats, all_labels, mask_lab, pca


def _score_on_feats(k, all_feats, mask_lab, all_labels, method, n_neighbors, verbose=False):
    """
    Cluster all_feats into k groups and return labelled-set accuracy.

    Returns positive accuracy (higher = better).
    """
    from methods.estimate_k.estimate_k_tuning import _cluster_with_method
    preds = _cluster_with_method(all_feats, k, method, n_neighbors=n_neighbors)
    acc = cluster_acc(all_labels.astype(int)[mask_lab], preds.astype(int)[mask_lab])
    if verbose:
        nmi = nmi_score(all_labels[mask_lab], preds[mask_lab])
        ari = ari_score(all_labels[mask_lab], preds[mask_lab])
        print(f'K = {k}')
        print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return acc


def binary_search(merge_test_loader, args):

    cluster_method = getattr(args, 'final_cluster_method', 'auto')
    n_neighbors    = int(getattr(args, 'spectral_n_neighbors', 10))
    # Resolve scoring method: 'auto' → 'kmeans' for brent/binary
    score_method = 'kmeans' if cluster_method == 'auto' else cluster_method

    print('Collecting and pre-processing features (once)...')
    all_feats, all_labels, mask_lab, pca = _collect_and_pca(merge_test_loader, args)

    min_classes = args.num_labeled_classes

    # Iter 0
    big_k    = args.max_classes
    small_k  = min_classes
    diff     = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    labelled_acc_big    = _score_on_feats(big_k,    all_feats, mask_lab, all_labels, score_method, n_neighbors)
    labelled_acc_small  = _score_on_feats(small_k,  all_feats, mask_lab, all_labels, score_method, n_neighbors)
    labelled_acc_middle = _score_on_feats(middle_k, all_feats, mask_lab, all_labels, score_method, n_neighbors)

    print(f'Iter 0: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
    all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k   = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):

        if labelled_acc_big > labelled_acc_small:

            small_k = middle_k
            labelled_acc_small = labelled_acc_middle
            diff     = big_k - small_k
            middle_k = int(0.5 * diff + small_k)

        else:

            big_k    = middle_k
            diff     = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            labelled_acc_big = labelled_acc_middle

        labelled_acc_middle = _score_on_feats(middle_k, all_feats, mask_lab, all_labels, score_method, n_neighbors)

        print(f'Iter {i}: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
        all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k   = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    k_hat = int(best_acc_at_k)
    print(f'Optimal K is {k_hat}')
    result  = def_result_cluster(all_feats, all_labels, mask_lab, pca, k_hat, score_method, n_neighbors)
    k_novel = max(1, k_hat - args.num_labeled_classes)
    return k_hat, k_novel, result


def scipy_optimise(merge_test_loader, args):

    cluster_method = getattr(args, 'final_cluster_method', 'auto')
    n_neighbors    = int(getattr(args, 'spectral_n_neighbors', 10))
    score_method   = 'kmeans' if cluster_method == 'auto' else cluster_method

    print('Collecting and pre-processing features (once)...')
    all_feats, all_labels, mask_lab, pca = _collect_and_pca(merge_test_loader, args)

    small_k = args.num_labeled_classes
    big_k   = args.max_classes

    def _neg_score(K):
        return -_score_on_feats(int(K), all_feats, mask_lab, all_labels,
                                score_method, n_neighbors, verbose=True)

    res   = minimize_scalar(_neg_score, bounds=(small_k, big_k), method='bounded', options={'disp': True})
    k_hat = int(round(res.x))
    print(f'Optimal K is {k_hat}')
    result  = def_result_cluster(all_feats, all_labels, mask_lab, pca, k_hat, score_method, n_neighbors)
    k_novel = max(1, k_hat - args.num_labeled_classes)
    return k_hat, k_novel, result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_classes', default=1000, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_exp_id', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--search_mode', type=str, default='brent', help='Mode for black box optimisation')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--probe_prop', type=float, default=0.2)
    parser.add_argument('--val_prop', type=float, default=0.2)
    parser.add_argument('--outlier_tau', type=float, default=0.01)
    # Agglo-specific
    parser.add_argument('--pca_dim', type=int, default=64)
    parser.add_argument('--agglo_linkages', type=str, default='ward,complete,average')
    parser.add_argument('--agglo_n_grid', type=int, default=25)
    parser.add_argument('--agglo_lo', type=float, default=0.0)
    parser.add_argument('--agglo_hi', type=float, default=10.0)
    parser.add_argument('--agglo_use_combined', action='store_true')
    parser.add_argument('--agglo_max_samples', type=int, default=5000)
    parser.add_argument('--agglo_max_total_samples', type=int, default=10_000)
    # Spectral-specific
    parser.add_argument('--spectral_n_neighbors', type=int, default=10,
                        help='(Spectral) number of neighbors for kNN affinity graph')
    parser.add_argument('--spectral_n_neighbors_candidates', type=str, default='10',
                        help='Comma-separated n_neighbors values for spectral tuning grid '
                             '(used when tuning_criterion != labelled_k or multiple values given)')
    parser.add_argument('--spectral_affinity', type=str, default='knn_euclidean',
                        choices=['knn_euclidean', 'knn_cosine', 'rbf',
                                 'cosine', 'polynomial', 'sigmoid', 'laplacian'],
                        help='Affinity graph type for spectral eigengap. '
                             'knn_euclidean = Gaussian-weighted kNN with Euclidean distances (default); '
                             'knn_cosine = same with cosine distances; '
                             'rbf = full pairwise RBF kernel (O(N^2)); '
                             'cosine/polynomial/sigmoid/laplacian = sklearn pairwise kernels (O(N^2)).')
    parser.add_argument('--spectral_gamma', type=float, default=None,
                        help='Gamma kernel coefficient (default: 1/n_features). '
                             'Used by rbf, cosine, polynomial, sigmoid, laplacian.')
    parser.add_argument('--spectral_gamma_candidates', type=str, default='auto',
                        help='Comma-separated gamma values for tuning grid. '
                             '"auto" expands to 1/n_features. '
                             'Used by rbf and sklearn pairwise kernel affinities.')
    parser.add_argument('--spectral_degree', type=float, default=3.0,
                        help='Degree for polynomial kernel. Ignored by other kernels.')
    parser.add_argument('--spectral_coef0', type=float, default=1.0,
                        help='Zero coefficient for polynomial and sigmoid kernels.')
    parser.add_argument('--spectral_assign_labels', type=str, default='kmeans',
                        choices=['kmeans', 'discretize', 'cluster_qr'],
                        help='SpectralClustering assign_labels method (default: kmeans).')
    # Agglo metric option
    parser.add_argument('--agglo_metric', type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help='Distance metric for AgglomerativeClustering. '
                             'cosine is incompatible with ward linkage — ward will be '
                             'automatically removed from the linkage candidates.')
    # XMeans-specific
    parser.add_argument('--xmeans_criterion', type=int, default=0,
                        help='XMeans splitting criterion: 0=BIC (default), 1=MNDL')
    parser.add_argument('--xmeans_criterion_candidates', type=str, default='0',
                        help='Comma-separated xmeans criterion values to search over '
                             '(e.g. "0,1" to try both BIC and MNDL)')
    # GMM-specific (used by brent_tuned)
    parser.add_argument('--gmm_covariance_type_candidates', type=str, default='full,diag',
                        help='(brent_tuned/gmm) Comma-separated GMM covariance types to search '
                             'in Phase-1 tuning. Options: full, tied, diag, spherical.')
    parser.add_argument('--gmm_init_params_candidates', type=str, default='kmeans',
                        help='(brent_tuned/gmm) Comma-separated GMM init strategies. '
                             'Options: kmeans, k-means++, random, random_from_data.')
    # GMM single-value args (used by fixed-K brent path)
    parser.add_argument('--gmm_covariance_type', type=str, default='full',
                        choices=['full', 'tied', 'diag', 'spherical'],
                        help='(fixed-K brent) GMM covariance type. Default: full.')
    parser.add_argument('--gmm_init_params', type=str, default='kmeans',
                        choices=['kmeans', 'k-means++', 'random', 'random_from_data'],
                        help='(fixed-K brent) GMM initialization strategy. Default: kmeans.')
    # Shared tuning framework (agglo, spectral, xmeans, gmeans, brent_tuned)
    parser.add_argument('--tuning_criterion', type=str, default='labelled_k',
                        choices=['labelled_k', 'test_cluster', 'combined',
                                 'labelled_cluster'],
                        help='Criterion for hyperparameter selection: '
                             'labelled_k = recover known k on labelled features (default); '
                             'test_cluster = old-class ACC on test split; '
                             'combined = weighted combination of both; '
                             'labelled_cluster = cluster_acc on labelled training features '
                             'at k_true (used as default by brent_tuned)')
    parser.add_argument('--tuning_weight', type=float, default=0.5,
                        help='Weight for labelled_k score in combined criterion '
                             '(test gets 1-weight)')
    parser.add_argument('--tune_pca', action='store_true',
                        help='Include pca_dim as a tunable hyperparameter; '
                             'searches over --pca_dim_candidates')
    parser.add_argument('--pca_dim_candidates', type=str, default='32,64,128',
                        help='Comma-separated PCA dimensionalities to search over '
                             'when --tune_pca is set')
    # HDBSCAN-specific
    parser.add_argument('--hdbscan_n_trials', type=int, default=120,
                        help='(HDBSCAN) number of Optuna trials for hyperparameter search')
    parser.add_argument('--hdbscan_max_labelled_samples', type=int, default=2000,
                        help='(HDBSCAN) max labelled samples used in Optuna tuning (speed guard)')
    # Final clustering method (brent/binary/autonovel/spectral)
    parser.add_argument('--final_cluster_method', type=str, default='auto',
                        choices=['auto', 'kmeans', 'spectral', 'gmm', 'agglo'],
                        help='Clustering algorithm for final result (and scoring for brent/binary). '
                             '"auto" uses each method\'s historical default (kmeans for '
                             'brent/binary/autonovel, spectral for spectral). '
                             '"agglo" uses AgglomerativeClustering(n_clusters=k, linkage=ward); '
                             'test split is assigned via nearest centroid.')
    # Fixed-K mode (bypasses K estimation entirely)
    parser.add_argument('--fixed_k', type=int, default=None,
                        help='Use this exact K instead of estimating it. '
                             'When set: brent/binary skip optimisation and cluster at fixed_k; '
                             'agglo uses n_clusters=fixed_k and tunes linkage/pca_dim; '
                             'spectral uses n_clusters=fixed_k and tunes affinity/assign_labels/n_neighbors/pca_dim.')
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='After K estimation, evaluate clustering on train and test splits')
    parser.add_argument('--num_novel_classes', type=int, default=None,
                        help='Limit the number of novel/unlabeled classes used '
                             '(None = use all classes from the dataset split)')
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True,
                        help='Use Semantic Shift Benchmark class splits for FGVC datasets '
                             '(scars, cub, aircraft). Must match the parent pipeline setting.')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cluster_accs = {}

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(args)

    if args.warmup_model_exp_id is not None:
        args.save_dir += '_' + args.warmup_model_exp_id
        print(f'Using features from experiment: {args.warmup_model_exp_id}')
    else:
        print(f'Using pretrained {args.model_name} features...')

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    train_transform, test_transform = None, None
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform, test_transform, args)

    # Convert to feature vector dataset
    test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=os.path.join(args.save_dir, 'test'))
    unlabelled_train_examples_test = FeatureVectorDataset(base_dataset=unlabelled_train_examples_test,
                                                          feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset = FeatureVectorDataset(base_dataset=train_dataset, feature_root=os.path.join(args.save_dir, 'train'))

    # --------------------
    # DATALOADERS
    # --------------------
    unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                         batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False)

    import json as _json

    fixed_k = getattr(args, 'fixed_k', None)

    print('Testing on all in the training data...')
    if fixed_k is not None and args.search_mode in ('brent', 'binary'):
        # Fixed-K brent/binary: skip optimisation entirely, cluster at fixed_k
        cluster_method  = getattr(args, 'final_cluster_method', 'auto')
        n_neighbors     = int(getattr(args, 'spectral_n_neighbors', 10))
        score_method    = 'kmeans' if cluster_method == 'auto' else cluster_method
        covariance_type = getattr(args, 'gmm_covariance_type', 'full')
        init_params     = getattr(args, 'gmm_init_params', 'kmeans')
        print(f'Fixed-K mode: skipping optimisation, clustering at k={fixed_k} '
              f'(method={score_method})')
        all_feats, all_labels, mask_lab, pca = _collect_and_pca(train_loader, args)
        result = def_result_cluster(all_feats, all_labels, mask_lab, pca,
                                    fixed_k, score_method, n_neighbors,
                                    covariance_type=covariance_type,
                                    init_params=init_params)
        print(f'Optimal K is {fixed_k}')
        best_params_dict = {'final_cluster_method': score_method,
                            'pca_dim': int(getattr(args, 'pca_dim', 0))}
        if score_method == 'gmm':
            best_params_dict['covariance_type'] = covariance_type
            best_params_dict['init_params'] = init_params
        print(f'Best params: {_json.dumps(best_params_dict)}')
    elif fixed_k is not None and args.search_mode == 'agglo':
        from methods.estimate_k.estimate_k_agglo import estimate_k_agglo_fixed_k
        _, _, result = estimate_k_agglo_fixed_k(train_loader, args, device,
                                                 test_loader=test_loader, k=fixed_k)
    elif fixed_k is not None and args.search_mode == 'spectral':
        from methods.estimate_k.estimate_k_spectral import estimate_k_spectral_fixed_k
        _, _, result = estimate_k_spectral_fixed_k(train_loader, args, device,
                                                    test_loader=test_loader, k=fixed_k)
    elif args.search_mode == 'brent':
        print('Optimising with Brents algorithm')
        _, _, result = scipy_optimise(merge_test_loader=train_loader, args=args)
    elif args.search_mode == 'autonovel':
        from methods.estimate_k.estimate_k_autonovel import estimate_k_autonovel
        _, _, result = estimate_k_autonovel(train_loader, args, device)
    elif args.search_mode == 'agglo':
        from methods.estimate_k.estimate_k_agglo import estimate_k_agglo
        _, _, result = estimate_k_agglo(train_loader, args, device,
                                        test_loader=test_loader)
    elif args.search_mode == 'xmeans':
        from methods.estimate_k.estimate_k_xgmeans import estimate_k_xmeans
        _, _, result = estimate_k_xmeans(train_loader, args, device,
                                         test_loader=test_loader)
    elif args.search_mode == 'gmeans':
        from methods.estimate_k.estimate_k_xgmeans import estimate_k_gmeans
        _, _, result = estimate_k_gmeans(train_loader, args, device,
                                         test_loader=test_loader)
    elif args.search_mode == 'spectral':
        from methods.estimate_k.estimate_k_spectral import estimate_k_spectral
        _, _, result = estimate_k_spectral(train_loader, args, device,
                                           test_loader=test_loader)
    elif args.search_mode == 'hdbscan':
        from methods.estimate_k.estimate_k_hdbscan import estimate_k_hdbscan
        _, _, result = estimate_k_hdbscan(train_loader, args, device,
                                          test_loader=test_loader)
    elif args.search_mode == 'brent_tuned':
        from methods.estimate_k.estimate_k_brent_tuned import estimate_k_brent_tuned
        _, _, result = estimate_k_brent_tuned(train_loader, args, device,
                                              test_loader=test_loader)
    else:
        _, _, result = binary_search(train_loader, args)

    # Report the PCA dimensionality actually used so the parent pipeline can replicate it
    _pca_used = result.get('pca')
    _pca_dim_used = int(_pca_used.n_components_) if _pca_used is not None else 0
    print(f'Selected PCA dim: {_pca_dim_used}')

    if args.evaluate:
        run_evaluation(result, test_loader, args, device)