"""
run_pipeline_hier.py — Hierarchical category discovery pipeline.

Extends the standard pipeline (run_pipeline.py) with coarse-label support:
  1. Feature extraction (pretrained backbone)              [same as run_pipeline.py]
  2a. Per-coarse K estimation                             [new — if --estimate_k_per_coarse]
  2b. Evaluation of pretrained model clustering           [same]
  3. Hierarchical training with coarse-label constraints  [new dispatch: uno_hier]
  3b. Feature extraction (trained backbone)               [same]
  3c. Evaluation of trained model clustering              [same]

Key assumptions:
  - Every coarse category is either fully known or fully unknown (no mixed coarses).
  - Coarse labels are provided as a JSON mapping: {fine_class_idx: coarse_class_idx}.

Run from the SelEx/ directory:
    python run_pipeline_hier.py \
        --dataset_name scars \
        --training_method uno_hier \
        --coarse_labels_path data/coarse_labels/scars.json \
        --use_oracle_k True \
        --epochs 5 --temperature 0.1
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from config import feature_extract_dir
from data.data_utils import build_coarse_of, load_coarse_map, oracle_k_per_coarse, verify_coarse_constraint
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from project_utils.cluster_and_log_utils import split_ami_score, split_cluster_acc_v2
from project_utils.general_utils import str2bool


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run(cmd, capture_output=False):
    print(f"\n>>> {' '.join(cmd)}\n")
    if capture_output:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed (exit {result.returncode})")
        return result.stdout
    else:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed (exit {result.returncode})")
        return None


# ---------------------------------------------------------------------------
# Step 1 – Extract features  (identical to run_pipeline.py)
# ---------------------------------------------------------------------------

def extract_features(args, warmup_model_dir=None, root_dir=None):
    cmd = [
        args.python, '-m', 'methods.clustering.extract_features',
        '--dataset', args.dataset_name,
        '--model_name', args.model_name,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--silent', 'False',
    ]
    if warmup_model_dir is not None:
        cmd += ['--warmup_model_dir', warmup_model_dir]
    if root_dir is not None:
        cmd += ['--root_dir', root_dir]
    if args.num_novel_classes is not None:
        cmd += ['--num_novel_classes', str(args.num_novel_classes)]
    _run(cmd)


# ---------------------------------------------------------------------------
# Step 2a – Per-coarse K estimation  (new)
# ---------------------------------------------------------------------------

def estimate_k_per_coarse(args, transformer_features_path=None):
    """
    Run K estimation as a subprocess.
    For --estimate_k_method transformer, dispatches to estimate_k_hier_transformer.py.
    Otherwise dispatches to estimate_k_hier.py.
    Returns the path to the k_results JSON (same as args.k_results_path).
    transformer_features_path: if set and method is 'transformer', also saves
      transformer-transformed features to this .npz path (Mode B).
    """
    # --- shared args for the new tuning/brent modules ---
    _new_shared = [
        '--dataset_name',      args.dataset_name,
        '--model_name',        args.model_name,
        '--coarse_labels_path', args.coarse_labels_path,
        '--k_results_path',    args.k_results_path,
        '--batch_size',        str(args.batch_size),
        '--num_workers',       str(args.num_workers),
        '--prop_train_labels', str(args.prop_train_labels),
        '--use_ssb_splits',    args.use_ssb_splits,
        '--cluster_methods',   args.cluster_methods,
    ] + (['--hier_split_type', args.hier_split_type]
         if args.hier_split_type is not None else []) \
      + (['--num_novel_classes', str(args.num_novel_classes)]
         if args.num_novel_classes is not None else [])

    if args.estimate_k_method == 'transformer':
        cmd = [
            args.python, '-m', 'methods.estimate_k_hier.estimate_k_hier_transformer',
            '--dataset_name',      args.dataset_name,
            '--model_name',        args.model_name,
            '--coarse_labels_path', args.coarse_labels_path,
            '--k_results_path',    args.k_results_path,
            '--batch_size',        str(args.batch_size),
            '--num_workers',       str(args.num_workers),
            '--prop_train_labels',  str(args.prop_train_labels),
            '--use_oracle_k',      str(args.use_oracle_k),
            '--n_models',          str(args.transformer_n_models),
            '--epochs',            str(args.transformer_epochs),
            '--transformer_batch_size', str(args.transformer_batch_size),
            '--d_model',           str(args.transformer_d_model),
            '--bottleneck_dim',    str(args.transformer_bottleneck_dim),
            '--lr',                str(args.transformer_lr),
            '--val_prop',          str(args.transformer_val_prop),
            '--device',            args.transformer_device,
            '--cluster_methods',   args.cluster_methods,
        ] + (['--num_novel_classes', str(args.num_novel_classes)]
             if args.num_novel_classes is not None else []) \
          + (['--use_ssb_splits', args.use_ssb_splits]) \
          + (['--ckpt_dir', args.transformer_ckpt_dir]
             if args.transformer_ckpt_dir is not None else []) \
          + (['--save_transformer_features_path', transformer_features_path]
             if transformer_features_path is not None else [])

    elif args.estimate_k_method.startswith('tuning_'):
        sub_method = args.estimate_k_method[len('tuning_'):]  # 'agglo', 'spectral', 'xmeans'
        cmd = [
            args.python, '-m', 'methods.estimate_k_hier.estimate_k_hier_tuning',
            '--estimate_k_method', sub_method,
            '--agglo_n_grid',           str(args.agglo_n_grid),
            '--agglo_lo',               str(args.agglo_lo),
            '--agglo_hi',               str(args.tuning_agglo_hi),
            '--spectral_gap_n_grid',    str(args.spectral_gap_n_grid),
            '--spectral_gap_lo',        str(args.spectral_gap_lo),
            '--spectral_gap_hi',        str(args.spectral_gap_hi),
            '--spectral_n_neighbors',   str(args.spectral_n_neighbors),
            '--pca_dim_candidates',     args.pca_dim_candidates,
        ] + _new_shared

    elif args.estimate_k_method.startswith('brent_hier_'):
        cluster_method = args.estimate_k_method[len('brent_hier_'):]  # 'kmeans', 'agglo', etc.
        cmd = [
            args.python, '-m', 'methods.estimate_k_hier.estimate_k_hier_brent',
            '--cluster_method',  cluster_method,
            '--n_probe_total',   str(args.n_probe_total),
            '--n_probe_related', str(args.n_probe_related),
        ] + _new_shared

    else:
        cmd = [
            args.python, '-m', 'methods.estimate_k_hier.estimate_k_hier',
            '--dataset_name',      args.dataset_name,
            '--model_name',        args.model_name,
            '--coarse_labels_path', args.coarse_labels_path,
            '--k_results_path',    args.k_results_path,
            '--search_mode',       args.estimate_k_method,
            '--max_classes',       str(args.max_classes),
            '--batch_size',        str(args.batch_size),
            '--num_workers',       str(args.num_workers),
            '--probe_coarse_prop', str(args.probe_coarse_prop),
            '--probe_prop',        str(args.probe_prop),
            '--val_prop',          str(args.val_prop),
            '--outlier_tau',       str(args.outlier_tau),
            '--pca_dim',           str(args.pca_dim),
            '--agglo_linkages',    args.agglo_linkages,
            '--agglo_n_grid',      str(args.agglo_n_grid),
            '--agglo_lo',          str(args.agglo_lo),
            '--agglo_hi',          str(args.agglo_hi),
            '--agglo_max_samples', str(args.agglo_max_samples),
            '--spectral_n_neighbors', str(args.spectral_n_neighbors),
            '--final_cluster_method', args.final_cluster_method,
            '--use_oracle_k',     str(args.use_oracle_k),
        ] + (['--agglo_use_combined'] if args.agglo_use_combined else []) \
          + (['--num_novel_classes', str(args.num_novel_classes)]
             if args.num_novel_classes is not None else []) \
          + (['--use_ssb_splits', args.use_ssb_splits])

    output = _run(cmd, capture_output=True)

    match = re.search(r'K estimation complete:\s*(\S+)', output)
    if match is None:
        raise RuntimeError(
            "Could not parse 'K estimation complete: X' from estimate_k_hier output.\n"
            f"Output was:\n{output}"
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# Step 2b / 3c – Clustering evaluation  (identical to run_pipeline.py)
# ---------------------------------------------------------------------------

def evaluate_features(args, feature_root, estimated_k_total):
    """
    Load pre-extracted features, run KMeans, and return clustering metrics for
    both the held-out test split and the unlabelled training split.

    Returns dict with keys 'test' and 'train_unlabelled', each containing
    {'acc': (all, old, new), 'ami': (all, old, new)}.
    """
    _, test_dataset, unlabelled_train_dataset, _ = get_datasets(
        args.dataset_name, train_transform=None, test_transform=None, args=args
    )

    results = {}

    for split_name, base_dataset, feature_subdir in [
        ('test',             test_dataset,             'test'),
        ('train_unlabelled', unlabelled_train_dataset, 'train'),
    ]:
        feature_dir = os.path.join(feature_root, feature_subdir)
        fv_dataset = FeatureVectorDataset(base_dataset=base_dataset, feature_root=feature_dir)
        loader = DataLoader(
            fv_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
        )
        all_feats, targets, mask = [], np.array([]), np.array([])
        print(f'Collecting {split_name} features...')
        for feats, label, _uq in loader:
            if isinstance(feats, torch.Tensor):
                feats = feats.numpy()
            all_feats.append(feats)
            targets = np.append(targets, label.numpy())
            mask = np.append(
                mask,
                np.array([x.item() in range(len(args.train_classes)) for x in label])
            )
        all_feats = np.concatenate(all_feats)
        mask = mask.astype(bool)
        print(f'Fitting KMeans (K={estimated_k_total}) on {split_name}...')
        kmeans = KMeans(n_clusters=estimated_k_total, random_state=0).fit(all_feats)
        preds  = kmeans.labels_
        acc_all, acc_old, acc_new = split_cluster_acc_v2(targets, preds, mask)
        ami_all, ami_old, ami_new = split_ami_score(targets, preds, mask)
        results[split_name] = {
            'acc': (acc_all, acc_old, acc_new),
            'ami': (ami_all, ami_old, ami_new),
        }

    return results


# ---------------------------------------------------------------------------
# Step 2b (Mode B) – Clustering evaluation on transformer features
# ---------------------------------------------------------------------------

def evaluate_transformer_features(args, transformer_features_path, estimated_k_total):
    """
    Load transformer-transformed features from .npz (produced by
    estimate_k_hier_transformer.py --save_transformer_features_path), run KMeans,
    and return clustering metrics in the same format as evaluate_features().

    Uses test_transformer_feats and unlab_transformer_feats from the .npz.
    """
    data = np.load(transformer_features_path)
    results = {}
    for split_name, feats_key, labels_key, mask_key in [
        ('test',             'test_transformer_feats',  'test_labels',  'test_mask'),
        ('train_unlabelled', 'unlab_transformer_feats', 'unlab_labels', 'unlab_mask'),
    ]:
        feats = data[feats_key].astype(np.float32)
        labels = data[labels_key].astype(int)
        mask = data[mask_key].astype(bool)
        print(f'Fitting KMeans (K={estimated_k_total}) on {split_name} '
              f'(transformer features dim={feats.shape[1]}) ...')
        kmeans = KMeans(n_clusters=estimated_k_total, random_state=0).fit(feats)
        preds = kmeans.labels_
        acc_all, acc_old, acc_new = split_cluster_acc_v2(labels, preds, mask)
        ami_all, ami_old, ami_new = split_ami_score(labels, preds, mask)
        results[split_name] = {
            'acc': (acc_all, acc_old, acc_new),
            'ami': (ami_all, ami_old, ami_new),
        }
    return results


# ---------------------------------------------------------------------------
# Step 3 – Hierarchical training
# ---------------------------------------------------------------------------

def run_training(args, k_results_path):
    """
    Dispatch to the chosen hierarchical training method.
    Parses 'Experiment saved to: X' and returns log_dir.
    """
    shared = [
        '--dataset_name',      args.dataset_name,
        '--batch_size',        str(args.train_batch_size),
        '--grad_from_block',   str(args.grad_from_block),
        '--epochs',            str(args.epochs),
        '--base_model',        args.model_name,
        '--num_workers',       str(args.num_workers),
        '--use_ssb_splits',    args.use_ssb_splits,
        '--weight_decay',      str(args.weight_decay),
        '--transform',         args.transform,
        '--lr',                str(args.lr),
        '--eval_funcs',        *args.eval_funcs,
        '--coarse_labels_path', args.coarse_labels_path,
        '--k_results_path',    k_results_path,
    ]
    if args.num_novel_classes is not None:
        shared += ['--num_novel_classes', str(args.num_novel_classes)]
    if args.hier_split_type is not None:
        shared += ['--hier_split_type', args.hier_split_type]

    if args.training_method == 'uno_hier':
        cmd = [args.python, '-m', 'methods.uno_hier_training.uno_hier_training'] + shared + [
            '--temperature',       str(args.temperature),
            '--num_heads',         str(args.num_heads),
            '--overcluster_factor', str(args.overcluster_factor),
            '--proj_dim',          str(args.proj_dim),
            '--hidden_dim',        str(args.hidden_dim),
            '--epsilon_sk',        str(args.epsilon_sk),
        ]
    elif args.training_method == 'simgcd_hier':
        cmd = [args.python, '-m', 'methods.simgcd_hier_training.simgcd_hier_training'] + shared + [
            '--sup_weight',                  str(args.sup_weight),
            '--memax_weight',                str(args.memax_weight),
            '--warmup_teacher_temp',         str(args.warmup_teacher_temp),
            '--teacher_temp',                str(args.teacher_temp),
            '--warmup_teacher_temp_epochs',  str(args.warmup_teacher_temp_epochs),
        ]
    elif args.training_method == 'gcd_hier':
        cmd = [args.python, '-m', 'methods.gcd_hier_training.gcd_hier_training'] + shared + [
            '--sup_con_weight',        str(args.sup_con_weight),
            '--temperature',           str(args.temperature),
            '--contrast_unlabel_only', str(args.contrast_unlabel_only),
        ]
    elif args.training_method == 'gcd_selex_hier':
        cmd = [args.python, '-m', 'methods.contrastive_hier_training.contrastive_hier_training'] + shared + [
            '--sup_con_weight',                str(args.sup_con_weight),
            '--temperature',                   str(args.temperature),
            '--contrast_unlabel_only',         str(args.contrast_unlabel_only),
            '--strategy',                      args.strategy,
            '--cluster_momentum',              str(args.cluster_momentum),
            '--unsupervised_smoothing',        str(args.unsupervised_smoothing),
            '--distance',                      args.distance,
            '--prototype_extraction_interval', str(args.prototype_extraction_interval),
            '--train_report_interval',         str(args.train_report_interval),
            '--unbalanced',                    str(args.unbalanced),
            '--n_views',                       str(args.n_views),
        ]
    elif args.training_method == 'gcd_selex_coarse':
        # Coarse-anchor SelEx: GT coarse labels anchor the hierarchy; no k_results_path needed.
        cmd = [args.python, '-m',
               'methods.contrastive_hier_training.contrastive_coarse_training'] + [
            '--dataset_name',      args.dataset_name,
            '--batch_size',        str(args.train_batch_size),
            '--grad_from_block',   str(args.grad_from_block),
            '--epochs',            str(args.epochs),
            '--base_model',        args.model_name,
            '--num_workers',       str(args.num_workers),
            '--use_ssb_splits',    args.use_ssb_splits,
            '--weight_decay',      str(args.weight_decay),
            '--transform',         args.transform,
            '--lr',                str(args.lr),
            '--eval_funcs',        *args.eval_funcs,
            '--coarse_labels_path', args.coarse_labels_path,
            '--sup_con_weight',                str(args.sup_con_weight),
            '--temperature',                   str(args.temperature),
            '--contrast_unlabel_only',         str(args.contrast_unlabel_only),
            '--strategy',                      args.strategy,
            '--cluster_momentum',              str(args.cluster_momentum),
            '--unsupervised_smoothing',        str(args.unsupervised_smoothing),
            '--distance',                      args.distance,
            '--prototype_extraction_interval', str(args.prototype_extraction_interval),
            '--train_report_interval',         str(args.train_report_interval),
            '--unbalanced',                    str(args.unbalanced),
            '--n_views',                       str(args.n_views),
        ] + (
            ['--num_novel_classes', str(args.num_novel_classes)]
            if args.num_novel_classes is not None else []
        ) + (
            ['--hier_split_type', args.hier_split_type]
            if args.hier_split_type is not None else []
        )
    else:
        raise ValueError(f"Unknown training_method: {args.training_method}")

    output = _run(cmd, capture_output=True)
    match  = re.search(r'Experiment saved to:\s*(\S+)', output)
    if match is None:
        raise RuntimeError(
            "Could not parse 'Experiment saved to: X' from training output.\n"
            f"Output was:\n{output}"
        )
    log_dir = match.group(1)
    print(f"Training log dir: {log_dir}")
    return log_dir


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _fmt(v):
    return f'{v:.4f}'


def print_results(args, total_k_unknown, pretrain_metrics, trained_metrics=None):
    width = 57
    sep = '=' * width
    print(f'\n{sep}')
    print('HIERARCHICAL PIPELINE RESULTS')
    print(sep)
    print(f'  Dataset:         {args.dataset_name}')
    print(f'  Training method: {args.training_method}')
    print(f'  Total unknown K: {total_k_unknown}')
    print()
    header = f'  {"":20s}  {"All":>7}  {"Old":>7}  {"New":>7}'
    print(header)
    print(f'  {"-"*20}  {"-------":>7}  {"-------":>7}  {"-------":>7}')

    def _row(label, triple):
        a, o, n = triple
        print(f'  {label:20s}  {_fmt(a):>7}  {_fmt(o):>7}  {_fmt(n):>7}')

    for split, slabel in [('test', 'Test'), ('train_unlabelled', 'Unlab Train')]:
        _row(f'Pretrain Acc [{slabel}]', pretrain_metrics[split]['acc'])
        _row(f'Pretrain AMI [{slabel}]', pretrain_metrics[split]['ami'])
    if trained_metrics is not None:
        for split, slabel in [('test', 'Test'), ('train_unlabelled', 'Unlab Train')]:
            _row(f'Trained  Acc [{slabel}]', trained_metrics[split]['acc'])
            _row(f'Trained  AMI [{slabel}]', trained_metrics[split]['ami'])
    print(sep)


def save_results_json(path, args, total_k_unknown, pretrain_metrics, trained_metrics=None):
    def _triple(t):
        return {'all': t[0], 'old': t[1], 'new': t[2]}
    data = {
        'dataset_name':   args.dataset_name,
        'training_method': args.training_method,
        'total_k_unknown': total_k_unknown,
        'pretrain': {
            split: {
                'acc': _triple(pretrain_metrics[split]['acc']),
                'ami': _triple(pretrain_metrics[split]['ami']),
            }
            for split in ('test', 'train_unlabelled')
        },
    }
    if trained_metrics is not None:
        data['trained'] = {
            split: {
                'acc': _triple(trained_metrics[split]['acc']),
                'ami': _triple(trained_metrics[split]['ami']),
            }
            for split in ('test', 'train_unlabelled')
        }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Results saved to {path}')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hierarchical category discovery pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset / model (same as run_pipeline.py)
    parser.add_argument('--dataset_name',      type=str,   default='scars',
                        help='Dataset name. Options: scars, cub, aircraft, inat_subspecies, aves_small, aves_full')
    parser.add_argument('--model_name',        type=str,   default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits',    type=str,   default='True')
    parser.add_argument('--hier_split_type',   type=str,   default=None,
                        choices=['5050', '6040', 'train_test', 'train_val'],
                        help='Use coarse-respecting split file instead of SSB/default. '
                             'Choices: 5050 (50/50 train/unlabeled), 6040 (60/40).')
    parser.add_argument('--num_novel_classes', type=int,   default=None)

    # Coarse labels (new)
    parser.add_argument('--coarse_labels_path', type=str, required=True,
                        help='JSON mapping fine_class_idx -> coarse_class_idx')
    parser.add_argument('--k_results_path',    type=str, default=None,
                        help='Path for per-coarse K JSON. Auto-generated if None.')
    parser.add_argument('--estimate_k_per_coarse', type=str2bool, default=False,
                        help='Run per-coarse K estimation before training')
    parser.add_argument('--use_oracle_k',      type=str2bool, default=False,
                        help='Use true fine-class counts per coarse as K (no estimation)')
    parser.add_argument('--probe_coarse_prop', type=float, default=1.0,
                        help='Fraction of known coarse categories used as probe pool')

    # K estimation method
    parser.add_argument('--estimate_k_method', type=str, default='autonovel',
                        choices=['brent', 'binary', 'autonovel', 'agglo',
                                 'xmeans', 'gmeans', 'spectral', 'transformer',
                                 'tuning_agglo', 'tuning_spectral', 'tuning_xmeans',
                                 'brent_hier_kmeans', 'brent_hier_agglo',
                                 'brent_hier_spectral', 'brent_hier_gmm_tied'])
    parser.add_argument('--max_classes',          type=int,   default=20)
    parser.add_argument('--probe_prop',           type=float, default=0.2)
    parser.add_argument('--val_prop',             type=float, default=0.2)
    parser.add_argument('--outlier_tau',          type=float, default=0.01)
    parser.add_argument('--pca_dim',              type=int,   default=0)
    parser.add_argument('--agglo_linkages',       type=str,   default='ward,complete,average')
    parser.add_argument('--agglo_n_grid',         type=int,   default=25)
    parser.add_argument('--agglo_lo',             type=float, default=0.0)
    parser.add_argument('--agglo_hi',             type=float, default=10.0)
    parser.add_argument('--agglo_use_combined',   action='store_true')
    parser.add_argument('--agglo_max_samples',    type=int,   default=5000)
    parser.add_argument('--spectral_n_neighbors', type=int,   default=10)
    parser.add_argument('--final_cluster_method', type=str,   default='auto',
                        choices=['auto', 'kmeans', 'spectral', 'gmm'])
    # New: shared eval methods for tuning/brent/transformer modules
    parser.add_argument('--cluster_methods', type=str,
                        default='kmeans,agglo,spectral,gmm_tied',
                        help='Comma-separated clustering methods for per-coarse eval')
    # New: tuning_agglo specific (separate hi default from existing agglo_hi=10.0)
    parser.add_argument('--tuning_agglo_hi', type=float, default=2.0,
                        help='(tuning_agglo) upper bound for distance threshold grid')
    # New: tuning_spectral specific
    parser.add_argument('--spectral_gap_n_grid',  type=int,   default=30)
    parser.add_argument('--spectral_gap_lo',      type=float, default=0.001)
    parser.add_argument('--spectral_gap_hi',      type=float, default=1.0)
    # New: tuning_xmeans specific
    parser.add_argument('--pca_dim_candidates',   type=str,   default='0,32,64,128',
                        help='(tuning_xmeans) comma-separated PCA dims to try')
    # New: brent_hier specific
    parser.add_argument('--n_probe_total',   type=int, default=4,
                        help='(brent_hier) target number of probe coarses per unknown coarse')
    parser.add_argument('--n_probe_related', type=int, default=2,
                        help='(brent_hier) max taxonomically related probe coarses')

    # Set Transformer K estimation hyperparameters (used when --estimate_k_method transformer)
    parser.add_argument('--use_transformer_features', type=str2bool, default=False,
                        help='(transformer) Mode B: use transformer-transformed features '
                             'for clustering evaluation instead of backbone features')
    parser.add_argument('--transformer_n_models',     type=int,   default=5,
                        help='(transformer) ensemble size')
    parser.add_argument('--transformer_epochs',       type=int,   default=50,
                        help='(transformer) training epochs per model')
    parser.add_argument('--transformer_batch_size',   type=int,   default=32,
                        help='(transformer) coarse groups per batch')
    parser.add_argument('--transformer_d_model',      type=int,   default=128,
                        help='(transformer) transformer hidden dimension')
    parser.add_argument('--transformer_bottleneck_dim', type=int, default=64,
                        help='(transformer) projection head output dimension '
                             '(also the feature dim used for clustering in Mode B)')
    parser.add_argument('--transformer_lr',           type=float, default=0.1,
                        help='(transformer) learning rate')
    parser.add_argument('--transformer_val_prop',     type=float, default=0.2,
                        help='(transformer) fraction of known coarses held out for validation')
    parser.add_argument('--transformer_device',       type=str,   default='cuda:0',
                        help='(transformer) device for transformer training/inference')
    parser.add_argument('--transformer_ckpt_dir',     type=str,   default=None,
                        help='(transformer) checkpoint directory; auto-generated if None')

    # Shared compute
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)

    # Training
    parser.add_argument('--skip_training', type=str2bool, default=True)
    parser.add_argument('--training_method', type=str, default='uno_hier',
                        choices=['uno_hier', 'simgcd_hier', 'gcd_hier',
                                 'gcd_selex_hier', 'gcd_selex_coarse'],
                        help='Hierarchical training method')
    parser.add_argument('--train_batch_size',  type=int,   default=32)
    parser.add_argument('--epochs',            type=int,   default=1)
    parser.add_argument('--grad_from_block',   type=int,   default=11)
    parser.add_argument('--lr',                type=float, default=0.1)
    parser.add_argument('--weight_decay',      type=float, default=5e-5)
    parser.add_argument('--transform',         type=str,   default='imagenet')
    parser.add_argument('--eval_funcs',        nargs='+',  default=['v2', 'ami'])
    # UNO-hier-specific hyperparams
    parser.add_argument('--temperature',       type=float, default=0.1,
                        help='(uno_hier) swapped-prediction loss temperature')
    parser.add_argument('--num_heads',         type=int,   default=5)
    parser.add_argument('--overcluster_factor', type=int,  default=3)
    parser.add_argument('--proj_dim',          type=int,   default=256)
    parser.add_argument('--hidden_dim',        type=int,   default=2048)
    parser.add_argument('--epsilon_sk',        type=float, default=0.05)
    # SimGCD-hier-specific hyperparams
    parser.add_argument('--sup_weight',        type=float, default=0.35,
                        help='(simgcd_hier) weight for supervised loss components')
    parser.add_argument('--memax_weight',      type=float, default=2.0,
                        help='(simgcd_hier) weight for MaxEnt regularisation')
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.07,
                        help='(simgcd_hier) initial teacher temperature')
    parser.add_argument('--teacher_temp',      type=float, default=0.04,
                        help='(simgcd_hier) final teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=30,
                        help='(simgcd_hier) epochs to warm up teacher temperature')
    # GCD-hier-specific hyperparams
    parser.add_argument('--sup_con_weight',        type=float,    default=0.5,
                        help='(gcd_hier/gcd_selex_hier) weight for supervised contrastive loss')
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False,
                        help='(gcd_hier/gcd_selex_hier) restrict NTXent to unlabeled samples only')
    # gcd_selex_hier-specific hyperparams
    parser.add_argument('--strategy',                      type=str,      default='zero_one',
                        choices=['zero_one', 'pair_dist', 'pair_cluster', 'none'],
                        help='(gcd_selex_hier) confusion factor strategy')
    parser.add_argument('--cluster_momentum',              type=float,    default=1.0,
                        help='(gcd_selex_hier) EMA momentum for cluster radii')
    parser.add_argument('--unsupervised_smoothing',        type=float,    default=1.0,
                        help='(gcd_selex_hier) weight for confusion-factor label smoothing')
    parser.add_argument('--distance',                      type=str,      default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help='(gcd_selex_hier) distance metric')
    parser.add_argument('--prototype_extraction_interval', type=int,      default=1,
                        help='(gcd_selex_hier) epochs between prototype re-extractions')
    parser.add_argument('--train_report_interval',         type=int,      default=200,
                        help='(gcd_selex_hier) epochs between full train-set evaluations')
    parser.add_argument('--unbalanced',                    type=str2bool, default=False,
                        help='(gcd_selex_hier) skip balanced cluster-size constraint in KMeans')
    parser.add_argument('--n_views',                       type=int,      default=2,
                        help='(gcd_selex_hier) number of augmentation views')

    # Output
    parser.add_argument('--output_file', type=str, default='results_hier.json')
    parser.add_argument('--python', type=str, default=sys.executable)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # Verify coarse constraint: no coarse category spans both splits
    coarse_map = load_coarse_map(args.coarse_labels_path)
    verify_coarse_constraint(args.train_classes, args.unlabeled_classes, coarse_map)
    print('Coarse constraint verified: no mixed coarses.')

    # Auto-generate k_results_path if not provided
    if args.k_results_path is None:
        args.k_results_path = os.path.join(
            feature_extract_dir, f'{args.model_name}_{args.dataset_name}',
            'k_per_coarse.json'
        )

    # Path for transformer-transformed features .npz (Mode B, only when applicable)
    _transformer_features_path = (
        os.path.splitext(os.path.abspath(args.k_results_path))[0] + '_transformer_features.npz'
        if args.estimate_k_method == 'transformer' and args.use_transformer_features
        else None
    )

    pretrain_feature_root = os.path.join(
        feature_extract_dir, f'{args.model_name}_{args.dataset_name}'
    )
    trained_feature_root = os.path.join(
        feature_extract_dir, 'trained', f'{args.model_name}_{args.dataset_name}'
    )

    # Determine total_k_unknown for evaluation (from k_results_path or oracle)
    def _get_total_k():
        if os.path.exists(args.k_results_path):
            with open(args.k_results_path) as f:
                k_per_coarse = json.load(f)
            return sum(int(v) for v in k_per_coarse.values())
        # Fall back to oracle
        coarse_map = load_coarse_map(args.coarse_labels_path)
        k_per_coarse = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)
        return sum(k_per_coarse.values())

    # ------------------------------------------------------------------
    # Step 1: Extract pretrained features
    # ------------------------------------------------------------------
    print('\n=== Step 1: Extract Features (pretrained backbone) ===')
    extract_features(args)

    # ------------------------------------------------------------------
    # Step 2a: Per-coarse K estimation (optional)
    # ------------------------------------------------------------------
    if args.estimate_k_per_coarse or args.use_oracle_k:
        print('\n=== Step 2a: Per-Coarse K Estimation ===')
        estimate_k_per_coarse(args, transformer_features_path=_transformer_features_path)
        print(f'K results saved to: {args.k_results_path}')

    total_k_unknown = _get_total_k()
    estimated_k_total = args.num_labeled_classes + total_k_unknown
    print(f'\nK summary: labeled={args.num_labeled_classes} | '
          f'unknown={total_k_unknown} | total={estimated_k_total}')

    # ------------------------------------------------------------------
    # Step 2b: Evaluate pretrained model clustering
    # ------------------------------------------------------------------
    print('\n=== Step 2b: Evaluate Pretrained Model Clustering ===')
    if (_transformer_features_path is not None
            and os.path.exists(_transformer_features_path)):
        print(f'  Using transformer-transformed features from {_transformer_features_path}')
        pretrain_metrics = evaluate_transformer_features(
            args, _transformer_features_path, estimated_k_total)
    else:
        pretrain_metrics = evaluate_features(args, pretrain_feature_root, estimated_k_total)
    for split, label in [('test', 'Test'), ('train_unlabelled', 'Train (Unlabelled)')]:
        print(f"  Pretrain Acc [{label}] — All: {pretrain_metrics[split]['acc'][0]:.4f} | "
              f"Old: {pretrain_metrics[split]['acc'][1]:.4f} | New: {pretrain_metrics[split]['acc'][2]:.4f}")
        print(f"  Pretrain AMI [{label}] — All: {pretrain_metrics[split]['ami'][0]:.4f} | "
              f"Old: {pretrain_metrics[split]['ami'][1]:.4f} | New: {pretrain_metrics[split]['ami'][2]:.4f}")

    trained_metrics = None

    if not args.skip_training:
        # --------------------------------------------------------------
        # Step 3: Hierarchical training
        # --------------------------------------------------------------
        print('\n=== Step 3: Hierarchical Training ===')
        log_dir = run_training(args, args.k_results_path)
        model_path = os.path.join(log_dir, 'checkpoints') + os.sep

        # --------------------------------------------------------------
        # Step 3b: Extract trained model features
        # --------------------------------------------------------------
        print('\n=== Step 3b: Extract Features (trained backbone) ===')
        trained_root_dir = os.path.join(feature_extract_dir, 'trained')
        extract_features(args, warmup_model_dir=model_path, root_dir=trained_root_dir)

        # --------------------------------------------------------------
        # Step 3c: Evaluate trained model clustering
        # --------------------------------------------------------------
        print('\n=== Step 3c: Evaluate Trained Model Clustering ===')
        trained_metrics = evaluate_features(args, trained_feature_root, estimated_k_total)
        for split, label in [('test', 'Test'), ('train_unlabelled', 'Train (Unlabelled)')]:
            print(f"  Trained Acc [{label}] — All: {trained_metrics[split]['acc'][0]:.4f} | "
                  f"Old: {trained_metrics[split]['acc'][1]:.4f} | New: {trained_metrics[split]['acc'][2]:.4f}")
            print(f"  Trained AMI [{label}] — All: {trained_metrics[split]['ami'][0]:.4f} | "
                  f"Old: {trained_metrics[split]['ami'][1]:.4f} | New: {trained_metrics[split]['ami'][2]:.4f}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print_results(args, total_k_unknown, pretrain_metrics, trained_metrics)

    if args.output_file:
        save_results_json(
            args.output_file, args, total_k_unknown,
            pretrain_metrics, trained_metrics,
        )


if __name__ == '__main__':
    main()
