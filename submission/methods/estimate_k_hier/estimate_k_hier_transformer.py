"""
estimate_k_hier_transformer.py — Set Transformer K estimation for hierarchical GCD.

Trains a SetTransformerForKAndSupCon ensemble on known coarse groups (where fine-class
labels are available for supervision), then predicts K for each unknown coarse group.

Key differences from heuristic K estimation methods (brent, autonovel, etc.):
  - Directly learns "what does a set of within-coarse features look like for K classes?"
  - Uses supervision from known coarses (CE loss on K + SupCon loss on token embeddings)
  - Ensemble of N independently trained models for robustness

Optionally (--save_transformer_features_path), saves transformer-transformed feature
arrays (train + test splits) to a .npz file so they can be used for clustering instead
of the original backbone features (Mode B).

Output format:
  - JSON: {str(coarse_id): k_i}  (same as all other estimate_k_hier methods)
  - Prints "K estimation complete: {k_results_path}" (parsed by run_pipeline_hier.py)

Run standalone:
    python -m methods.estimate_k_hier.estimate_k_hier_transformer \\
        --dataset_name scars \\
        --coarse_labels_path data/coarse_labels/scars.json \\
        --k_results_path /tmp/scars_k_transformer.json \\
        --n_models 5 --epochs 50
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import feature_extract_dir
from data.data_utils import build_coarse_of, load_coarse_map, oracle_k_per_coarse
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from methods.estimate_k_hier.estimate_k_hier import load_all_features
from methods.estimate_k_hier.model_components import (
    SetTransformerForKAndSupCon, predict_ensemble, predict_ensemble_per_model,
    load_ensemble, train_one_model,
)
from project_utils.general_utils import str2bool


# ---------------------------------------------------------------------------
# Build coarse-group sequences from loaded features
# ---------------------------------------------------------------------------

def build_coarse_sequences(all_feats, all_labels, all_mask, coarse_of):
    """
    Group features by coarse ID and build the 'seqs' list format expected by
    SetTransformerForKAndSupCon training and inference.

    For known coarses (all fine classes labeled): fine-class labels are available
    as supervision for SupCon + K prediction.
    For unknown coarses (all fine classes unlabeled): dummy fine-class label = 0
    (only used at inference time where labels are ignored).

    Args:
        all_feats:  [N, D] float32 — L2-normalized backbone features
        all_labels: [N] int64 — remapped class labels (0..num_classes-1)
        all_mask:   [N] bool — True = labeled/known-class sample
        coarse_of:  dict mapping remapped_label -> coarse_id

    Returns:
        known_seqs:   list of seq dicts for known coarses (for training)
        unknown_seqs: list of seq dicts for unknown coarses (for inference)
        Kmax:         max K_true across known coarses
        input_dim:    feature dimension D
    """
    # Map each sample to its coarse ID
    unique_labels = np.unique(all_labels)
    label_to_coarse = {int(lbl): int(coarse_of[lbl]) for lbl in unique_labels}

    known_labels = set(all_labels[all_mask].tolist())
    unknown_labels = set(all_labels[~all_mask].tolist())

    known_coarses, unknown_coarses = set(), set()
    for lbl in unique_labels:
        c = label_to_coarse[int(lbl)]
        if int(lbl) in known_labels:
            known_coarses.add(c)
        else:
            unknown_coarses.add(c)

    input_dim = int(all_feats.shape[1])
    known_seqs = []
    unknown_seqs = []
    Kmax = 1

    # --- Known coarse sequences (train/val for the transformer) ---
    for c in sorted(known_coarses):
        # Indices of all samples in this coarse group
        c_idx = np.where(
            np.array([label_to_coarse[int(lbl)] == c for lbl in all_labels], dtype=bool)
        )[0]
        if len(c_idx) == 0:
            continue

        c_feats = all_feats[c_idx].astype(np.float32)
        c_labels_global = all_labels[c_idx]

        # Remap fine-class labels to 0..K_c-1 within this coarse group
        unique_fine = sorted(set(c_labels_global.tolist()))
        fine_remap = {orig: i for i, orig in enumerate(unique_fine)}
        c_labels_local = np.array([fine_remap[int(lbl)] for lbl in c_labels_global], dtype=np.int64)
        k_true = len(unique_fine)
        Kmax = max(Kmax, k_true)

        known_seqs.append({
            "Xp": c_feats,
            "y": c_labels_local,
            "k_true": k_true,
            "split_id": int(c),
            "obs_ids": c_idx.astype(np.int64),
        })

    # --- Unknown coarse sequences (inference only) ---
    for c in sorted(unknown_coarses):
        c_idx = np.where(
            np.array([label_to_coarse[int(lbl)] == c for lbl in all_labels], dtype=bool)
        )[0]
        if len(c_idx) == 0:
            continue

        c_feats = all_feats[c_idx].astype(np.float32)
        # Labels unknown — use 0 as placeholder (not used during inference)
        c_labels_dummy = np.zeros(len(c_idx), dtype=np.int64)

        # k_true = actual fine-class count (used only for logging/metrics, not supervision)
        c_labels_global = all_labels[c_idx]
        k_true_actual = len(set(c_labels_global.tolist()))

        unknown_seqs.append({
            "Xp": c_feats,
            "y": c_labels_dummy,
            "k_true": k_true_actual,
            "split_id": int(c),
            "obs_ids": c_idx.astype(np.int64),
        })

    return known_seqs, unknown_seqs, Kmax, input_dim


def build_test_coarse_sequences(test_feats, test_labels, coarse_of):
    """
    Build coarse-group sequences for test/unlabelled features (Mode B).
    Labels are placeholders (fine-class identity unknown for unlabeled coarses).
    """
    label_to_coarse = {int(lbl): int(coarse_of[lbl])
                       for lbl in np.unique(test_labels)
                       if 0 <= int(lbl) < len(coarse_of)}

    coarses = set(label_to_coarse.values())
    seqs = []

    for c in sorted(coarses):
        c_idx = np.where(
            np.array([label_to_coarse.get(int(lbl), -1) == c for lbl in test_labels], dtype=bool)
        )[0]
        if len(c_idx) == 0:
            continue
        c_feats = test_feats[c_idx].astype(np.float32)
        c_labels_global = test_labels[c_idx]
        unique_fine = sorted(set(c_labels_global.tolist()))
        fine_remap = {orig: i for i, orig in enumerate(unique_fine)}
        c_labels_local = np.array([fine_remap[int(lbl)] for lbl in c_labels_global], dtype=np.int64)
        seqs.append({
            "Xp": c_feats,
            "y": c_labels_local,
            "k_true": len(unique_fine),
            "split_id": int(c),
            "obs_ids": c_idx.astype(np.int64),
        })

    return seqs


def load_test_features(dataset, feature_root, batch_size, num_workers, device):
    """
    Load pre-extracted features for a test/unlabelled dataset.
    Returns (feats [N,D] float32, labels [N] int64).
    """
    feat_dataset = FeatureVectorDataset(base_dataset=dataset,
                                        feature_root=feature_root)
    loader = DataLoader(feat_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    feats_list, labels_list = [], []
    for batch in loader:
        # batch is (feats, label, uq_idx) for non-MergedDataset
        feats, label = batch[0], batch[1]
        feats = F.normalize(feats.float().to(device), dim=-1)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(label.numpy() if isinstance(label, torch.Tensor) else np.array(label))
    return (np.concatenate(feats_list, axis=0),
            np.concatenate(labels_list, axis=0).astype(np.int64))


def reassemble_features(seqs_results, N, bottleneck_dim):
    """
    Reassemble per-coarse-group transformer token embeddings back into a
    flat [N, bottleneck_dim] array aligned with the original feature array.

    seqs_results: list of dicts from predict_ensemble, each with 'obs_ids' and 'token_embs'.
    N: total number of samples in the original feature array.
    """
    out = np.zeros((N, bottleneck_dim), dtype=np.float32)
    for r in seqs_results:
        obs_ids = r["obs_ids"]       # original positions in the flat array
        token_embs = r["token_embs"]  # [n, bottleneck_dim]
        out[obs_ids] = token_embs
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Set Transformer K estimation for hierarchical category discovery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / features
    parser.add_argument('--dataset_name',      type=str, default='scars')
    parser.add_argument('--model_name',        type=str, default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=1.0)
    parser.add_argument('--use_ssb_splits',    type=str2bool, default=True)
    parser.add_argument('--hier_split_type',   type=str, default=None)
    parser.add_argument('--num_novel_classes', type=int, default=None)
    parser.add_argument('--root_dir',          type=str, default=feature_extract_dir)
    parser.add_argument('--batch_size',        type=int, default=128)
    parser.add_argument('--num_workers',       type=int, default=4)
    # Coarse labels
    parser.add_argument('--coarse_labels_path', type=str, required=True)
    parser.add_argument('--k_results_path',     type=str, required=True)
    parser.add_argument('--use_oracle_k',       type=str2bool, default=False)
    # Transformer hyperparameters
    parser.add_argument('--n_models',           type=int,   default=5,
                        help='Number of models in the ensemble')
    parser.add_argument('--epochs',             type=int,   default=50)
    parser.add_argument('--transformer_batch_size', type=int, default=32,
                        help='Batch size (number of coarse groups per batch)')
    parser.add_argument('--max_tokens_per_split', type=int, default=1024,
                        help='Max tokens (images) per coarse group per epoch step')
    parser.add_argument('--d_model',            type=int,   default=128)
    parser.add_argument('--nhead',              type=int,   default=8)
    parser.add_argument('--num_layers',         type=int,   default=2)
    parser.add_argument('--ff_dim',             type=int,   default=256)
    parser.add_argument('--dropout',            type=float, default=0.2)
    parser.add_argument('--bottleneck_dim',     type=int,   default=64)
    parser.add_argument('--supcon_lambda',      type=float, default=0.3)
    parser.add_argument('--lr',                 type=float, default=0.1)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--weight_decay',       type=float, default=1e-4)
    parser.add_argument('--jitter_std',         type=float, default=0.02)
    parser.add_argument('--token_dropout_p',    type=float, default=0.05)
    parser.add_argument('--mixup_alpha',        type=float, default=0.2)
    parser.add_argument('--mixup_p',            type=float, default=0.5)
    parser.add_argument('--val_prop',           type=float, default=0.2,
                        help='Fraction of known coarses held out for val monitoring')
    parser.add_argument('--seed',               type=int,   default=1000)
    parser.add_argument('--device',             type=str,   default='cuda:0')
    # Checkpoints
    parser.add_argument('--ckpt_dir',           type=str,   default=None,
                        help='Dir to save transformer checkpoints. '
                             'Defaults to {k_results_path}_transformer_ckpts/')
    # Mode B: save transformer features
    parser.add_argument('--save_transformer_features_path', type=str, default=None,
                        help='If set, save transformer-transformed features (train+test) '
                             'to this .npz path. Used by run_pipeline_hier.py Mode B.')
    # Clustering evaluation
    parser.add_argument('--cluster_methods', type=str,
                        default='kmeans,agglo,spectral,gmm_tied',
                        help='Comma-separated clustering methods for evaluation')

    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)
    # Compute oracle K (int keys) — used by both oracle and non-oracle paths for eval
    oracle_k_int = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)

    # ---- Oracle mode ----
    if args.use_oracle_k:
        k_per_coarse = {str(k): int(v) for k, v in oracle_k_int.items()}
        print(f"Oracle K per coarse: {k_per_coarse}")
        os.makedirs(os.path.dirname(os.path.abspath(args.k_results_path)), exist_ok=True)
        with open(args.k_results_path, 'w') as f:
            json.dump(k_per_coarse, f, indent=2)

        # Load features for clustering evaluation
        save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
        print(f'Loading features for clustering evaluation from {save_dir} ...')
        train_dataset, _, _, _ = get_datasets(
            args.dataset_name, train_transform=None, test_transform=None, args=args
        )
        all_feats, all_labels, all_mask = load_all_features(
            train_dataset, save_dir, args.batch_size, args.num_workers, device
        )
        _run_clustering_eval(args, all_feats, all_labels, coarse_of,
                             k_per_coarse, oracle_k_int,
                             transformer_feats=None,
                             results=None, models=None, unknown_seqs=None)
        print(f'\nK estimation complete: {args.k_results_path}')
        return

    # ---- Load backbone features ----
    save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')
    print(f'Loading features from {save_dir} ...')
    train_dataset, test_dataset, unlabelled_train_dataset, _ = get_datasets(
        args.dataset_name, train_transform=None, test_transform=None, args=args
    )
    all_feats, all_labels, all_mask = load_all_features(
        train_dataset, save_dir, args.batch_size, args.num_workers, device
    )
    print(f'Loaded {len(all_feats)} train feature vectors (dim={all_feats.shape[1]})')
    print(f'  Labeled: {all_mask.sum()} | Unlabeled: {(~all_mask).sum()}')

    # ---- Build coarse-group sequences ----
    known_seqs, unknown_seqs, Kmax, input_dim = build_coarse_sequences(
        all_feats, all_labels, all_mask, coarse_of
    )
    print(f'Known coarses: {len(known_seqs)} | Unknown coarses: {len(unknown_seqs)}')
    print(f'Kmax across known coarses: {Kmax} | Feature dim: {input_dim}')

    # Train/val split of known coarses (by coarse group, not by sample)
    n_val = max(1, round(args.val_prop * len(known_seqs)))
    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(known_seqs)).tolist()
    val_seqs   = [known_seqs[i] for i in perm[:n_val]]
    train_seqs = [known_seqs[i] for i in perm[n_val:]]
    print(f'Train coarses: {len(train_seqs)} | Val coarses: {len(val_seqs)}')

    if len(train_seqs) == 0:
        raise RuntimeError(
            'No training sequences available. Need at least 1 known coarse for training. '
            'Check that some coarse categories have all fine classes labeled.'
        )

    # ---- Set up checkpoint directory ----
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        base = os.path.splitext(os.path.abspath(args.k_results_path))[0]
        ckpt_dir = base + '_transformer_ckpts'
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Train ensemble ----
    print(f'\nTraining {args.n_models} Set Transformer model(s) for {args.epochs} epochs ...')
    model_dirs = []
    for m in range(args.n_models):
        mdir = train_one_model(
            model_id=m,
            input_dim=input_dim,
            Kmax_train=Kmax,
            train_seqs=train_seqs,
            val_seqs=val_seqs,
            ckpt_dir=ckpt_dir,
            epochs=args.epochs,
            batch_size=args.transformer_batch_size,
            max_tokens_per_split=args.max_tokens_per_split,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            supcon_lambda=args.supcon_lambda,
            jitter_std=args.jitter_std,
            token_dropout_p=args.token_dropout_p,
            mixup_alpha=args.mixup_alpha,
            mixup_p=args.mixup_p,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            device=device,
            seed=args.seed,
        )
        model_dirs.append(mdir)

    # ---- Load ensemble (best EMA checkpoints) ----
    models = load_ensemble(
        model_dirs=model_dirs,
        which='best_ema',
        input_dim=input_dim,
        Kmax_train=Kmax,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        device=device,
    )
    print(f'Loaded {len(models)} models for ensemble inference.')

    # ---- Predict K for unknown coarses ----
    print('\nRunning ensemble inference on unknown coarses ...')
    results = predict_ensemble(
        models=models,
        seqs=unknown_seqs,
        device=device,
        batch_size=args.transformer_batch_size,
        max_tokens_per_split=None,  # use all tokens at inference
        seed=args.seed,
    )

    k_per_coarse = {}
    for r in results:
        sid = str(r["split_id"])
        k_hat = max(1, int(r["k_hat"]))
        k_true = int(r["k_true"])
        p_split = r["p_split"]
        print(f"  Coarse {sid}: k_hat={k_hat}  k_true={k_true}  p_split={p_split:.3f}")
        k_per_coarse[sid] = k_hat

    # Log K-accuracy on validation coarses
    print('\nRunning inference on val coarses (K-accuracy check) ...')
    val_results = predict_ensemble(
        models=models, seqs=val_seqs, device=device,
        batch_size=args.transformer_batch_size,
        max_tokens_per_split=None, seed=args.seed,
    )
    k_hats = np.array([r["k_hat"] for r in val_results])
    k_trues = np.array([r["k_true"] for r in val_results])
    acc_exact = float(np.mean(k_hats == k_trues))
    acc_pm1 = float(np.mean(np.abs(k_hats - k_trues) <= 1))
    print(f'Val K-accuracy: exact={acc_exact:.3f}  ±1={acc_pm1:.3f} (n={len(val_results)})')

    # ---- Save K results JSON ----
    os.makedirs(os.path.dirname(os.path.abspath(args.k_results_path)), exist_ok=True)
    with open(args.k_results_path, 'w') as f:
        json.dump(k_per_coarse, f, indent=2)

    total_k = sum(k_per_coarse.values())
    print(f'\nPer-coarse K: {k_per_coarse}')
    print(f'Total unknown clusters: {total_k}')

    # ---- Mode B: save transformer features ----
    # (Run before clustering eval so transformer_feats are available)
    if args.save_transformer_features_path is not None:
        print(f'\nMode B: saving transformer features to {args.save_transformer_features_path} ...')
        N_train = len(all_feats)

        # --- Train transformer features ---
        all_seqs = known_seqs + unknown_seqs
        train_infer = predict_ensemble(
            models=models, seqs=all_seqs, device=device,
            batch_size=args.transformer_batch_size,
            max_tokens_per_split=None, seed=args.seed,
        )
        train_transformer_feats = reassemble_features(train_infer, N_train, args.bottleneck_dim)
        print(f'  Train transformer features: {train_transformer_feats.shape}')

        # --- Test features → transformer features ---
        test_feats, test_labels = load_test_features(
            test_dataset,
            os.path.join(save_dir, 'test'),
            args.batch_size, args.num_workers, device,
        )
        test_seqs = build_test_coarse_sequences(test_feats, test_labels, coarse_of)
        test_infer = predict_ensemble(
            models=models, seqs=test_seqs, device=device,
            batch_size=args.transformer_batch_size,
            max_tokens_per_split=None, seed=args.seed,
        )
        test_transformer_feats = reassemble_features(test_infer, len(test_feats), args.bottleneck_dim)
        test_mask = (test_labels < len(args.train_classes)).astype(bool)
        print(f'  Test transformer features: {test_transformer_feats.shape}')

        # --- Unlabelled train features → transformer features ---
        unlab_feats, unlab_labels = load_test_features(
            unlabelled_train_dataset,
            os.path.join(save_dir, 'train'),
            args.batch_size, args.num_workers, device,
        )
        unlab_seqs = build_test_coarse_sequences(unlab_feats, unlab_labels, coarse_of)
        unlab_infer = predict_ensemble(
            models=models, seqs=unlab_seqs, device=device,
            batch_size=args.transformer_batch_size,
            max_tokens_per_split=None, seed=args.seed,
        )
        unlab_transformer_feats = reassemble_features(unlab_infer, len(unlab_feats), args.bottleneck_dim)
        unlab_mask = (unlab_labels < len(args.train_classes)).astype(bool)
        print(f'  Unlabelled train transformer features: {unlab_transformer_feats.shape}')

        os.makedirs(os.path.dirname(os.path.abspath(args.save_transformer_features_path)), exist_ok=True)
        np.savez(
            args.save_transformer_features_path,
            train_feats=all_feats,
            train_labels=all_labels,
            train_mask=all_mask,
            train_transformer_feats=train_transformer_feats,
            test_feats=test_feats,
            test_labels=test_labels,
            test_mask=test_mask,
            test_transformer_feats=test_transformer_feats,
            unlab_feats=unlab_feats,
            unlab_labels=unlab_labels,
            unlab_mask=unlab_mask,
            unlab_transformer_feats=unlab_transformer_feats,
        )
        print(f'  Saved to {args.save_transformer_features_path}')

    # ---- Clustering evaluation ----
    # Extract per-sample transformer bottleneck features from K-prediction results.
    # obs_ids are positions in the flat all_feats array; unknown-coarse positions are
    # filled; known-coarse positions stay zero — harmless because eval_per_coarse only
    # iterates coarses present in k_per_coarse (unknown coarses only).
    transformer_feats = reassemble_features(results, len(all_feats), args.bottleneck_dim)

    # Mode B provides a full-coarse NPZ (known + unknown); prefer it when available.
    save_path = getattr(args, 'save_transformer_features_path', None)
    if save_path is not None and os.path.exists(save_path):
        npz = np.load(save_path)
        key = 'train_transformer_feats' if 'train_transformer_feats' in npz else list(npz.keys())[0]
        transformer_feats = npz[key].astype(np.float32)

    _run_clustering_eval(args, all_feats, all_labels, coarse_of,
                         k_per_coarse, oracle_k_int,
                         transformer_feats=transformer_feats,
                         results=results,
                         models=models,
                         unknown_seqs=unknown_seqs)

    print(f'\nK estimation complete: {args.k_results_path}')


# ---------------------------------------------------------------------------
# Clustering evaluation helper
# ---------------------------------------------------------------------------

def _eval_dist_avg(per_model_results, all_labels, k_per_coarse, oracle_k):
    """
    Average pairwise cosine-distance matrices across all models per coarse, then
    cluster with agglomerative (average linkage) and spectral (affinity precomputed).

    Returns {int(coarse_id): {dist_avg_agglo_acc, dist_avg_agglo_ami,
                               dist_avg_spectral_acc, dist_avg_spectral_ami}}
    """
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import adjusted_mutual_info_score
    from project_utils.cluster_utils import cluster_acc as _cacc

    results = {}
    for r in per_model_results:
        sid = str(r["split_id"])
        if sid not in k_per_coarse:
            continue
        c = int(sid)
        oids = r["obs_ids"]
        embs = r["per_model_embs"]   # list of [n, D] L2-normed arrays
        n = len(oids)
        k_hat = max(1, int(k_per_coarse[sid]))
        k_true = oracle_k.get(c, int(len(np.unique(all_labels[oids]))))
        entry = {"k_true": k_true, "k_hat": k_hat}

        if n < 2 or k_hat < 2:
            for key in ("dist_avg_agglo_acc", "dist_avg_agglo_ami",
                        "dist_avg_spectral_acc", "dist_avg_spectral_ami"):
                entry[key] = float("nan")
            results[c] = entry
            continue

        # Average cosine distance matrices (features are already L2-normalised)
        dist_mat = np.zeros((n, n), dtype=np.float64)
        for emb in embs:
            sim = emb.astype(np.float64) @ emb.astype(np.float64).T
            dist_mat += 1.0 - np.clip(sim, -1.0, 1.0)
        dist_mat /= max(1, len(embs))
        np.fill_diagonal(dist_mat, 0.0)
        dist_mat = np.clip(dist_mat, 0.0, None)

        k_eff = min(k_hat, n)
        true_labels = all_labels[oids]

        for method_key, do_fn in [
            ("dist_avg_agglo",
             lambda d, k: AgglomerativeClustering(
                 n_clusters=k, metric="precomputed", linkage="average"
             ).fit_predict(d)),
            ("dist_avg_spectral",
             lambda d, k: SpectralClustering(
                 n_clusters=k, affinity="precomputed",
                 assign_labels="cluster_qr", random_state=0,
             ).fit_predict(np.clip(1.0 - d / max(d.max(), 1e-8), 0.0, 1.0))),
        ]:
            try:
                preds = do_fn(dist_mat, k_eff)
                entry[f"{method_key}_acc"] = float(_cacc(
                    true_labels.astype(int), preds.astype(int)))
                entry[f"{method_key}_ami"] = float(adjusted_mutual_info_score(
                    true_labels, preds))
            except Exception as exc:
                print(f"  WARNING {method_key} coarse {sid}: {exc}")
                entry[f"{method_key}_acc"] = float("nan")
                entry[f"{method_key}_ami"] = float("nan")

        results[c] = entry
    return results


def _run_clustering_eval(args, all_feats, all_labels, coarse_of,
                         k_per_coarse, oracle_k_int,
                         transformer_feats=None,
                         results=None,
                         models=None,
                         unknown_seqs=None):
    """
    Run per-coarse clustering + evaluation; save _metrics.json.

    Extra arguments vs. original:
      results       : list of dicts from predict_ensemble (carries p_split per coarse)
      models        : loaded ensemble (used to run predict_ensemble_per_model for dist_avg)
      unknown_seqs  : coarse-group sequence dicts for the unknown coarses
    """
    from methods.estimate_k_hier.hier_eval_utils import (
        eval_per_coarse, aggregate_metrics, save_hier_metrics, print_aggregate_summary,
        compute_precision_curve, print_precision_summary,
    )
    from sklearn.preprocessing import normalize

    oracle_k = {int(k): int(v) for k, v in oracle_k_int.items()}

    cluster_methods_str = getattr(args, 'cluster_methods', 'kmeans,agglo,spectral,gmm_tied')
    cluster_methods = [m.strip() for m in cluster_methods_str.split(',')]

    feats_normed = normalize(all_feats, norm='l2', axis=1).astype(np.float32)

    alt_feats = None
    if transformer_feats is not None:
        alt_feats = normalize(transformer_feats, norm='l2', axis=1).astype(np.float32)

    print('\nRunning per-coarse clustering evaluation...')
    per_coarse = eval_per_coarse(
        feats_normed, all_labels, coarse_of, k_per_coarse, oracle_k,
        methods=cluster_methods, alt_feats=alt_feats, alt_name='transformer',
    )

    # ---- dist_avg strategy ----
    if models is not None and unknown_seqs is not None:
        device = next(models[0].parameters()).device
        transformer_batch_size = getattr(args, 'transformer_batch_size', 32)
        seed = getattr(args, 'seed', 0)
        print('Running per-model inference for dist_avg strategy...')
        per_model_results = predict_ensemble_per_model(
            models=models, seqs=unknown_seqs, device=device,
            batch_size=transformer_batch_size, seed=seed,
        )
        dist_results = _eval_dist_avg(per_model_results, all_labels, k_per_coarse, oracle_k)
        for c, d in dist_results.items():
            if c in per_coarse:
                per_coarse[c].update(
                    {k: v for k, v in d.items() if k not in ('k_true', 'k_hat')}
                )
    else:
        per_model_results = None

    agg = aggregate_metrics(per_coarse, methods=cluster_methods,
                            alt_name='transformer' if alt_feats is not None else None)

    # Extend aggregate with dist_avg means if available
    if per_model_results is not None:
        for split_name, entries in [
            ('all', list(per_coarse.values())),
            ('multi_class', [e for e in per_coarse.values() if e.get('k_true', 1) > 1]),
        ]:
            d = agg.setdefault(split_name, {})
            for key in ('dist_avg_agglo_acc', 'dist_avg_agglo_ami',
                        'dist_avg_spectral_acc', 'dist_avg_spectral_ami'):
                vals = [e[key] for e in entries if key in e and not np.isnan(e[key])]
                d[key] = float(np.mean(vals)) if vals else float('nan')

    print_aggregate_summary(agg, methods=cluster_methods,
                            alt_name='transformer' if alt_feats is not None else None)

    # Print dist_avg summary if present
    if per_model_results is not None:
        for split_name in ('all', 'multi_class'):
            d = agg.get(split_name, {})
            n = d.get('n_coarses', '?')
            for key in ('dist_avg_agglo_acc', 'dist_avg_agglo_ami',
                        'dist_avg_spectral_acc', 'dist_avg_spectral_ami'):
                val = d.get(key, float('nan'))
                print(f"  [{split_name} n={n}] {key}: {val:.3f}")

    # ---- Confidence / precision curve ----
    extra_data = {}
    if results is not None:
        # Build per-coarse confidence from p_split
        confidence_per_coarse = {}
        for r in results:
            sid = str(r["split_id"])
            confidence_per_coarse[sid] = float(r.get("p_split", 0.5))

        precision_curve = compute_precision_curve(
            confidence_per_coarse, oracle_k, n_thresholds=21
        )
        method_label = 'oracle_transformer' if getattr(args, 'use_oracle_k', False) else 'transformer'
        print_precision_summary(precision_curve, method_name=method_label)

        extra_data = {
            "confidence_scores": {
                sid: {
                    "confidence": round(conf, 4),
                    "k_hat": int(k_per_coarse.get(sid, 1)),
                    "k_true": int(oracle_k.get(int(sid), 1)),
                }
                for sid, conf in confidence_per_coarse.items()
            },
            "precision_curve": precision_curve,
        }

    metrics_path = args.k_results_path.replace('.json', '_metrics.json')
    method_name = 'oracle_transformer' if getattr(args, 'use_oracle_k', False) else 'transformer'
    save_hier_metrics(k_per_coarse, per_coarse, agg, method_name, args.dataset_name,
                      metrics_path, extra_data=extra_data if extra_data else None)


if __name__ == '__main__':
    main()
