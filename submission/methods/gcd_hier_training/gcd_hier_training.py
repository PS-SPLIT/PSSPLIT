"""
gcd_hier_training.py — Hierarchical GCD paper training for coarse-label category discovery.

Extends the GCD paper method (methods/gcd_training/gcd_training.py) with:
  - Selectable NTXent mode: global (standard) or within-coarse (--ntxent_mode)
  - Selectable fine SupCon mode: global negatives or within-coarse negatives (--supcon_mode)
  - Optional coarse-label SupCon over ALL samples (--use_coarse_supcon)
  - Optional coarse-balanced batch sampler (--use_coarse_sampler)
  - Fixed per-coarse evaluation for strict hierarchical splits (global KMeans decomposed)
  - Per-coarse separate KMeans evaluation with oracle/estimated K per coarse

Dataset structure assumption:
  Every coarse category is either fully known (all fine classes in train_classes)
  or fully unknown (all fine classes in unlabeled_classes). No mixed coarses.

Run via run_pipeline_hier.py with --training_method gcd_hier, or standalone:
    python -m methods.gcd_hier_training.gcd_hier_training \
        --dataset_name cub \
        --coarse_labels_path data/coarse_labels/cub.json \
        --ntxent_mode global --supcon_mode within_coarse \
        --use_coarse_supcon True --seed 1
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import dino_pretrain_path2, exp_root
from data.augmentations import get_transform
from data.data_utils import (CoarseBalancedBatchSampler, CoarseMergedDataset,
                              build_coarse_of, load_coarse_map, oracle_k_per_coarse)
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from models import vision_transformer2 as vits
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import AverageMeter, get_mean_lr, init_experiment, str2bool

# Reuse model components from standard GCD (no duplication)
from methods.gcd_training.gcd_training import (
    ContrastiveLearningViewGenerator,
    SupConLoss,
    info_nce_logits,
)

# build_proto_allocation used only for total_k_unknown computation
from methods.uno_hier_training.uno_hier_training import build_proto_allocation


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def within_coarse_sup_con_loss(features, labels, coarse_labels,
                                temperature=0.07, base_temperature=0.07):
    """
    Supervised contrastive loss restricted to within-coarse negatives.

    Positives: same fine class label (augmented views of the same sample).
    Denominator: within-coarse pairs only (same coarse, not self).
    Cross-coarse pairs are excluded entirely.
    Anchors whose denominator contains only their own positives (singleton coarses,
    or coarses underrepresented in the batch) are skipped.

    Args:
        features:      [B, 2, D] — both views, L2-normalised
        labels:        [B]        — fine class labels (labeled samples only)
        coarse_labels: [B]        — coarse labels
        temperature:   float

    Returns:
        Scalar loss (zero-grad fallback when no valid anchors).
    """
    device = features.device
    B = features.shape[0]
    if B == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # [2B, D] — both views as both anchors and contrasts
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    dot = torch.div(
        torch.matmul(contrast_feature, contrast_feature.T), temperature
    )
    dot_max, _ = torch.max(dot, dim=1, keepdim=True)
    dot = dot - dot_max.detach()

    # Positive mask: same fine class [2B, 2B]
    labels_2b = torch.cat([labels, labels], dim=0)
    pos_mask = torch.eq(labels_2b.unsqueeze(0), labels_2b.unsqueeze(1)).float()

    # Same-coarse mask [2B, 2B]
    coarse_2b = torch.cat([coarse_labels, coarse_labels], dim=0)
    same_coarse = torch.eq(coarse_2b.unsqueeze(0), coarse_2b.unsqueeze(1)).float()

    # Self-mask — zero out diagonal
    self_mask = torch.scatter(
        torch.ones(2 * B, 2 * B, device=device), 1,
        torch.arange(2 * B, device=device).view(-1, 1), 0
    )

    pos_mask  = pos_mask  * self_mask          # same fine class, not self
    denom_mask = same_coarse * self_mask       # within-coarse, not self

    # Skip anchors with no within-coarse negatives beyond their positives
    has_neg = ((denom_mask - pos_mask).sum(1) > 0)
    if has_neg.sum() == 0:
        return dot.sum() * 0.0

    exp_dot  = torch.exp(dot) * denom_mask
    log_prob = dot - torch.log(exp_dot.sum(1, keepdim=True) + 1e-8)

    valid_pos  = pos_mask[has_neg]
    pos_count  = valid_pos.sum(1).clamp(min=1)
    mean_log   = (valid_pos * log_prob[has_neg]).sum(1) / pos_count
    loss       = -(temperature / base_temperature) * mean_log
    return loss.mean()


# ---------------------------------------------------------------------------
# Within-coarse NTXent
# ---------------------------------------------------------------------------

def info_nce_logits_coarse(features, coarse_labels_2v, temperature=1.0):
    """
    NTXent loss restricted to within-coarse negatives.

    For each anchor, the denominator includes only samples from the same coarse
    category (excluding self). Cross-coarse pairs are not used as negatives.
    Anchors with no within-coarse negatives beyond their positive pair are skipped.

    Args:
        features:        [2B, dim] — two views concatenated, already L2-normalised
        coarse_labels_2v:[2B]      — coarse label for each row
        temperature:     float

    Returns:
        Scalar loss.
    """
    device = features.device
    n = features.size(0)   # 2B
    B = n // 2

    sample_ids  = torch.cat([torch.arange(B), torch.arange(B)]).to(device)
    sim         = torch.matmul(features, features.T) / temperature
    self_mask   = torch.eye(n, dtype=torch.bool, device=device)
    pos_mask    = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)) & ~self_mask
    same_coarse = (coarse_labels_2v.unsqueeze(0) == coarse_labels_2v.unsqueeze(1))
    denom_mask  = same_coarse & ~self_mask

    has_neg  = (denom_mask & ~pos_mask).sum(1) > 0
    sim_denom = sim.masked_fill(~denom_mask, float('-inf'))
    log_denom = torch.logsumexp(sim_denom, dim=1)
    sim_pos   = (sim * pos_mask.float()).sum(1)

    ntxent = (-sim_pos + log_denom)[has_neg]
    return ntxent.mean() if ntxent.numel() > 0 else sim.sum() * 0.0


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _cluster_acc(y_true, y_pred):
    """Overall clustering accuracy via Hungarian matching."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    D = max(y_true.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i, j in zip(y_pred, y_true):
        w[i, j] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / len(y_true)


def test_kmeans(backbone, loader, epoch, save_name, n_clusters, args):
    """Evaluate backbone features with a single global KMeans clustering."""
    backbone.eval()
    all_feats, targets, mask = [], np.array([]), np.array([])

    for images, label, _ in tqdm(loader):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feats = backbone(images)
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.numpy())
        mask = np.append(mask, np.array([
            x.item() in range(len(args.train_classes)) for x in label
        ]))

    all_feats = np.concatenate(all_feats)
    preds = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(all_feats)

    results = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask,
        eval_funcs=args.eval_funcs, save_name=save_name,
        T=epoch, writer=args.writer, print_output=True,
    )
    return results[args.eval_funcs[0]]


def test_kmeans_per_coarse(backbone, loader, epoch, n_clusters,
                            full_allocation, known_coarses, coarse_of, args):
    """
    Global KMeans decomposed per coarse — fixed for strict hierarchical splits.

    Runs one global KMeans then evaluates clustering accuracy within each coarse
    using Hungarian matching. Reports macro-averages over known and unknown coarses.

    Unlike the original version, this correctly handles strict hierarchical splits
    (where known coarses have only old-class samples and unknown coarses have only
    new-class samples) by dropping the requirement for both old and new within each
    coarse.
    """
    backbone.eval()
    all_feats, all_labels = [], []

    for images, label, _ in loader:
        with torch.no_grad():
            feats = backbone(images.cuda(non_blocking=True))
        all_feats.append(F.normalize(feats, dim=-1).cpu().numpy())
        all_labels.append(label.numpy())

    all_feats  = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    all_coarse = np.array([int(coarse_of[int(l)]) for l in all_labels])

    preds = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(all_feats)

    known_accs, unknown_accs = [], []
    for c in sorted(full_allocation.keys()):
        c_mask = (all_coarse == c)
        if c_mask.sum() < 2:
            continue
        acc_c    = _cluster_acc(all_labels[c_mask], preds[c_mask])
        is_known = c in known_coarses
        tag      = 'known' if is_known else 'unknown'
        (known_accs if is_known else unknown_accs).append(acc_c)
        print(f'  Coarse {c:3d} ({tag}) | ACC {acc_c:.4f}')
        if args.writer is not None:
            args.writer.add_scalar(f'PerCoarseGlobal_{c}/ACC', acc_c, epoch)

    mean_known   = float(np.mean(known_accs))   if known_accs   else float('nan')
    mean_unknown = float(np.mean(unknown_accs)) if unknown_accs else float('nan')
    mean_all     = float(np.mean(known_accs + unknown_accs))
    print(f'  [Global] Macro Known {mean_known:.4f} | Unknown {mean_unknown:.4f} | All {mean_all:.4f}')
    if args.writer is not None:
        args.writer.add_scalar('PerCoarseGlobal/Known',   mean_known,   epoch)
        args.writer.add_scalar('PerCoarseGlobal/Unknown', mean_unknown, epoch)
        args.writer.add_scalar('PerCoarseGlobal/All',     mean_all,     epoch)


def test_kmeans_per_coarse_separate(backbone, loader, epoch,
                                     full_allocation, known_coarses, coarse_of, args):
    """
    Per-coarse separate KMeans with oracle/estimated K per coarse.

    For each coarse, runs an independent KMeans using that coarse's allocated K.
    Clustering quality is measured with Hungarian matching ACC within each coarse.
    Reports macro-averages over known and unknown coarses.
    """
    backbone.eval()
    all_feats, all_labels = [], []

    for images, label, _ in loader:
        with torch.no_grad():
            feats = backbone(images.cuda(non_blocking=True))
        all_feats.append(F.normalize(feats, dim=-1).cpu().numpy())
        all_labels.append(label.numpy())

    all_feats  = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    all_coarse = np.array([int(coarse_of[int(l)]) for l in all_labels])

    known_accs, unknown_accs = [], []
    for c, k_c in sorted(full_allocation.items()):
        c_mask = (all_coarse == c)
        n_c    = int(c_mask.sum())
        if n_c < 2 or k_c < 2:
            continue
        k_c    = min(k_c, n_c)   # can't have more clusters than samples
        preds_c = KMeans(n_clusters=k_c, random_state=0).fit_predict(all_feats[c_mask])
        acc_c   = _cluster_acc(all_labels[c_mask], preds_c)
        is_known = c in known_coarses
        tag      = 'known' if is_known else 'unknown'
        (known_accs if is_known else unknown_accs).append(acc_c)
        print(f'  Coarse {c:3d} ({tag}, k={k_c}) | ACC {acc_c:.4f}')
        if args.writer is not None:
            args.writer.add_scalar(f'PerCoarseSep_{c}/ACC', acc_c, epoch)

    mean_known   = float(np.mean(known_accs))   if known_accs   else float('nan')
    mean_unknown = float(np.mean(unknown_accs)) if unknown_accs else float('nan')
    mean_all     = float(np.mean(known_accs + unknown_accs))
    print(f'  [Separate] Macro Known {mean_known:.4f} | Unknown {mean_unknown:.4f} | All {mean_all:.4f}')
    if args.writer is not None:
        args.writer.add_scalar('PerCoarseSep/Known',   mean_known,   epoch)
        args.writer.add_scalar('PerCoarseSep/Unknown', mean_unknown, epoch)
        args.writer.add_scalar('PerCoarseSep/All',     mean_all,     epoch)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(projection_head, backbone, train_loader, test_loader,
          unlabelled_loader, full_allocation, total_k_unknown,
          known_coarses, coarse_of, args):

    optimizer = SGD(
        list(projection_head.parameters()) + list(backbone.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler    = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                   eta_min=args.lr * 1e-3)
    sup_con_crit = SupConLoss()
    best_test_acc_old = 0.0
    n_clusters = args.num_labeled_classes + total_k_unknown

    for epoch in range(args.epochs):
        backbone.train()
        projection_head.train()
        loss_record = AverageMeter()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, coarse_labels, mask_lab = batch
            mask_lab      = mask_lab[:, 0]
            class_labels  = class_labels.cuda(non_blocking=True)
            mask_lab      = mask_lab.cuda(non_blocking=True).bool()
            coarse_labels = coarse_labels.cuda(non_blocking=True)
            images        = torch.cat(images, dim=0).cuda(non_blocking=True)

            features = backbone(images)
            features = projection_head(features)
            features = F.normalize(features, dim=-1)

            # --- NTXent (unsupervised contrastive) ---
            if args.contrast_unlabel_only:
                f1u, f2u  = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1u, f2u], dim=0)
                coarse_2v = torch.cat(
                    [coarse_labels[~mask_lab], coarse_labels[~mask_lab]], dim=0
                )
            else:
                con_feats = features
                coarse_2v = torch.cat([coarse_labels, coarse_labels], dim=0)

            if args.ntxent_mode == 'global':
                contrastive_logits, contrastive_labels_ce = info_nce_logits(con_feats, args)
                contrastive_loss = torch.nn.CrossEntropyLoss()(
                    contrastive_logits, contrastive_labels_ce
                )
            else:  # 'coarse'
                contrastive_loss = info_nce_logits_coarse(
                    con_feats, coarse_2v, args.temperature
                )

            # --- Fine-class SupCon (labeled samples only) ---
            f1_lab, f2_lab = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats  = torch.cat([f1_lab.unsqueeze(1), f2_lab.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            if args.supcon_mode == 'global':
                fine_sup_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            else:  # 'within_coarse'
                fine_sup_loss = within_coarse_sup_con_loss(
                    sup_con_feats, sup_con_labels, coarse_labels[mask_lab],
                    temperature=sup_con_crit.temperature,
                )

            # --- Coarse SupCon (all samples, coarse labels as supervision) ---
            if args.use_coarse_supcon:
                f1_all, f2_all   = features.chunk(2)
                coarse_con_feats = torch.cat(
                    [f1_all.unsqueeze(1), f2_all.unsqueeze(1)], dim=1
                )
                coarse_sup_loss = sup_con_crit(coarse_con_feats, labels=coarse_labels)
            else:
                coarse_sup_loss = 0.0

            loss = (
                (1 - args.sup_con_weight) * contrastive_loss
                + args.sup_con_weight * fine_sup_loss
                + args.coarse_supcon_weight * coarse_sup_loss
            )

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f}')
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)
        scheduler.step()

        with torch.no_grad():
            if (epoch + 1) % args.train_report_interval == 0:
                print('Testing on unlabelled examples in the training data...')
                all_acc, old_acc, new_acc = test_kmeans(
                    backbone, unlabelled_loader, epoch=epoch,
                    save_name='Train ACC Unlabelled',
                    n_clusters=n_clusters, args=args,
                )

            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_kmeans(
                backbone, test_loader, epoch=epoch,
                save_name='Test ACC',
                n_clusters=n_clusters, args=args,
            )

        if (epoch + 1) % args.train_report_interval == 0:
            print(f'Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
        print(f'Test  Accuracies: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}')

        if (epoch + 1) % args.train_report_interval == 0:
            with torch.no_grad():
                print('Per-coarse evaluation (global KMeans):')
                test_kmeans_per_coarse(
                    backbone, unlabelled_loader, epoch,
                    n_clusters, full_allocation, known_coarses, coarse_of, args,
                )
                print('Per-coarse evaluation (separate KMeans per coarse):')
                test_kmeans_per_coarse_separate(
                    backbone, unlabelled_loader, epoch,
                    full_allocation, known_coarses, coarse_of, args,
                )

        torch.save(backbone.state_dict(), args.model_path)
        if old_acc_test > best_test_acc_old:
            best_test_acc_old = old_acc_test
            print(f'Best old-class ACC on test set: {old_acc_test:.4f}')
            torch.save(backbone.state_dict(),
                       os.path.join(args.model_dir, 'model_best.pt'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hierarchical GCD paper training (SelEx-integrated)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / model
    parser.add_argument('--dataset_name',      type=str,      default='scars')
    parser.add_argument('--base_model',        type=str,      default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float,    default=0.5)
    parser.add_argument('--use_ssb_splits',    type=str2bool, default=True)
    parser.add_argument('--num_novel_classes', type=int,      default=None)
    parser.add_argument('--warmup_model_dir',  type=str,      default=None,
                        help='Directory containing model_best.pt for backbone init.')
    parser.add_argument('--seed',              type=int,      default=1)
    # Optimisation
    parser.add_argument('--batch_size',        type=int,   default=128)
    parser.add_argument('--num_workers',       type=int,   default=4)
    parser.add_argument('--epochs',            type=int,   default=200)
    parser.add_argument('--lr',                type=float, default=0.1)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--grad_from_block',   type=int,   default=11)
    parser.add_argument('--transform',         type=str,   default='imagenet')
    parser.add_argument('--n_views',           type=int,   default=2)
    # GCD-specific
    parser.add_argument('--sup_con_weight',        type=float,    default=0.35)
    parser.add_argument('--temperature',           type=float,    default=1.0)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    # Loss mode flags
    parser.add_argument('--ntxent_mode',  type=str, default='coarse',
                        choices=['global', 'coarse'],
                        help='global: standard NTXent; coarse: within-coarse NTXent')
    parser.add_argument('--supcon_mode',  type=str, default='global',
                        choices=['global', 'within_coarse'],
                        help='global: all negatives; within_coarse: same-coarse negatives only')
    parser.add_argument('--use_coarse_supcon',    type=str2bool, default=False,
                        help='Add coarse-label SupCon loss over all samples')
    parser.add_argument('--coarse_supcon_weight', type=float,    default=0.25,
                        help='Weight for coarse SupCon term')
    parser.add_argument('--use_coarse_sampler',   type=str2bool, default=False,
                        help='Inverse-frequency coarse-balanced batch sampler')
    # Hierarchical-specific
    parser.add_argument('--coarse_labels_path', type=str, required=True,
                        help='JSON mapping fine_class_idx -> coarse_class_idx')
    parser.add_argument('--k_results_path',     type=str, default=None,
                        help='JSON with per-coarse K: {coarse_id: k_i}. '
                             'Uses oracle K if not provided.')
    parser.add_argument('--hier_split_type',    type=str, default=None)
    # Eval / logging
    parser.add_argument('--eval_funcs',           nargs='+', default=['v2'])
    parser.add_argument('--train_report_interval', type=int,  default=200)
    parser.add_argument('--exp_root',              type=str,  default=exp_root)
    parser.add_argument('--exp_id',                type=str,  default=None,
                        help='Fixed experiment ID for the log dir (default: use timestamp)')

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0')

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # --- Coarse label setup ---
    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    if args.k_results_path and os.path.exists(args.k_results_path):
        with open(args.k_results_path) as f:
            k_per_coarse = {int(k): v for k, v in json.load(f).items()}
    else:
        k_per_coarse = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)

    unknown_allocation, total_k_unknown = build_proto_allocation(k_per_coarse)

    # Known coarses and their per-coarse K (number of fine classes)
    known_coarses   = {coarse_of[i] for i in range(args.num_labeled_classes)}
    known_coarse_k  = Counter(coarse_of[i] for i in range(args.num_labeled_classes))

    # Full allocation: {coarse_id: int_k} for both unknown and known coarses.
    # k_per_coarse already stores int K for unknown coarses; unknown_allocation
    # stores (start, end) prototype index ranges used elsewhere — do NOT use it here.
    full_allocation = {int(c): int(k) for c, k in k_per_coarse.items()}
    full_allocation.update(known_coarse_k)

    print(f'K: labeled={args.num_labeled_classes} | unknown={total_k_unknown} | '
          f'total={args.num_labeled_classes + total_k_unknown}')
    print(f'Known coarses: {len(known_coarses)} | Unknown coarses: {len(unknown_allocation)}')

    init_experiment(args, runner_name=['gcd_hier', 'dinov2', args.dataset_name], exp_id=args.exp_id)

    args.interpolation  = 3
    args.crop_pct       = 0.875
    args.image_size     = 224
    args.feat_dim       = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim    = 65536

    # ---- Backbone ----
    print('Loading DINOv2 vitb14-reg4 backbone')
    backbone   = vits.__dict__['vit_base']()
    state_dict = torch.load(dino_pretrain_path2, map_location='cpu')
    backbone.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        print(f'Loading backbone weights from {args.warmup_model_dir}')
        backbone.load_state_dict(
            torch.load(args.warmup_model_dir + 'model_best.pt', map_location='cpu'),
            strict=False,
        )

    for param in backbone.parameters():
        param.requires_grad = False
    for name, param in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                param.requires_grad = True

    backbone = backbone.to(device)

    # ---- Transforms & datasets ----
    train_transform, test_transform = get_transform(
        args.transform, image_size=args.image_size, args=args
    )
    train_transform = ContrastiveLearningViewGenerator(train_transform, n_views=args.n_views)

    _, test_dataset, unlabelled_train_examples_test, raw_datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )
    train_dataset = CoarseMergedDataset(
        labelled_dataset=raw_datasets['train_labelled'],
        unlabelled_dataset=raw_datasets['train_unlabelled'],
        coarse_of=coarse_of,
    )

    label_len      = len(raw_datasets['train_labelled'])
    unlabelled_len = len(raw_datasets['train_unlabelled'])

    # ---- Batch sampler ----
    if args.use_coarse_sampler:
        all_classes_ordered   = list(args.train_classes) + list(args.unlabeled_classes)
        target_transform_dict = {c: i for i, c in enumerate(all_classes_ordered)}
        ld_targets = raw_datasets['train_labelled'].targets
        ud_targets = raw_datasets['train_unlabelled'].targets
        all_remapped    = ([target_transform_dict[int(t)] for t in ld_targets] +
                           [target_transform_dict[int(t)] for t in ud_targets])
        all_coarse_samp = [coarse_of[r] for r in all_remapped]

        coarse_index_groups = defaultdict(list)
        for idx, c in enumerate(all_coarse_samp):
            coarse_index_groups[c].append(idx)

        batch_sampler = CoarseBalancedBatchSampler(
            coarse_indices=coarse_index_groups,
            batch_size=args.batch_size,
            min_coarse_samples=4,
        )
        train_loader = DataLoader(
            train_dataset, num_workers=args.num_workers,
            batch_sampler=batch_sampler, pin_memory=True,
        )
    else:
        sample_weights = torch.DoubleTensor([
            1 if i < label_len else label_len / unlabelled_len
            for i in range(len(train_dataset))
        ])
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset)
        )
        train_loader = DataLoader(
            train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
            shuffle=False, sampler=sampler, drop_last=True, pin_memory=True,
        )
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False,
    )
    unlabelled_loader = DataLoader(
        unlabelled_train_examples_test, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False,
    )

    # ---- Projection head ----
    projection_head = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    ).to(device)

    train(projection_head, backbone, train_loader, test_loader,
          unlabelled_loader, full_allocation, total_k_unknown,
          known_coarses, coarse_of, args)


if __name__ == '__main__':
    main()
