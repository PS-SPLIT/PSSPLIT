"""
uno_hier_training.py — Hierarchical UNO+ training for coarse-label category discovery.

Extends UNO+ (methods/uno_training/uno_training.py) with per-coarse prototype masking:
  - Prototypes are partitioned into contiguous blocks, one block per unknown coarse category.
  - Sinkhorn-Knopp soft targets are generated per-coarse block (not globally).
  - Logits for unknown prototypes outside a sample's coarse category are masked to -1e9.
  - The overclustering head remains global (no coarse masking), as in UNO_main_updated.

Dataset structure assumption:
  Every coarse category is either fully known (all fine classes in train_classes)
  or fully unknown (all fine classes in unlabeled_classes). No mixed coarses.

Run via run_pipeline_hier.py with --training_method uno_hier, or standalone:
    python -m methods.uno_hier_training.uno_hier_training \
        --dataset_name scars \
        --coarse_labels_path data/coarse_labels/scars.json \
        --k_results_path /tmp/scars_k.json \
        --epochs 5 --temperature 0.1
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import dino_pretrain_path2, exp_root
from data.augmentations import get_transform
from data.data_utils import (CoarseBalancedBatchSampler, CoarseMergedDataset,
                              build_coarse_of, load_coarse_map,
                              oracle_k_per_coarse)
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from models import vision_transformer2 as vits
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import AverageMeter, get_mean_lr, init_experiment, str2bool

# Re-use network components and loss helpers from uno_training (avoid duplication)
from methods.uno_training.uno_training import (
    ContrastiveLearningViewGenerator,
    MultiHead,
    Prototypes,
    SinkhornKnopp,
    cross_entropy_loss,
    swapped_prediction,
)


# ---------------------------------------------------------------------------
# Prototype allocation
# ---------------------------------------------------------------------------

def build_proto_allocation(k_per_coarse):
    """
    Build contiguous prototype index ranges for unknown coarse categories.

    Parameters
    ----------
    k_per_coarse : dict  {str(coarse_id): int(k_i)}

    Returns
    -------
    allocation  : dict  {int(coarse_id): (start, end)}   [start, end) into unknown prototype space
    total_k     : int   sum of all K_i (total unknown prototypes)
    """
    allocation = {}
    offset = 0
    for c_str in sorted(k_per_coarse, key=lambda x: int(x)):
        k_i = int(k_per_coarse[c_str])
        allocation[int(c_str)] = (offset, offset + k_i)
        offset += k_i
    return allocation, offset


# ---------------------------------------------------------------------------
# Per-coarse logit masking
# ---------------------------------------------------------------------------

def build_unlab_proto_mask(coarse_unlab, allocation, num_labeled, total_k_unlab):
    """
    Build a boolean mask over the joint [nlc + total_k_unlab] logit space.
    True = allowed prototype for this sample.

    Parameters
    ----------
    coarse_unlab : torch.Tensor  (B_unlab,) long  — coarse IDs for unlabeled samples
    allocation   : dict  {coarse_id: (start, end)}
    num_labeled  : int   number of labeled classes (nlc)
    total_k_unlab: int   total unknown prototypes

    Returns
    -------
    mask : torch.BoolTensor  (B_unlab, nlc + total_k_unlab)
    """
    B = len(coarse_unlab)
    mask = torch.zeros(B, num_labeled + total_k_unlab, dtype=torch.bool,
                       device=coarse_unlab.device)
    mask[:, :num_labeled] = True   # labeled head always visible
    for i, c in enumerate(coarse_unlab.tolist()):
        if c not in allocation:
            continue  # old unlabeled from known coarse: labeled head only (already True above)
        s, e = allocation[c]
        mask[i, num_labeled + s : num_labeled + e] = True
    return mask


def mask_joint_logits(logits, mask):
    """Set disallowed logit positions to -1e9. logits and mask: same shape."""
    return logits.masked_fill(~mask, -1e9)


# ---------------------------------------------------------------------------
# Per-coarse Sinkhorn-Knopp targets
# ---------------------------------------------------------------------------

def masked_sinkhorn_targets(unlab_logits, coarse_unlab, allocation, sk):
    """
    Apply Sinkhorn-Knopp within each coarse prototype block independently.

    Parameters
    ----------
    unlab_logits : torch.Tensor  (B_unlab, total_k_unlab)
    coarse_unlab : torch.Tensor  (B_unlab,) long
    allocation   : dict  {coarse_id: (start, end)}
    sk           : SinkhornKnopp module

    Returns
    -------
    targets : torch.Tensor  (B_unlab, total_k_unlab)
    """
    targets = torch.zeros_like(unlab_logits)
    for c, (s, e) in allocation.items():
        idx = (coarse_unlab == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        targets[idx, s:e] = sk(unlab_logits[idx, s:e])
    return targets


# ---------------------------------------------------------------------------
# UNOHierModel — same structure as UNOPlusModel but total_k_unknown = Σ K_i
# ---------------------------------------------------------------------------

class UNOHierModel(nn.Module):
    """Hierarchical UNO classification heads (backbone-free)."""

    def __init__(self, feat_dim, num_labeled, total_k_unknown, proj_dim=256,
                 hidden_dim=2048, overcluster_factor=3, num_heads=5,
                 num_hidden_layers=1):
        super().__init__()
        self.head_lab = Prototypes(feat_dim, num_labeled)
        self.head_unlab = MultiHead(
            feat_dim, hidden_dim, proj_dim, total_k_unknown,
            num_heads, num_hidden_layers,
        )
        # Overclustering head stays global (no per-coarse constraint)
        self.head_unlab_over = MultiHead(
            feat_dim, hidden_dim, proj_dim, total_k_unknown * overcluster_factor,
            num_heads, num_hidden_layers,
        )

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        self.head_unlab.normalize_prototypes()
        self.head_unlab_over.normalize_prototypes()

    def forward(self, feats):
        logits_lab = self.head_lab(F.normalize(feats, dim=1))   # [B, nlc]
        logits_unlab, _      = self.head_unlab(feats)           # [H, B, K_total]
        logits_unlab_over, _ = self.head_unlab_over(feats)      # [H, B, K_total*of]
        return logits_lab, logits_unlab, logits_unlab_over


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _cluster_acc(y_true, y_pred):
    """Hungarian-matching accuracy between true labels and cluster assignments."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    return cost[row_ind, col_ind].sum() / len(y_true)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_uno_hier(backbone, model, loader, epoch, save_name, best_head,
                  allocation, args):
    """Evaluate with argmax over cat([logits_lab, logits_unlab[best_head]])."""
    backbone.eval()
    model.eval()
    all_preds, targets, mask = [], np.array([]), np.array([])

    for images, label, _ in tqdm(loader):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feats = backbone(images)
            logits_lab, logits_unlab, _ = model(feats)
            logits_inc = torch.cat([logits_lab, logits_unlab[best_head]], dim=-1)
            preds = logits_inc.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        targets = np.append(targets, label.numpy())
        mask = np.append(mask, np.array([
            x.item() in range(len(args.train_classes)) for x in label
        ]))

    all_preds = np.concatenate(all_preds)
    results = log_accs_from_preds(
        y_true=targets, y_pred=all_preds, mask=mask.astype(bool),
        eval_funcs=args.eval_funcs, save_name=save_name,
        T=epoch, writer=args.writer, print_output=True,
    )
    first_func = args.eval_funcs[0]
    return results[first_func]


def _log_per_coarse_acc(backbone, model, loader, epoch, best_head,
                        full_allocation, known_coarses, coarse_of, args):
    """Per-coarse Hungarian-matching accuracy from the parametric head."""
    backbone.eval()
    model.eval()
    all_preds, all_labels, all_coarse = [], [], []

    for images, label, _ in loader:
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feats = backbone(images)
            logits_lab, logits_unlab, _ = model(feats)
            logits_inc = torch.cat([logits_lab, logits_unlab[best_head]], dim=-1)
            preds = logits_inc.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        lbl_np = label.numpy()
        all_labels.append(lbl_np)
        all_coarse.append(np.array([int(coarse_of[int(l)]) for l in lbl_np]))

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_coarse = np.concatenate(all_coarse)

    known_accs, unknown_accs = [], []
    for c in sorted(full_allocation.keys()):
        c_mask = (all_coarse == c)
        if c_mask.sum() < 2:
            continue
        acc_c    = _cluster_acc(all_labels[c_mask], all_preds[c_mask])
        is_known = c in known_coarses
        tag      = 'known' if is_known else 'unknown'
        (known_accs if is_known else unknown_accs).append(acc_c)
        print(f'  Coarse {c:3d} ({tag}) | ACC {acc_c:.4f}')
        if args.writer is not None:
            args.writer.add_scalar(f'PerCoarseGlobal_{c}/ACC', acc_c, epoch)

    mean_known   = float(np.mean(known_accs))   if known_accs   else float('nan')
    mean_unknown = float(np.mean(unknown_accs)) if unknown_accs else float('nan')
    mean_all     = float(np.mean(known_accs + unknown_accs)) if (known_accs or unknown_accs) else float('nan')
    print(f'  [Global] Macro Known {mean_known:.4f} | Unknown {mean_unknown:.4f} | All {mean_all:.4f}')
    if args.writer is not None:
        args.writer.add_scalar('PerCoarseGlobal/Known',   mean_known,   epoch)
        args.writer.add_scalar('PerCoarseGlobal/Unknown', mean_unknown, epoch)
        args.writer.add_scalar('PerCoarseGlobal/All',     mean_all,     epoch)


def test_kmeans_per_coarse_separate(backbone, loader, epoch,
                                     full_allocation, known_coarses, coarse_of, args):
    """Independent sklearn KMeans per coarse on backbone features."""
    from sklearn.cluster import KMeans
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
        k_c     = min(k_c, n_c)
        preds_c = KMeans(n_clusters=k_c, random_state=0, n_init=10).fit_predict(all_feats[c_mask])
        acc_c   = _cluster_acc(all_labels[c_mask], preds_c)
        is_known = c in known_coarses
        tag      = 'known' if is_known else 'unknown'
        (known_accs if is_known else unknown_accs).append(acc_c)
        print(f'  Coarse {c:3d} ({tag}, k={k_c}) | ACC {acc_c:.4f}')
        if args.writer is not None:
            args.writer.add_scalar(f'PerCoarseSep_{c}/ACC', acc_c, epoch)

    mean_known   = float(np.mean(known_accs))   if known_accs   else float('nan')
    mean_unknown = float(np.mean(unknown_accs)) if unknown_accs else float('nan')
    mean_all     = float(np.mean(known_accs + unknown_accs)) if (known_accs or unknown_accs) else float('nan')
    print(f'  [Separate] Macro Known {mean_known:.4f} | Unknown {mean_unknown:.4f} | All {mean_all:.4f}')
    if args.writer is not None:
        args.writer.add_scalar('PerCoarseSep/Known',   mean_known,   epoch)
        args.writer.add_scalar('PerCoarseSep/Unknown', mean_unknown, epoch)
        args.writer.add_scalar('PerCoarseSep/All',     mean_all,     epoch)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(backbone, model, train_loader, test_loader, unlabelled_loader,
          allocation, coarse_of, full_allocation, known_coarses, args):
    sk  = SinkhornKnopp(num_iters=args.num_iters_sk, epsilon=args.epsilon_sk).cuda()
    nlc = args.num_labeled_classes
    total_k = sum(e - s for s, e in allocation.values())

    optimizer = SGD(
        list(backbone.parameters()) + list(model.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                               eta_min=args.lr * 1e-3)
    best_test_acc_old = 0.0
    best_head = 0

    for epoch in range(args.epochs):
        backbone.train()
        model.train()
        loss_record  = AverageMeter()
        loss_per_head = torch.zeros(args.num_heads).cuda()

        for batch in tqdm(train_loader):
            # CoarseMergedDataset returns: (images, class_labels, uq_idxs, coarse_labels, mask_lab)
            images, class_labels, uq_idxs, coarse_labels, mask_lab = batch
            mask_lab     = mask_lab[:, 0].cuda().bool()
            class_labels = class_labels.cuda(non_blocking=True)
            coarse_labels = coarse_labels.cuda(non_blocking=True)
            views = [v.cuda(non_blocking=True) for v in images]
            B = class_labels.shape[0]

            coarse_unlab = coarse_labels[~mask_lab]  # coarse IDs for unlabeled samples

            model.normalize_prototypes()

            feats_list = [backbone(v) for v in views]
            logits_lab_list, logits_unlab_list, logits_unlab_over_list = [], [], []
            for feats in feats_list:
                ll, lu, luo = model(feats)
                logits_lab_list.append(ll)
                logits_unlab_list.append(lu)
                logits_unlab_over_list.append(luo)

            H = args.num_heads

            # Build combined logits [H, B, nlc + total_k]
            logits_list, logits_over_list = [], []
            for ll, lu, luo in zip(logits_lab_list, logits_unlab_list, logits_unlab_over_list):
                ll_exp = ll.unsqueeze(0).expand(H, -1, -1)
                logits_list.append(torch.cat([ll_exp, lu], dim=-1))
                logits_over_list.append(torch.cat([ll_exp, luo], dim=-1))

            targets_lab = F.one_hot(
                class_labels[mask_lab], num_classes=nlc
            ).float()

            V = len(views)
            targets      = torch.zeros_like(logits_list[0]).unsqueeze(0).expand(V, -1, -1, -1).clone()
            targets_over = torch.zeros_like(logits_over_list[0]).unsqueeze(0).expand(V, -1, -1, -1).clone()

            has_unlab = mask_lab.logical_not().any()

            for v in range(V):
                for h in range(H):
                    # Labeled: one-hot in labeled head positions
                    targets[v, h, mask_lab, :nlc]      = targets_lab
                    targets_over[v, h, mask_lab, :nlc] = targets_lab

                    if has_unlab:
                        # --- Hierarchical UNO+ target construction ---
                        # Unknown prototypes: per-coarse SK (not global)
                        targets[v, h, ~mask_lab, nlc:] = masked_sinkhorn_targets(
                            logits_list[v][h, ~mask_lab, nlc:],
                            coarse_unlab, allocation, sk,
                        )
                        # Labeled head positions for unlabeled: SK over full concat
                        # (same as UNO+ for the labeled-head portion)
                        targets[v, h, ~mask_lab, :nlc] = sk(
                            logits_list[v][h, ~mask_lab]
                        )[:, :nlc]

                        # Overclustering head: global SK (no per-coarse constraint)
                        targets_over[v, h, ~mask_lab, :] = sk(
                            logits_over_list[v][h, ~mask_lab]
                        )

                        # --- Per-coarse logit masking before loss computation ---
                        proto_mask = build_unlab_proto_mask(
                            coarse_unlab, allocation, nlc, total_k
                        )
                        logits_list[v] = logits_list[v].clone()
                        logits_list[v][h, ~mask_lab] = mask_joint_logits(
                            logits_list[v][h, ~mask_lab], proto_mask
                        )

            loss_cluster = swapped_prediction(
                logits_list, [targets[v] for v in range(V)], args.temperature
            )
            loss_overcluster = swapped_prediction(
                logits_over_list, [targets_over[v] for v in range(V)], args.temperature
            )

            loss_per_head += loss_cluster.detach()
            loss = (loss_cluster.mean() + loss_overcluster.mean()) / 2
            loss_record.update(loss.item(), B)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        best_head = loss_per_head.argmin().item()
        print(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f} '
              f'| Best Head: {best_head} | LR: {get_mean_lr(optimizer):.6f}')
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        scheduler.step()

        with torch.no_grad():
            all_acc_test, old_acc_test, new_acc_test = test_uno_hier(
                backbone, model, test_loader, epoch=epoch,
                save_name='Test ACC', best_head=best_head,
                allocation=allocation, args=args,
            )
            print(f'Test  Accuracies: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}')

            if (epoch + 1) % args.train_report_interval == 0:
                all_acc, old_acc, new_acc = test_uno_hier(
                    backbone, model, unlabelled_loader, epoch=epoch,
                    save_name='Train ACC Unlabelled', best_head=best_head,
                    allocation=allocation, args=args,
                )
                print(f'Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
                print('Per-coarse evaluation (global head):')
                _log_per_coarse_acc(
                    backbone, model, unlabelled_loader, epoch, best_head,
                    full_allocation, known_coarses, coarse_of, args,
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
        description='Hierarchical UNO+ training (SelEx-integrated)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / model
    parser.add_argument('--dataset_name',      type=str,   default='scars')
    parser.add_argument('--base_model',        type=str,   default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits',    type=str2bool, default=True)
    parser.add_argument('--num_novel_classes', type=int,   default=None)
    parser.add_argument('--num_unlabeled_classes', type=int, default=None)
    parser.add_argument('--warmup_model_dir',  type=str,   default=None)
    # Coarse labels
    parser.add_argument('--coarse_labels_path', type=str, required=True,
                        help='JSON mapping fine_class_idx -> coarse_class_idx')
    parser.add_argument('--k_results_path',    type=str, default=None,
                        help='JSON with per-coarse K {coarse_id: k}. '
                             'If None, oracle K is derived from coarse JSON.')
    # Optimisation
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--epochs',        type=int,   default=200)
    parser.add_argument('--lr',            type=float, default=0.1)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--transform',     type=str,   default='imagenet')
    # UNO-specific
    parser.add_argument('--temperature',   type=float, default=0.1)
    parser.add_argument('--num_heads',     type=int,   default=5)
    parser.add_argument('--overcluster_factor', type=int, default=3)
    parser.add_argument('--proj_dim',      type=int,   default=256)
    parser.add_argument('--hidden_dim',    type=int,   default=2048)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--epsilon_sk',    type=float, default=0.05)
    parser.add_argument('--num_iters_sk',  type=int,   default=3)
    # Eval / logging
    parser.add_argument('--eval_funcs', nargs='+', default=['v2'])
    parser.add_argument('--exp_root',   type=str,  default=exp_root)
    parser.add_argument('--exp_id',     type=str,  default=None,
                        help='Fixed experiment ID for the log dir (default: use timestamp)')
    # Split / reproducibility / reporting
    parser.add_argument('--hier_split_type',       type=str,    default=None)
    parser.add_argument('--seed',                  type=int,    default=1)
    parser.add_argument('--train_report_interval', type=int,    default=200)
    parser.add_argument('--use_coarse_sampler',    type=str2bool, default=False,
                        help='CoarseBalancedBatchSampler instead of WeightedRandomSampler')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0')

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes = len(args.train_classes)
    if args.num_unlabeled_classes is None:
        args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['uno_hier', 'dinov2', args.dataset_name], exp_id=args.exp_id)

    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768

    # ---- Load coarse labels ----
    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    # ---- Load or derive K per coarse ----
    if args.k_results_path is not None and os.path.exists(args.k_results_path):
        with open(args.k_results_path) as f:
            raw = json.load(f)
        k_per_coarse_int = {int(k): int(v) for k, v in raw.items()}
        print(f'Loaded per-coarse K from {args.k_results_path}')
    else:
        print('No k_results_path found — deriving oracle K from coarse JSON.')
        k_per_coarse_int = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)
        print(f'Oracle K per coarse: {k_per_coarse_int}')

    k_per_coarse = {str(k): int(v) for k, v in k_per_coarse_int.items()}

    allocation, total_k_unknown = build_proto_allocation(k_per_coarse)
    known_coarses  = {coarse_of[i] for i in range(args.num_labeled_classes)}
    known_coarse_k = Counter(coarse_of[i] for i in range(args.num_labeled_classes))
    full_allocation = {int(c): int(k) for c, k in k_per_coarse_int.items()}
    full_allocation.update(known_coarse_k)
    print(f'K: labeled={args.num_labeled_classes} | unknown={total_k_unknown} | '
          f'total={args.num_labeled_classes + total_k_unknown}')
    print(f'Known coarses: {len(known_coarses)} | Unknown coarses: {len(k_per_coarse_int)}')

    # ---- Backbone ----
    backbone = vits.__dict__['vit_base']()
    state_dict = torch.load(dino_pretrain_path2, map_location='cpu')
    backbone.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        print(f'Loading backbone from {args.warmup_model_dir}')
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
    train_transform = ContrastiveLearningViewGenerator(train_transform, n_views=2)

    _, _, unlabelled_train_examples_test, raw_datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )
    # Build test dataset separately (no multi-view transform)
    _, test_dataset_clean, _, _ = get_datasets(
        args.dataset_name, None, test_transform, args
    )

    # CoarseMergedDataset wraps labeled + unlabeled datasets with coarse labels
    train_dataset = CoarseMergedDataset(
        labelled_dataset=raw_datasets['train_labelled'],
        unlabelled_dataset=raw_datasets['train_unlabelled'],
        coarse_of=coarse_of,
    )

    label_len      = len(raw_datasets['train_labelled'])
    unlabelled_len = len(raw_datasets['train_unlabelled'])

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
        test_dataset_clean, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False,
    )
    unlabelled_loader = DataLoader(
        unlabelled_train_examples_test, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False,
    )

    # ---- Model ----
    model = UNOHierModel(
        feat_dim=args.feat_dim,
        num_labeled=args.num_labeled_classes,
        total_k_unknown=total_k_unknown,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        overcluster_factor=args.overcluster_factor,
        num_heads=args.num_heads,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    train(backbone, model, train_loader, test_loader, unlabelled_loader,
          allocation, coarse_of, full_allocation, known_coarses, args)


if __name__ == '__main__':
    main()
