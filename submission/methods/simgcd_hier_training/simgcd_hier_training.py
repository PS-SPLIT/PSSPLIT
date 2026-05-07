"""
simgcd_hier_training.py — Hierarchical SimGCD training for coarse-label category discovery.

Extends SimGCD (methods/simgcd_training/simgcd_training.py) with per-coarse prototype masking:
  - Prototypes are partitioned into contiguous blocks, one block per unknown coarse category.
  - Student and teacher logits for unlabeled samples are masked to -1e9 for prototypes
    outside their coarse category block before the distillation and MaxEnt losses.
  - Supervised CE, supervised contrastive, and unsupervised contrastive losses are optionally
    upgraded with within-coarse and coarse-level variants from gcd_hier_training.

Also adds constrained evaluation: since coarse labels are known for every sample, predictions
can be restricted to only the fine-class prototypes within the sample's coarse, giving a
more principled hierarchical accuracy metric.

Dataset structure assumption:
  Every coarse category is either fully known (all fine classes in train_classes)
  or fully unknown (all fine classes in unlabeled_classes). No mixed coarses.

Run via run_pipeline_hier.py with --training_method simgcd_hier, or standalone:
    python -m methods.simgcd_hier_training.simgcd_hier_training \
        --dataset_name cub \
        --coarse_labels_path data/coarse_labels/cub.json \
        --hier_split_type 5050 --epochs 5
"""

import argparse
import json
import math
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
                              build_coarse_of, load_coarse_map, oracle_k_per_coarse)
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from models import vision_transformer2 as vits
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import init_experiment, str2bool

# Reuse model components and losses from standard SimGCD (no duplication)
from methods.simgcd_training.simgcd_training import (
    ContrastiveLearningViewGenerator,
    DINOHead,
    DistillLoss,
    SupConLoss,
    get_params_groups,
    info_nce_logits,
)

# Reuse per-coarse masking helpers from UNO-hier (no duplication)
from methods.uno_hier_training.uno_hier_training import (
    build_proto_allocation,
    build_unlab_proto_mask,
    mask_joint_logits,
)

# Within-coarse and coarse-level contrastive losses from gcd-hier
from methods.gcd_hier_training.gcd_hier_training import (
    within_coarse_sup_con_loss,
    info_nce_logits_coarse,
)


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


def test_per_coarse(model, loader, epoch, full_allocation, known_coarses, coarse_of, args):
    """Per-coarse accuracy using prototype logit argmax + Hungarian matching."""
    model.eval()
    all_preds, all_labels, all_coarse = [], [], []

    for images, label, _ in loader:
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
        all_preds.append(logits.argmax(1).cpu().numpy())
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


def test_constrained(model, loader, epoch, save_name, allocation, coarse_of, args):
    """
    Global evaluation with per-coarse logit masking.

    Since coarse labels are known for every sample, we restrict the argmax to only the
    fine-class prototypes belonging to each sample's coarse before computing accuracy.
    For known coarses: only the labeled prototype indices for that coarse are visible.
    For unknown coarses: only that coarse's unknown prototype block is visible.
    """
    model.eval()
    nlc = args.num_labeled_classes

    # Inverse map: known coarse -> list of labeled class indices in that coarse
    coarse_to_labeled = {}
    for i in range(nlc):
        c = coarse_of[i]
        coarse_to_labeled.setdefault(c, []).append(i)

    preds, targets, old_new_mask = [], [], np.array([])

    for images, label, _ in tqdm(loader):
        images = images.cuda(non_blocking=True)
        lbl_np = label.numpy()
        with torch.no_grad():
            _, logits = model(images)

        logits = logits.clone()
        for i, l in enumerate(lbl_np):
            c = int(coarse_of[int(l)])
            allowed = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            if c in allocation:
                s, e = allocation[c]
                allowed[nlc + s : nlc + e] = True
            else:
                for idx in coarse_to_labeled[c]:
                    allowed[idx] = True
            logits[i] = logits[i].masked_fill(~allowed, -1e9)

        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(lbl_np)
        old_new_mask = np.append(old_new_mask, np.array([
            x.item() in range(nlc) for x in label
        ]))

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    results = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=old_new_mask,
        eval_funcs=args.eval_funcs, save_name=save_name,
        T=epoch, writer=args.writer, print_output=True,
    )
    return results[args.eval_funcs[0]]


def test_kmeans_per_coarse_separate(backbone, loader, epoch,
                                     full_allocation, known_coarses, coarse_of, args):
    """Independent KMeans per coarse on backbone features (pure feature geometry, ignores head)."""
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
    mean_all     = float(np.mean(known_accs + unknown_accs)) if (known_accs or unknown_accs) else float('nan')
    print(f'  [Separate] Macro Known {mean_known:.4f} | Unknown {mean_unknown:.4f} | All {mean_all:.4f}')
    if args.writer is not None:
        args.writer.add_scalar('PerCoarseSep/Known',   mean_known,   epoch)
        args.writer.add_scalar('PerCoarseSep/Unknown', mean_unknown, epoch)
        args.writer.add_scalar('PerCoarseSep/All',     mean_all,     epoch)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, backbone, train_loader, test_loader, unlabelled_loader,
          allocation, total_k_unknown, coarse_of,
          full_allocation, known_coarses, args):
    optimizer = SGD(get_params_groups(model), lr=args.lr, momentum=0.9,
                    weight_decay=args.weight_decay)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                eta_min=args.lr * 1e-3)

    warmup_epochs = min(args.warmup_teacher_temp_epochs, args.epochs)
    cluster_criterion = DistillLoss(
        warmup_epochs, args.epochs, args.n_views,
        args.warmup_teacher_temp, args.teacher_temp,
    )
    sup_con_crit = SupConLoss()

    nlc = args.num_labeled_classes
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, coarse_labels, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels  = class_labels.cuda(non_blocking=True)
            mask_lab       = mask_lab.cuda(non_blocking=True).bool()
            coarse_labels  = coarse_labels.cuda(non_blocking=True)
            images         = torch.cat(images, dim=0).cuda(non_blocking=True)

            batch_size_local = class_labels.size(0)
            coarse_unlab = coarse_labels[~mask_lab]

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = model(images)
                teacher_out = student_out.detach()

                # --- Per-coarse masking of prototype logits (unlabeled samples only) ---
                proto_mask = build_unlab_proto_mask(
                    coarse_unlab, allocation, nlc, total_k_unknown
                )
                unlab_idx = (~mask_lab).nonzero(as_tuple=True)[0]

                student_out_masked = student_out.clone()
                teacher_out_masked = teacher_out.clone()
                for view_offset in (0, batch_size_local):
                    idx = unlab_idx + view_offset
                    student_out_masked[idx] = mask_joint_logits(
                        student_out[idx], proto_mask
                    )
                    teacher_out_masked[idx] = mask_joint_logits(
                        teacher_out[idx], proto_mask
                    )

                # Supervised CE loss (labeled only, no masking needed)
                sup_logits = torch.cat(
                    [f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0
                )
                sup_labels = torch.cat(
                    [class_labels[mask_lab] for _ in range(2)], dim=0
                )
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # Distillation loss on masked logits
                cluster_loss = cluster_criterion(
                    student_out_masked, teacher_out_masked, epoch
                )
                avg_probs = (student_out_masked / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = (
                    -torch.sum(torch.log(avg_probs ** (-avg_probs)))
                    + math.log(float(len(avg_probs)))
                )
                cluster_loss += args.memax_weight * me_max_loss

                # --- NTXent in projection space (optionally within-coarse) ---
                if args.contrast_unlabel_only:
                    f1u = student_proj.chunk(2)[0][~mask_lab]
                    f2u = student_proj.chunk(2)[1][~mask_lab]
                    con_feats = torch.cat([f1u, f2u], dim=0)
                    coarse_2v = torch.cat(
                        [coarse_labels[~mask_lab], coarse_labels[~mask_lab]], dim=0
                    )
                else:
                    con_feats = student_proj
                    coarse_2v = torch.cat([coarse_labels, coarse_labels], dim=0)

                if args.ntxent_mode == 'global':
                    contrastive_logits, contrastive_labels_ce = info_nce_logits(student_proj)
                    contrastive_loss = nn.CrossEntropyLoss()(
                        contrastive_logits, contrastive_labels_ce
                    )
                else:  # 'coarse'
                    con_feats_norm = F.normalize(con_feats, dim=-1)
                    contrastive_loss = info_nce_logits_coarse(con_feats_norm, coarse_2v)

                # --- Fine-class SupCon (labeled only, optionally within-coarse) ---
                f1_lab, f2_lab = [f[mask_lab] for f in student_proj.chunk(2)]
                student_proj_sup = torch.cat(
                    [f1_lab.unsqueeze(1), f2_lab.unsqueeze(1)], dim=1
                )
                student_proj_sup = F.normalize(student_proj_sup, dim=-1)

                if args.supcon_mode == 'global':
                    sup_con_loss = sup_con_crit(
                        student_proj_sup, labels=class_labels[mask_lab]
                    )
                else:  # 'within_coarse'
                    sup_con_loss = within_coarse_sup_con_loss(
                        student_proj_sup, class_labels[mask_lab], coarse_labels[mask_lab],
                        temperature=sup_con_crit.temperature,
                    )

                # --- Coarse SupCon over all samples (optional) ---
                if args.use_coarse_supcon:
                    f1_all, f2_all = student_proj.chunk(2)
                    coarse_con_feats = torch.cat([
                        F.normalize(f1_all, dim=-1).unsqueeze(1),
                        F.normalize(f2_all, dim=-1).unsqueeze(1),
                    ], dim=1)
                    coarse_sup_loss = sup_con_crit(coarse_con_feats, labels=coarse_labels)
                else:
                    coarse_sup_loss = 0.0

                loss = (
                    (1 - args.sup_weight) * cluster_loss
                    + args.sup_weight * cls_loss
                    + (1 - args.sup_weight) * contrastive_loss
                    + args.sup_weight * sup_con_loss
                    + args.coarse_supcon_weight * coarse_sup_loss
                )

            total_loss += loss.item()
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch}/{args.epochs}][{batch_idx}/{len(train_loader)}] '
                      f'loss {loss.item():.5f}')

        print(f'Train Epoch: {epoch} Avg Loss: {total_loss / len(train_loader):.4f}')
        scheduler.step()

        # Test-set eval every epoch — constrained (checkpoint selection) + unconstrained
        all_acc_c, old_acc_c, new_acc_c = test_constrained(
            model, test_loader, epoch=epoch,
            save_name='Test ACC Constrained',
            allocation=allocation, coarse_of=coarse_of, args=args,
        )
        print(f'Test (constrained)  : All {all_acc_c:.4f} | Old {old_acc_c:.4f} | New {new_acc_c:.4f}')

        all_acc, old_acc, new_acc = test(
            model, test_loader, epoch=epoch, save_name='Test ACC', args=args,
        )
        print(f'Test (unconstrained): All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')

        projector = model[1]
        torch.save(backbone.state_dict(), args.model_path)
        torch.save(projector.state_dict(), os.path.join(args.model_dir, 'proj_head.pt'))
        if old_acc_c >= best_acc:
            best_acc = old_acc_c
            torch.save(backbone.state_dict(),
                       os.path.join(args.model_dir, 'model_best.pt'))
            torch.save(projector.state_dict(),
                       os.path.join(args.model_dir, 'proj_head_best.pt'))

        # Expensive evals only at the report interval
        if (epoch + 1) % args.train_report_interval == 0:
            train_all_c, train_old_c, train_new_c = test_constrained(
                model, unlabelled_loader, epoch=epoch,
                save_name='Train ACC Constrained',
                allocation=allocation, coarse_of=coarse_of, args=args,
            )
            print(f'Train (constrained): All {train_all_c:.4f} | Old {train_old_c:.4f} | New {train_new_c:.4f}')
            print('Per-coarse evaluation (global head):')
            test_per_coarse(model, unlabelled_loader, epoch,
                            full_allocation, known_coarses, coarse_of, args)
            print('Per-coarse evaluation (separate KMeans per coarse):')
            test_kmeans_per_coarse_separate(backbone, unlabelled_loader, epoch,
                                             full_allocation, known_coarses, coarse_of, args)


def test(model, loader, epoch, save_name, args):
    model.eval()
    preds, targets, mask = [], [], np.array([])

    for images, label, _ in tqdm(loader):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(label.cpu().numpy())
        mask = np.append(mask, np.array([
            x.item() in range(len(args.train_classes)) for x in label
        ]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    results = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask,
        eval_funcs=args.eval_funcs, save_name=save_name,
        T=epoch, writer=args.writer, print_output=True,
    )
    first_func = args.eval_funcs[0]
    return results[first_func]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hierarchical SimGCD training (SelEx-integrated)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / model
    parser.add_argument('--dataset_name', type=str, default='scars')
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--num_novel_classes', type=int, default=None)
    parser.add_argument('--num_unlabeled_classes', type=int, default=None,
                        help='Estimated unlabeled class count. Falls back to true split count.')
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    # Optimisation
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--n_views', type=int, default=2)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--sup_weight', type=float, default=0.35)
    # SimGCD-specific
    parser.add_argument('--memax_weight', type=float, default=2.0)
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.07)
    parser.add_argument('--teacher_temp', type=float, default=0.04)
    parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=30)
    # Loss mode flags (matching gcd_hier)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--ntxent_mode', type=str, default='coarse',
                        choices=['global', 'coarse'],
                        help='global: standard NTXent; coarse: within-coarse NTXent')
    parser.add_argument('--supcon_mode', type=str, default='global',
                        choices=['global', 'within_coarse'],
                        help='global: all negatives; within_coarse: same-coarse negatives only')
    parser.add_argument('--use_coarse_supcon', type=str2bool, default=False,
                        help='Add coarse-label SupCon loss over all samples')
    parser.add_argument('--coarse_supcon_weight', type=float, default=0.25,
                        help='Weight for the coarse SupCon term')
    parser.add_argument('--use_coarse_sampler', type=str2bool, default=False,
                        help='Inverse-frequency coarse-balanced batch sampler')
    # Hierarchical-specific
    parser.add_argument('--coarse_labels_path', type=str, required=True,
                        help='JSON mapping fine_class_idx -> coarse_class_idx')
    parser.add_argument('--k_results_path', type=str, default=None,
                        help='JSON with per-coarse K: {coarse_id: k_i}. '
                             'Uses oracle K if not provided.')
    parser.add_argument('--hier_split_type', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_report_interval', type=int, default=200)
    # Eval / logging
    parser.add_argument('--eval_funcs', nargs='+', default=['v2'])
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--exp_id',   type=str, default=None,
                        help='Fixed experiment ID for the log dir (default: use timestamp)')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes = len(args.train_classes)
    if args.num_unlabeled_classes is None:
        args.num_unlabeled_classes = len(args.unlabeled_classes)

    # --- Coarse label setup ---
    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    if args.k_results_path and os.path.exists(args.k_results_path):
        with open(args.k_results_path) as f:
            k_per_coarse = {int(k): v for k, v in json.load(f).items()}
    else:
        k_per_coarse = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)

    allocation, total_k_unknown = build_proto_allocation(k_per_coarse)

    # full_allocation: {coarse_id: int_k} for both unknown and known coarses (used in eval)
    known_coarses   = {coarse_of[i] for i in range(args.num_labeled_classes)}
    known_coarse_k  = Counter(coarse_of[i] for i in range(args.num_labeled_classes))
    full_allocation = {int(c): int(k) for c, k in k_per_coarse.items()}
    full_allocation.update(known_coarse_k)

    print(f'K: labeled={args.num_labeled_classes} | unknown={total_k_unknown} | '
          f'total={args.num_labeled_classes + total_k_unknown}')
    print(f'Known coarses: {len(known_coarses)} | Unknown coarses: {len(k_per_coarse)}')

    init_experiment(args, runner_name=['simgcd_hier', 'dinov2', args.dataset_name], exp_id=args.exp_id)

    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + total_k_unknown

    print(f'mlp_out_dim = {args.num_labeled_classes} (labeled) + '
          f'{total_k_unknown} (unknown) = {args.mlp_out_dim}')

    # ---- Backbone ----
    backbone = vits.__dict__['vit_base']()
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

    # ---- Model ----
    projector = DINOHead(
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    )
    model = nn.Sequential(backbone, projector).to(device)

    train(model, backbone, train_loader, test_loader, unlabelled_loader,
          allocation, total_k_unknown, coarse_of,
          full_allocation, known_coarses, args)


if __name__ == '__main__':
    main()
