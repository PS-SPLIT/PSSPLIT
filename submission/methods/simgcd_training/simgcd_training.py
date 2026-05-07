"""
simgcd_training.py — SimGCD training method integrated into the SelEx pipeline.

Uses the SimGCD 4-component loss (supervised CE + distillation clustering +
entropy regularisation + supervised contrastive) with SelEx's DINOv2 backbone,
data loading, and experiment management utilities.

Run via run_pipeline.py with --training_method simgcd, or standalone:
    python -m methods.simgcd_training.simgcd_training --dataset_name cifar10 ...
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import dino_pretrain_path2, dino_v1_vitb16_path, exp_root
from data.augmentations import get_transform
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from models import vision_transformer2 as vits
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import init_experiment, str2bool


# ---------------------------------------------------------------------------
# SimGCD model components
# (copied from SimGCD-main/model.py; kept self-contained to avoid cross-repo deps)
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return x_proj, logits


class ContrastiveLearningViewGenerator:
    """Produce n_views augmented copies of each image."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for _ in range(self.n_views)]
        return [self.base_transform[i](x) for i in range(self.n_views)]


class SupConLoss(nn.Module):
    """Supervised contrastive loss (from SimGCD/model.py)."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('features must be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class DistillLoss(nn.Module):
    """Self-distillation clustering loss with teacher temperature schedule."""
    def __init__(self, warmup_teacher_temp_epochs, nepochs, ncrops=2,
                 warmup_teacher_temp=0.07, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1).detach().chunk(2)

        total_loss, n_loss_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                total_loss += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                n_loss_terms += 1
        return total_loss / n_loss_terms


def info_nce_logits(features, n_views=2, temperature=1.0):
    device = features.device
    b_ = features.size(0) // 2
    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    return logits, labels


def get_params_groups(model):
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.bias') or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, backbone, train_loader, unlabelled_loader, args):
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

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels = class_labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = model(images)
                teacher_out = student_out.detach()

                # Supervised classification loss (labeled examples only)
                sup_logits = torch.cat(
                    [f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0
                )
                sup_labels = torch.cat(
                    [class_labels[mask_lab] for _ in range(2)], dim=0
                )
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # Unsupervised clustering loss (distillation + entropy regularisation)
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = (
                    -torch.sum(torch.log(avg_probs ** (-avg_probs)))
                    + math.log(float(len(avg_probs)))
                )
                cluster_loss += args.memax_weight * me_max_loss

                # Unsupervised contrastive loss
                contrastive_logits, contrastive_labels = info_nce_logits(student_proj)
                contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss (labeled examples only)
                student_proj_sup = torch.cat(
                    [f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1
                )
                student_proj_sup = F.normalize(student_proj_sup, dim=-1)
                sup_con_loss = SupConLoss()(student_proj_sup, labels=class_labels[mask_lab])

                loss = ((1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                        + (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss)

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

        # Evaluate on unlabelled training examples (transductive)
        all_acc, old_acc, new_acc = test(
            model, unlabelled_loader, epoch=epoch,
            save_name='Train ACC Unlabelled', args=args,
        )
        print(f'Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')

        # Save latest backbone (bare state_dict for SelEx extract_features.py compatibility)
        projector = model[1]
        torch.save(backbone.state_dict(), args.model_path)
        torch.save(projector.state_dict(), os.path.join(args.model_dir, 'proj_head.pt'))
        if old_acc >= best_acc:
            best_acc = old_acc
            torch.save(backbone.state_dict(),
                       os.path.join(args.model_dir, 'model_best.pt'))
            torch.save(projector.state_dict(),
                       os.path.join(args.model_dir, 'proj_head_best.pt'))


def test(model, loader, epoch, save_name, args):
    model.eval()
    if len(loader.dataset) == 0:
        return float('nan'), float('nan'), float('nan')
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
        description='SimGCD training (SelEx-integrated)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / model
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--hier_split_type', type=str, default=None)
    parser.add_argument('--num_novel_classes', type=int, default=None)
    parser.add_argument('--num_unlabeled_classes', type=int, default=None,
                        help='Estimated unlabeled class count from pipeline. '
                             'Falls back to true split count if not provided.')
    parser.add_argument('--warmup_model_dir', type=str, default=None,
                        help='Directory containing model_best.pt for backbone init.')
    parser.add_argument('--use_dino_v1', type=str2bool, default=False,
                        help='Use DINOv1 vitb16 backbone instead of DINOv2 vitb14_reg4.')
    parser.add_argument('--seed', type=int, default=1)
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
    # Eval / logging
    parser.add_argument('--eval_funcs', nargs='+', default=['v2'])
    parser.add_argument('--train_report_interval', type=int, default=1)
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

    backbone_tag = 'dinov1' if args.use_dino_v1 else 'dinov2'
    init_experiment(args, runner_name=['simgcd', backbone_tag, args.dataset_name], exp_id=args.exp_id)

    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ---- Backbone ----
    if args.use_dino_v1:
        print('Loading DINOv1 vitb16 backbone')
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        state_dict = torch.load(dino_v1_vitb16_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
    else:
        print('Loading DINOv2 vitb14-reg4 backbone')
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

    train_dataset, test_dataset, unlabelled_train_examples_test, _ = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )

    # Balance labeled / unlabeled examples in each batch
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
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
    unlabelled_loader = DataLoader(
        unlabelled_train_examples_test, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False,
    )

    # ---- Model (backbone + classification head) ----
    projector = DINOHead(
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    )
    model = nn.Sequential(backbone, projector).to(device)

    train(model, backbone, train_loader, unlabelled_loader, args)


if __name__ == '__main__':
    main()
