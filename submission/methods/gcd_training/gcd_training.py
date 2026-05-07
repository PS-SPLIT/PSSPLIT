"""
gcd_training.py — GCD paper training method integrated into the SelEx pipeline.

Faithful port of the "Generalized Category Discovery" (Vaze et al. 2022)
contrastive training, using SelEx's DINOv2 backbone, data loading, and
experiment management utilities.

Run via run_pipeline.py with --training_method gcd_paper, or standalone:
    python -m methods.gcd_training.gcd_training --dataset_name cifar10 ...
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import dino_pretrain_path2, dino_v1_vitb16_path, exp_root
from data.augmentations import get_transform
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from models import vision_transformer2 as vits
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import AverageMeter, get_mean_lr, init_experiment, str2bool


# ---------------------------------------------------------------------------
# GCD model components
# (inlined from generalized-category-discovery-main; kept self-contained)
# ---------------------------------------------------------------------------

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning (Khosla et al., 2020).
    Also supports unsupervised contrastive loss (SimCLR) when no labels given.
    """
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


class ContrastiveLearningViewGenerator:
    """Produce n_views augmented copies of each image."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


def info_nce_logits(features, args):
    """NTXent logits and labels for contrastive loss."""
    device = features.device
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for _ in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / args.temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    return logits, labels


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def test_kmeans(backbone, loader, epoch, save_name, args):
    """Evaluate backbone features with KMeans clustering."""
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
    n_clusters = args.num_labeled_classes + args.num_unlabeled_classes
    preds = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(all_feats)

    results = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask,
        eval_funcs=args.eval_funcs, save_name=save_name,
        T=epoch, writer=args.writer, print_output=True,
    )
    first_func = args.eval_funcs[0]
    return results[first_func]


def train(projection_head, backbone, train_loader, test_loader,
          unlabelled_loader, args):
    optimizer = SGD(
        list(projection_head.parameters()) + list(backbone.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                eta_min=args.lr * 1e-3)
    sup_con_crit = SupConLoss()
    best_test_acc_old = 0.0

    for epoch in range(args.epochs):
        backbone.train()
        projection_head.train()
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels = class_labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            # Extract features and project
            features = backbone(images)
            features = projection_head(features)
            features = F.normalize(features, dim=-1)

            # Contrastive loss (all or unlabeled-only)
            if args.contrast_unlabel_only:
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                con_feats = features

            contrastive_logits, contrastive_labels = info_nce_logits(con_feats, args)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # Supervised contrastive loss (labeled only)
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            loss = ((1 - args.sup_con_weight) * contrastive_loss
                    + args.sup_con_weight * sup_con_loss)

            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))
            loss_record.update(loss.item(), class_labels.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f} '
              f'| Seen Class Acc: {train_acc_record.avg:.4f}')
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        scheduler.step()

        with torch.no_grad():
            if (epoch + 1) % args.train_report_interval == 0:
                print('Testing on unlabelled examples in the training data...')
                all_acc, old_acc, new_acc = test_kmeans(
                    backbone, unlabelled_loader, epoch=epoch,
                    save_name='Train ACC Unlabelled', args=args,
                )
            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_kmeans(
                backbone, test_loader, epoch=epoch,
                save_name='Test ACC', args=args,
            )

        if (epoch + 1) % args.train_report_interval == 0:
            print(f'Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
        print(f'Test  Accuracies: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}')

        # Save latest backbone checkpoint
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
        description='GCD paper training (SelEx-integrated)',
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
    # GCD-specific
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    # Eval / logging
    parser.add_argument('--eval_funcs', nargs='+', default=['v2'])
    parser.add_argument('--train_report_interval', type=int, default=200)
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
    init_experiment(args, runner_name=['gcd', backbone_tag, args.dataset_name], exp_id=args.exp_id)

    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

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

    backbone = backbone.to(device)

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
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False,
    )
    unlabelled_loader = DataLoader(
        unlabelled_train_examples_test, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False,
    )

    # ---- Projection head (65536-dim contrastive embedding) ----
    projection_head = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    ).to(device)

    train(projection_head, backbone, train_loader, test_loader,
          unlabelled_loader, args)


if __name__ == '__main__':
    main()
