"""
uno_training.py — UNO+ training method integrated into the SelEx pipeline.

Implements UNO+ (Generalized Category Discovery variant of UNO, Fini et al. 2021).
Original UNO is designed for Novel Class Discovery (NCD) where unlabeled classes are
entirely disjoint from labeled classes.  UNO+ adapts it for GCD by running
Sinkhorn-Knopp over the *concatenated* labeled + unlabeled logits, so that unlabeled
samples can be pseudo-labeled as any class (old or new).

Reference (GCD paper, Vaze et al. 2022):
    "To adapt this mechanism, we simply concatenate both the labelled and unlabelled
    heads, thus allowing generated pseudo-labels for the unlabelled samples to belong
    to any class in the dataset."

Run via run_pipeline.py with --training_method uno_plus, or standalone:
    python -m methods.uno_training.uno_training --dataset_name cifar10 ...
"""

import argparse
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
from project_utils.general_utils import AverageMeter, get_mean_lr, init_experiment, str2bool


# ---------------------------------------------------------------------------
# Inlined from UNO-main/utils/nets.py (unchanged)
# ---------------------------------------------------------------------------

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()
        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_prototypes,
                 num_heads, num_hidden_layers=1):
        super().__init__()
        self.num_heads = num_heads
        self.projectors = nn.ModuleList([
            MLP(input_dim, hidden_dim, output_dim, num_hidden_layers)
            for _ in range(num_heads)
        ])
        self.prototypes = nn.ModuleList([
            Prototypes(output_dim, num_prototypes)
            for _ in range(num_heads)
        ])
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        # Returns ([H, B, num_prototypes], [H, B, proj_dim])
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


# ---------------------------------------------------------------------------
# Inlined from UNO-main/utils/sinkhorn_knopp.py (unchanged)
# ---------------------------------------------------------------------------

class SinkhornKnopp(nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.num_iters):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()


# ---------------------------------------------------------------------------
# UNO+ model (backbone-free — heads only)
# ---------------------------------------------------------------------------

class UNOPlusModel(nn.Module):
    """Classification heads for UNO+.  Backbone is kept separate."""

    def __init__(self, feat_dim, num_labeled, num_unlabeled, proj_dim=256,
                 hidden_dim=2048, overcluster_factor=3, num_heads=5,
                 num_hidden_layers=1):
        super().__init__()
        self.head_lab = Prototypes(feat_dim, num_labeled)
        if num_unlabeled > 0:
            self.head_unlab = MultiHead(
                feat_dim, hidden_dim, proj_dim, num_unlabeled,
                num_heads, num_hidden_layers,
            )
            self.head_unlab_over = MultiHead(
                feat_dim, hidden_dim, proj_dim, num_unlabeled * overcluster_factor,
                num_heads, num_hidden_layers,
            )
        else:
            # Supervised-only mode: no unlabeled head needed.
            self.head_unlab = None
            self.head_unlab_over = None

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if self.head_unlab is not None:
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward(self, feats):
        """
        feats: [B, feat_dim]
        Returns:
            logits_lab:       [B, num_labeled]
            logits_unlab:     [H, B, num_unlabeled]  or None if supervised-only
            logits_unlab_over:[H, B, num_unlabeled * overcluster_factor]  or None
        """
        logits_lab = self.head_lab(F.normalize(feats, dim=1))
        if self.head_unlab is not None:
            logits_unlab, _ = self.head_unlab(feats)
            logits_unlab_over, _ = self.head_unlab_over(feats)
        else:
            logits_unlab = None
            logits_unlab_over = None
        return logits_lab, logits_unlab, logits_unlab_over


# ---------------------------------------------------------------------------
# Loss helpers (inlined from UNO's Discoverer, adapted for plain PyTorch)
# ---------------------------------------------------------------------------

def cross_entropy_loss(preds, targets, temperature):
    """
    Soft cross-entropy with temperature scaling.
    preds, targets: [H, B, C]
    Returns: [H]  (mean over batch for each head)
    """
    preds = F.log_softmax(preds / temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


def swapped_prediction(logits_list, targets_list, temperature):
    """
    Swapped prediction loss across views.
    logits_list:  list of V tensors, each [H, B, C]
    targets_list: list of V tensors, each [H, B, C]  (targets for large crops only)
    Returns: [H]
    """
    num_large = len(targets_list)
    num_crops = len(logits_list)
    loss = 0
    for v in range(num_large):
        for ov in range(num_crops):
            if ov != v:
                loss = loss + cross_entropy_loss(logits_list[ov], targets_list[v], temperature)
    return loss / (num_large * (num_crops - 1))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class ContrastiveLearningViewGenerator:
    """Produce n_views augmented copies of each image (for train loader)."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


def test_uno(backbone, model, loader, epoch, save_name, best_head, args):
    """Evaluate with argmax over concatenated labeled + best-unlabeled-head logits."""
    backbone.eval()
    model.eval()
    if len(loader.dataset) == 0:
        return float('nan'), float('nan'), float('nan')
    all_preds, targets, mask = [], np.array([]), np.array([])

    for images, label, _ in tqdm(loader):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feats = backbone(images)
            logits_lab, logits_unlab, _ = model(feats)
            if logits_unlab is not None:
                logits_inc = torch.cat([logits_lab, logits_unlab[best_head]], dim=-1)
            else:
                logits_inc = logits_lab
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(backbone, model, train_loader, test_loader, unlabelled_loader, args):
    sk = SinkhornKnopp(num_iters=args.num_iters_sk, epsilon=args.epsilon_sk).cuda()
    nlc = args.num_labeled_classes

    optimizer = SGD(
        list(backbone.parameters()) + list(model.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                eta_min=args.lr * 1e-3)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    best_test_acc_old = 0.0
    best_head = 0

    for epoch in range(args.epochs):
        backbone.train()
        model.train()
        loss_record = AverageMeter()
        loss_per_head = torch.zeros(args.num_heads).cuda()

        for batch in tqdm(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0].cuda().bool()
            if args.fully_unsupervised:
                mask_lab = torch.zeros_like(mask_lab)
            class_labels = class_labels.cuda(non_blocking=True)
            views = [v.cuda(non_blocking=True) for v in images]
            B = class_labels.shape[0]

            # Normalize prototypes before each forward pass
            model.normalize_prototypes()

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # Forward backbone for each view → feats: [B, feat_dim]
                feats_list = [backbone(v) for v in views]

                # Forward heads for each view
                logits_lab_list, logits_unlab_list, logits_unlab_over_list = [], [], []
                for feats in feats_list:
                    ll, lu, luo = model(feats)
                    logits_lab_list.append(ll)        # [B, nlc]
                    logits_unlab_list.append(lu)      # [H, B, n_unlab]
                    logits_unlab_over_list.append(luo)  # [H, B, n_unlab*of]

                H = args.num_heads

                # Build combined logits [H, B, nlc + n_unlab] for each view.
                # When head_unlab is None (supervised-only), use labeled logits only.
                logits_list, logits_over_list = [], []
                for ll, lu, luo in zip(logits_lab_list, logits_unlab_list, logits_unlab_over_list):
                    ll_exp = ll.unsqueeze(0).expand(H, -1, -1)  # [H, B, nlc]
                    if lu is not None:
                        logits_list.append(torch.cat([ll_exp, lu], dim=-1))
                        logits_over_list.append(torch.cat([ll_exp, luo], dim=-1))
                    else:
                        logits_list.append(ll_exp)
                        logits_over_list.append(ll_exp)

                # One-hot targets for labeled samples
                targets_lab = F.one_hot(
                    class_labels[mask_lab], num_classes=nlc
                ).float()

                # Build soft targets (UNO+ modification: SK over full concat logits)
                V = len(views)
                targets = torch.zeros_like(logits_list[0]).float().unsqueeze(0).expand(V, -1, -1, -1).clone()
                targets_over = torch.zeros_like(logits_over_list[0]).float().unsqueeze(0).expand(V, -1, -1, -1).clone()

                for v in range(V):
                    for h in range(H):
                        # Labeled samples: one-hot in labeled positions
                        targets[v, h, mask_lab, :nlc] = targets_lab
                        targets_over[v, h, mask_lab, :nlc] = targets_lab
                        # Unlabeled samples: SK over logits (mode controls which)
                        if mask_lab.logical_not().any():
                            if args.ncd_mode and logits_unlab_list[v] is not None:
                                # NCD: SK over head_unlab logits only; unlabeled samples
                                # do not predict into the labeled-head portion.
                                targets[v, h, ~mask_lab, nlc:] = sk(
                                    logits_unlab_list[v][h, ~mask_lab]
                                )
                                targets_over[v, h, ~mask_lab, nlc:] = sk(
                                    logits_unlab_over_list[v][h, ~mask_lab]
                                )
                            else:
                                # GCD+ (default): SK over FULL concatenated logits
                                targets[v, h, ~mask_lab, :] = sk(
                                    logits_list[v][h, ~mask_lab]
                                )
                                targets_over[v, h, ~mask_lab, :] = sk(
                                    logits_over_list[v][h, ~mask_lab]
                                )

                # Swapped prediction losses → [H]
                loss_cluster = swapped_prediction(
                    logits_list, [targets[v] for v in range(V)], args.temperature
                )
                loss_overcluster = swapped_prediction(
                    logits_over_list, [targets_over[v] for v in range(V)], args.temperature
                )

                # Accumulate per-head loss for best-head selection
                loss_per_head += loss_cluster.detach()

                loss = (loss_cluster.mean() + loss_overcluster.mean()) / 2
                loss_record.update(loss.item(), B)

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

        best_head = loss_per_head.argmin().item()
        print(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f} '
              f'| Best Head: {best_head} | LR: {get_mean_lr(optimizer):.6f}')
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        scheduler.step()

        with torch.no_grad():
            all_acc, old_acc, new_acc = test_uno(
                backbone, model, unlabelled_loader, epoch=epoch,
                save_name='Train ACC Unlabelled', best_head=best_head, args=args,
            )
            all_acc_test, old_acc_test, new_acc_test = test_uno(
                backbone, model, test_loader, epoch=epoch,
                save_name='Test ACC', best_head=best_head, args=args,
            )

        print(f'Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
        print(f'Test  Accuracies: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}')

        # Save latest backbone + heads (bare state_dicts for SelEx extract_features.py compatibility)
        torch.save(backbone.state_dict(), args.model_path)
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'uno_heads.pt'))
        torch.save({'best_head': best_head}, os.path.join(args.model_dir, 'best_head.pt'))

        if old_acc_test > best_test_acc_old:
            best_test_acc_old = old_acc_test
            print(f'Best old-class ACC on test set: {old_acc_test:.4f}')
            torch.save(backbone.state_dict(),
                       os.path.join(args.model_dir, 'model_best.pt'))
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'uno_heads_best.pt'))
            torch.save({'best_head': best_head}, os.path.join(args.model_dir, 'best_head_best.pt'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='UNO+ training (SelEx-integrated)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / model
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--hier_split_type', type=str, default=None)
    parser.add_argument('--ncd_mode', type=str2bool, default=False,
                        help='NCD-style SK: applied over head_unlab logits for unlabeled '
                             'samples only, not over concatenated logits (GCD+ default).')
    parser.add_argument('--fully_unsupervised', type=str2bool, default=False,
                        help='Override mask_lab to all-False so all samples use SK targets; '
                             'removes all GT-label supervision.')
    parser.add_argument('--num_novel_classes', type=int, default=None)
    parser.add_argument('--num_unlabeled_classes', type=int, default=None,
                        help='Estimated unlabeled class count from pipeline.')
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    # Optimisation
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--transform', type=str, default='imagenet')
    # UNO+-specific
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Softmax temperature for swapped-prediction loss')
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--overcluster_factor', type=int, default=3)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--epsilon_sk', type=float, default=0.05)
    parser.add_argument('--num_iters_sk', type=int, default=3)
    # Eval / logging
    parser.add_argument('--eval_funcs', nargs='+', default=['v2'])
    parser.add_argument('--train_report_interval', type=int, default=1)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--exp_id',   type=str, default=None,
                        help='Fixed experiment ID for the log dir (default: use timestamp)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_dino_v1', type=str2bool, default=False,
                        help='Use DINOv1 vitb16 instead of DINOv2 vitb14_reg4.')
    parser.add_argument('--fp16', action='store_true', default=False)

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
    init_experiment(args, runner_name=['uno', backbone_tag, args.dataset_name], exp_id=args.exp_id)

    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768

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
    train_transform = ContrastiveLearningViewGenerator(train_transform, n_views=2)

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

    # ---- UNO+ model (heads only) ----
    model = UNOPlusModel(
        feat_dim=args.feat_dim,
        num_labeled=args.num_labeled_classes,
        num_unlabeled=args.num_unlabeled_classes,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        overcluster_factor=args.overcluster_factor,
        num_heads=args.num_heads,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    train(backbone, model, train_loader, test_loader, unlabelled_loader, args)


if __name__ == '__main__':
    main()
