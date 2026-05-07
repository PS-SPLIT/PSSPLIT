"""
contrastive_coarse_training.py — Coarse-label SelEx contrastive training.

Based on methods/contrastive_training/contrastive_training.py. Differs from the existing
contrastive_hier_training.py in how the pseudo-label hierarchy is built:

  Level 0 (finest, K = 2 × num_coarses):
    Within-coarse 2-split, initialised from:
      - 2+ known fine classes in coarse: use labeled fine-class means as centres
      - 1 known fine class or unknown coarse: KMeans(2) within the coarse
  Level 1 (K = num_coarses):
    Ground-truth coarse labels assigned directly (no KMeans).
  Level 2+ (K = num_coarses/2, …):
    Standard halving via KMeans on the coarse prototypes.

The supervised contrastive loss also uses GT coarse labels:
  - Full features → GT fine SupCon (labeled samples only)        weight 1
  - dim/2 features → GT coarse SupCon (ALL samples)              weight 0.5
  - dim/4 features → level-0 pseudo-labels (2×coarses split)     weight 0.25
  - dim/8+ features → further coarsened pseudo-labels            weight 0.125, …

The unsupervised NTXent is the standard global version (no within-coarse restriction).

Designed for datasets where each coarse has ≈1–2 fine classes (aves_small, inat_subspecies).

Run via run_pipeline_hier.py with --training_method gcd_selex_coarse, or standalone:
    python -m methods.contrastive_hier_training.contrastive_coarse_training \
        --dataset_name aves_small \
        --coarse_labels_path data/coarse_labels/aves_small.json \
        --hier_split_type 5050 \
        --epochs 5
"""

import argparse
import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import exp_root, dino_pretrain_path2, dino_v1_vitb16_path
from data.augmentations import get_transform
from data.data_utils import (CoarseMergedDataset, build_coarse_of, load_coarse_map,
                             oracle_k_per_coarse)
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from kmeans_pytorch import kmeans
from models import vision_transformer2 as vits
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.cluster_utils import AverageMeter
from project_utils.general_utils import (get_mean_lr, init_experiment,
                                          str2bool, get_dino_head_weights)

warnings.filterwarnings("ignore")

device = torch.device('cuda:0')


# ---------------------------------------------------------------------------
# Loss utilities  (unchanged from contrastive_training.py)
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity, smoothing=0.5):
        target_smooth = (F.one_hot(target, input.size(1)).float() * (1 - smoothing)
                         + smoothing * similarity)
        return torch.nn.CrossEntropyLoss()(input, target_smooth)


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning (Khosla et al. 2020).
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, is_code=False):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dims')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
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

        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist = -dist / (dist.sum(dim=1) + 1e-10)
        else:
            dist = -torch.cdist(anchor_feature, contrast_feature)

        anchor_dot_contrast = torch.div(dist, self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class ContrastiveLearningViewGenerator:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


def info_nce_logits(features, confusion_factor, args, is_code=False):
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for _ in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    features = F.normalize(features, dim=1)
    if is_code:
        dist = torch.cdist(features, features, p=2)
        similarity_matrix = -dist / (dist.sum(dim=1) + 1e-10)
    else:
        similarity_matrix = -torch.cdist(features, features)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs = confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs = confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels, log_confs


# ---------------------------------------------------------------------------
# Plotting helpers  (unchanged)
# ---------------------------------------------------------------------------

def plot_a_loss(x, y, name):
    plt.figure(figsize=(10, 10))
    plt.title(name + ' Loss')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(x, y, 'cornflowerblue', linewidth=3)
    plt.xlim(0, x.shape[0] - 1)
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join('Plots', name + '_Loss.png'))
    plt.close()


def plot_acc(x, old, new, all_, name):
    plt.figure(figsize=(10, 10))
    plt.title(f'Accuracy using {name} for clustering')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(x, old,  color='cornflowerblue', label='Old', linewidth=3)
    plt.plot(x, new,  color='orangered',      label='New', linewidth=3)
    plt.plot(x, all_, color='limegreen',      label='All', linewidth=3)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.xlim(0, x.shape[0] - 1)
    plt.locator_params(axis='both', nbins=10)
    plt.legend()
    plt.savefig(os.path.join('Plots', f'Accuracy_{name}.png'))
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation  (unchanged from non-hier version)
# ---------------------------------------------------------------------------

def test_kmeans(model, test_loader, epoch, save_name, args, Use_GPU=True):
    model.eval()
    if len(test_loader.dataset) == 0:
        return float('nan'), float('nan'), float('nan')

    all_feats, targets, mask = [], np.array([]), np.array([])
    for images, label, _ in tqdm(test_loader):
        images = images.cuda()
        feats = model(images)
        all_feats.append(F.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([
            True if x.item() in range(len(args.train_classes)) else False for x in label
        ]))

    mask = mask.astype(bool)
    all_feats = np.concatenate(all_feats)
    n_clusters = args.num_labeled_classes + args.num_unlabeled_classes

    preds, _ = kmeans(
        X=torch.from_numpy(all_feats),
        num_clusters=n_clusters, distance='euclidean',
        device=torch.device('cpu'), tqdm_flag=False,
    )
    preds = preds.numpy()

    results = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch,
        eval_funcs=args.eval_funcs, save_name=save_name,
        writer=args.writer,
        save_path=os.path.join(args.log_dir, 'results.jsonl')
    )
    all_acc, old_acc, new_acc = results[args.eval_funcs[0]]
    return all_acc, old_acc, new_acc


# ---------------------------------------------------------------------------
# Per-coarse evaluation helpers
# ---------------------------------------------------------------------------

def _cluster_acc(y_true, y_pred):
    """Hungarian-matching accuracy between true labels and cluster assignments."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    D = max(y_true.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i, j in zip(y_pred, y_true):
        w[i, j] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / len(y_true)


def test_kmeans_per_coarse(model, loader, epoch,
                            full_allocation, known_coarses, coarse_of, args):
    """Global KMeans on all features, results decomposed per coarse."""
    model.eval()
    all_feats, all_labels = [], []
    for images, label, _ in loader:
        with torch.no_grad():
            feats = model(images.cuda(non_blocking=True))
        all_feats.append(F.normalize(feats, dim=-1).cpu().numpy())
        all_labels.append(label.numpy())
    all_feats  = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    all_coarse = np.array([int(coarse_of[int(l)]) for l in all_labels])

    n_clusters = args.num_labeled_classes + args.num_unlabeled_classes
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
    mean_all     = float(np.mean(known_accs + unknown_accs)) if (known_accs or unknown_accs) else float('nan')
    print(f'  [Global] Macro Known {mean_known:.4f} | Unknown {mean_unknown:.4f} | All {mean_all:.4f}')
    if args.writer is not None:
        args.writer.add_scalar('PerCoarseGlobal/Known',   mean_known,   epoch)
        args.writer.add_scalar('PerCoarseGlobal/Unknown', mean_unknown, epoch)
        args.writer.add_scalar('PerCoarseGlobal/All',     mean_all,     epoch)


def test_kmeans_per_coarse_separate(model, loader, epoch,
                                     full_allocation, known_coarses, coarse_of, args):
    """Independent KMeans per coarse with oracle K per coarse."""
    model.eval()
    all_feats, all_labels = [], []
    for images, label, _ in loader:
        with torch.no_grad():
            feats = model(images.cuda(non_blocking=True))
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
# Coarse pseudo-label extraction
# ---------------------------------------------------------------------------

def extract_labeled_protos_coarse(model, train_loader, coarse_of, args):
    """
    Build the pseudo-label hierarchy using coarse labels as the anchor.

    Level 0  (K = 2 × num_coarses): within-coarse 2-split
    Level 1  (K = num_coarses):     ground-truth coarse assignment
    Level 2+ (K = num_coarses/2, …): standard halving via KMeans

    Returns
    -------
    ids            : np.ndarray  — unique sample indices (for _id_to_pos lookup)
    prototype_list : list[Tensor] — cluster centres at each level
    preds_list     : list[Tensor] — per-sample cluster assignments at each level
    metrics        : dict         — all_acc / old_acc / new_acc on unlabeled set
    """
    model.eval()

    all_feats, targets, mask, ids, coarses = (
        [], np.array([]), np.array([]), np.array([]), np.array([])
    )
    metrics = {}

    for images, label, uq_idx, coarse_label, mask_lab_ in tqdm(train_loader):
        images = images[0].cuda()
        label = label.to(device)
        mask_lab_ = mask_lab_.to(device).bool()

        feats = model(images)
        all_feats.append(F.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        ids     = np.append(ids,     uq_idx.cpu().numpy())
        mask    = np.append(mask,    mask_lab_.cpu().bool().numpy())
        coarses = np.append(coarses, coarse_label.cpu().numpy())

    mask    = mask.astype(bool)
    targets = targets.astype(int)
    coarses = coarses.astype(int)
    all_feats = np.concatenate(all_feats)

    # ---- Level 1: GT coarse assignment ----
    unique_coarses  = sorted(np.unique(coarses))
    num_coarses     = len(unique_coarses)
    coarse_to_idx   = {c: i for i, c in enumerate(unique_coarses)}
    coarse_preds    = np.array([coarse_to_idx[c] for c in coarses], dtype=int)
    coarse_centers  = np.stack([all_feats[coarses == c].mean(0) for c in unique_coarses])

    # ---- Level 0: within-coarse 2-split ----
    # Build map: coarse_id → list of remapped fine labels in that coarse
    fine_per_coarse = defaultdict(list)
    for remapped_fine in range(len(coarse_of)):
        fine_per_coarse[int(coarse_of[remapped_fine])].append(remapped_fine)

    level0_centers = []
    for c_idx, c in enumerate(unique_coarses):
        c_feats    = all_feats[coarses == c]
        known_fine = [f for f in fine_per_coarse[c] if f < args.num_labeled_classes]

        if len(known_fine) >= 2:
            # Initialise from labeled sample means of the first two known fine classes.
            labeled_mask = mask  # True = labeled in the merged dataset
            f0_feats = all_feats[(targets == known_fine[0]) & labeled_mask]
            f1_feats = all_feats[(targets == known_fine[1]) & labeled_mask]
            ctr0 = f0_feats.mean(0) if len(f0_feats) > 0 else c_feats.mean(0)
            ctr1 = f1_feats.mean(0) if len(f1_feats) > 0 else c_feats.mean(0)
        else:
            # KMeans(2) within the coarse (handles 1-fine-known and all-unknown cases).
            km2 = KMeans(n_clusters=2, random_state=0, n_init=3).fit(c_feats)
            ctr0 = km2.cluster_centers_[0]
            ctr1 = km2.cluster_centers_[1]

        level0_centers.extend([ctr0, ctr1])

    level0_centers = np.stack(level0_centers).astype(np.float32)  # (2*num_coarses, D)

    # Assign each sample to nearest within-coarse centre
    level0_preds = np.zeros(len(targets), dtype=int)
    for c_idx, c in enumerate(unique_coarses):
        c_mask = (coarses == c)
        cf = all_feats[c_mask]
        d0 = np.linalg.norm(cf - level0_centers[2 * c_idx],     axis=1)
        d1 = np.linalg.norm(cf - level0_centers[2 * c_idx + 1], axis=1)
        level0_preds[c_mask] = np.where(d0 <= d1, 2 * c_idx, 2 * c_idx + 1)

    # ---- Metrics: evaluate level-0 preds on unlabeled set ----
    mask_cls  = np.array([t < args.num_labeled_classes for t in targets])
    u_targets = targets[~mask]
    u_preds   = level0_preds[~mask]
    u_mask_cls = mask_cls[~mask]
    if len(u_preds) > 0:
        results = log_accs_from_preds(
            y_true=u_targets, y_pred=u_preds, mask=u_mask_cls,
            eval_funcs=args.eval_funcs,
            save_name='Level-0 Train ACC Unlabelled', print_output=True,
            save_path=os.path.join(args.log_dir, 'results.jsonl')
        )
        all_acc, old_acc, new_acc = results[args.eval_funcs[0]]
    else:
        all_acc, old_acc, new_acc = float('nan'), float('nan'), float('nan')
    metrics['all_acc'], metrics['old_acc'], metrics['new_acc'] = all_acc, old_acc, new_acc

    # ---- Levels 2+: standard halving above coarse ----
    prototype_list = [
        torch.from_numpy(level0_centers),
        torch.from_numpy(coarse_centers.astype(np.float32)),
    ]
    preds_list = [
        torch.from_numpy(level0_preds),
        torch.from_numpy(coarse_preds),
    ]

    label_proto          = coarse_centers.copy()
    n_level              = num_coarses
    # current_coarse_preds starts as per-sample coarse index (0..num_coarses-1).
    # Each iteration replaces it by mapping through km.labels_ of the CURRENT fit,
    # which has one entry per center at the *previous* level — not per original coarse.
    current_coarse_preds = torch.from_numpy(coarse_preds)
    while n_level > 1:
        n_level = max(n_level // 2, 1)
        km = KMeans(n_clusters=n_level, random_state=0).fit(label_proto)
        label_proto = km.cluster_centers_.astype(np.float32)
        current_coarse_preds = torch.from_numpy(km.labels_)[current_coarse_preds]
        prototype_list.append(torch.from_numpy(label_proto))
        preds_list.append(current_coarse_preds.clone())

    return ids, prototype_list, preds_list, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(projection_head, model, train_loader, test_loader,
          unlabelled_train_loader, merge_train_loader,
          full_allocation, known_coarses, coarse_of, args):

    optimizer = SGD(
        list(projection_head.parameters()) + list(model.parameters()),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )

    sup_con_crit             = SupConLoss()
    strategy                 = args.strategy
    cluster_momentum         = args.cluster_momentum
    unsupervised_smoothing   = args.unsupervised_smoothing
    train_report_interval    = args.train_report_interval
    proto_extract_interval   = args.prototype_extraction_interval
    Distance                 = args.distance

    best_epoch, best_test_acc_lab = 0, 0
    best_stats = []
    Total_loss, Contrastive_loss = [], []
    accuracy_old, accuracy_new, accuracy_all = [], [], []

    for epoch in range(args.epochs):
        loss_record      = AverageMeter()
        loss_cons_record = AverageMeter()

        # ---- Refresh cluster prototypes ----
        with torch.no_grad():
            if epoch % proto_extract_interval == 0:
                uq_index, cluster_protos_list, preds_ind_list, metrics = \
                    extract_labeled_protos_coarse(model, merge_train_loader, coarse_of, args)

                uq_index_int = uq_index.astype(np.int64)
                _id_to_pos   = np.full(int(uq_index_int.max()) + 1, -1, dtype=np.int64)
                _id_to_pos[uq_index_int] = np.arange(len(uq_index_int), dtype=np.int64)

                for i in range(len(preds_ind_list)):
                    preds_ind_list[i]     = preds_ind_list[i].to(device).long()
                    cluster_protos_list[i] = cluster_protos_list[i].to(device)
                    if Distance == 'cosine':
                        cluster_protos_list[i] = (
                            cluster_protos_list[i]
                            / torch.norm(cluster_protos_list[i], dim=1).unsqueeze(1)
                        )

                cluster_distances_list, cluster_radius_list = [], []
                for i in range(len(preds_ind_list)):
                    if Distance == 'euclidean':
                        cd = torch.cdist(cluster_protos_list[i], cluster_protos_list[i])
                    else:
                        cd = torch.matmul(cluster_protos_list[i], cluster_protos_list[i].T)
                    cluster_distances_list.append(cd.clone())
                    cr = (cd + torch.eye(cd.shape[0]).to(device) * cd.max()).min(dim=1)[0] / 2
                    cluster_radius_list.append(cr.clone())

        projection_head.train()
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, coarse_labels, mask_lab = batch
            mask_lab      = mask_lab[:, 0]
            class_labels  = class_labels.to(device)
            coarse_labels = coarse_labels.to(device)
            mask_lab      = mask_lab.to(device).bool()
            images        = torch.cat(images, dim=0).to(device)

            # ---- Forward ----
            features_raw = model(images)                   # [2B, 768]
            features_1, features_2 = features_raw.detach().chunk(2)
            all_features = torch.cat([features_1, features_2], dim=0)
            features     = projection_head(features_raw)   # [2B, mlp_out_dim]
            features     = F.normalize(features, dim=-1)

            # ---- Confusion factor ----
            with torch.no_grad():
                confusion_factor = 0
                if Distance == 'euclidean':
                    pair_dist = torch.cdist(all_features, all_features)
                else:
                    nf = all_features / torch.norm(all_features.unsqueeze(1))
                    pair_dist = torch.matmul(nf, nf.T)

                for i in range(len(preds_ind_list)):
                    cluster_labels = (
                        preds_ind_list[i][_id_to_pos[uq_idxs.cpu().long().numpy()]]
                    ).clone()

                    # Total K at this level derived from the predictions directly
                    k_at_level     = int(preds_ind_list[i].max().item()) + 1
                    cluster_indexer = F.one_hot(cluster_labels.long(), k_at_level).float().T

                    cluster_indexer = torch.cat([cluster_indexer, cluster_indexer], dim=1)
                    n_samples_c     = cluster_indexer.sum(dim=1).unsqueeze(1)
                    n_samples_c[n_samples_c == 0] = 1

                    if Distance == 'euclidean':
                        distance = torch.cdist(all_features, cluster_protos_list[i].float())
                    else:
                        nf = all_features / torch.norm(all_features.unsqueeze(1))
                        distance = torch.matmul(nf, cluster_protos_list[i].float().T)

                    cluster_radius_list[i] = (
                        (cluster_indexer * distance.T).sum(dim=1) / n_samples_c.squeeze()
                        * (1 - cluster_momentum) + cluster_radius_list[i] * cluster_momentum
                    )

                    cluster_labels_2v = torch.cat([cluster_labels, cluster_labels])
                    if Distance == 'euclidean':
                        if strategy == 'zero_one':
                            confusion_factor += (
                                pair_dist > 2 * cluster_radius_list[i][cluster_labels_2v]
                            ).float() / 2 ** i
                        elif strategy == 'pair_dist':
                            confusion_factor += pair_dist / 2 ** i
                        elif strategy == 'pair_cluster':
                            confusion_factor += distance[:, cluster_labels_2v] / 2 ** i
                    else:
                        if strategy == 'zero_one':
                            confusion_factor += (
                                pair_dist < distance[:, cluster_labels_2v] / 2
                            ).float() / 2 ** i
                        elif strategy == 'pair_dist':
                            confusion_factor += -pair_dist / 2 ** i
                        elif strategy == 'pair_cluster':
                            confusion_factor += -distance[:, cluster_labels_2v] / 2 ** i

                confusion_factor = (
                    (confusion_factor - confusion_factor.min())
                    / (confusion_factor.max() - confusion_factor.min() + 1e-7)
                )
                confusion_factor = confusion_factor / confusion_factor.sum(
                    dim=1, keepdim=True
                ).clamp(min=1e-8)

            torch.cuda.empty_cache()

            # ---- Unsupervised contrastive loss (global NTXent with confusion factor) ----
            if args.contrast_unlabel_only:
                f1, f2    = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
                unlab_idx = (~mask_lab).nonzero(as_tuple=True)[0]
                B         = class_labels.size(0)
                unlab_2v  = torch.cat([unlab_idx, B + unlab_idx])
                conf_sliced = confusion_factor[unlab_2v][:, unlab_2v]
            else:
                con_feats   = features
                conf_sliced = confusion_factor

            contrastive_logits, contrastive_labels, similarity = info_nce_logits(
                features=con_feats, confusion_factor=conf_sliced, args=args
            )
            contrastive_loss = LabelSmoothingLoss()(
                contrastive_logits, contrastive_labels, similarity, unsupervised_smoothing
            )

            # ---- Supervised contrastive loss ----
            f1n, f2n = features.chunk(2)
            semisup_con_feats = torch.cat([f1n.unsqueeze(1), f2n.unsqueeze(1)], dim=1)

            # Full features: GT fine SupCon (labeled only)
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats  = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss   = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            dimension = semisup_con_feats.shape[-1]

            # dim/2: GT coarse SupCon (ALL samples)
            sup_con_loss += sup_con_crit(
                semisup_con_feats[:, :, :dimension // 2],
                labels=coarse_labels
            ) / 2

            # dim/4, dim/8, …: pseudo-label hierarchy (i+2 shift because coarse occupies dim/2)
            for i in range(len(preds_ind_list)):
                sup_con_loss += sup_con_crit(
                    semisup_con_feats[:, :, :int(dimension / 2 ** (i + 2))],
                    labels=preds_ind_list[i][_id_to_pos[uq_idxs.cpu().long().numpy()]]
                ) / 2 ** (i + 2)

            # ---- Total loss ----
            loss = (1 - args.sup_con_weight) * contrastive_loss + \
                   args.sup_con_weight * sup_con_loss / 2

            loss_cons_record.update(loss.item(), class_labels.size(0))
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {} Avg Loss: {:.3f} | Contrastive: {:.3f}'.format(
            epoch, loss_record.avg, loss_cons_record.avg))

        Total_loss.append(loss_record.avg)
        Contrastive_loss.append(loss_cons_record.avg)

        with torch.no_grad():
            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_kmeans(
                model, test_loader, epoch=epoch, save_name='Test ACC', args=args,
            )
            if (epoch + 1) % train_report_interval == 0:
                print('Testing on unlabelled examples in the training data...')
                all_acc, old_acc, new_acc = test_kmeans(
                    model, unlabelled_train_loader, epoch=epoch,
                    save_name='Train ACC Unlabelled', args=args, Use_GPU=False
                )
                print('Per-coarse evaluation (global KMeans):')
                test_kmeans_per_coarse(model, unlabelled_train_loader, epoch,
                                       full_allocation, known_coarses, coarse_of, args)
                print('Per-coarse evaluation (separate KMeans per coarse):')
                test_kmeans_per_coarse_separate(model, unlabelled_train_loader, epoch,
                                                full_allocation, known_coarses, coarse_of, args)
            else:
                all_acc, old_acc, new_acc = (
                    metrics['all_acc'], metrics['old_acc'], metrics['new_acc']
                )

            accuracy_old.append(old_acc)
            accuracy_new.append(new_acc)
            accuracy_all.append(all_acc)

        # ---- Logging ----
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        if (epoch + 1) % train_report_interval == 0:
            print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
                all_acc, old_acc, new_acc))
        print('Test Accuracies:  All {:.4f} | Old {:.4f} | New {:.4f}'.format(
            all_acc_test, old_acc_test, new_acc_test))

        exp_lr_scheduler.step()
        torch.save(model.state_dict(), args.model_path)
        print(f'Model saved to {args.model_path}')
        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')

        if old_acc_test > best_test_acc_lab:
            print(f'Best ACC on new Classes on disjoint test set: {new_acc_test:.4f}...')
            best_stats = [all_acc, old_acc, new_acc]
            torch.save(model.state_dict(), args.model_path[:-3] + '_best.pt')
            torch.save(projection_head.state_dict(),
                       args.model_path[:-3] + '_proj_head_best.pt')
            best_test_acc_lab = old_acc_test
            best_epoch = epoch

        print(f'Best Train Epoch: {best_epoch}')

    print('############# Final Reports #############')
    if best_stats:
        print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
            *best_stats))
    print(f'Best Train Epoch: {best_epoch}')

    epochs_arr = np.arange(args.epochs)
    plot_a_loss(epochs_arr, np.array(Total_loss),       'Total')
    plot_a_loss(epochs_arr, np.array(Contrastive_loss), 'Contrastive')
    plot_acc(epochs_arr, accuracy_old, accuracy_new, accuracy_all, 'features')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Coarse-anchor SelEx Contrastive Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--dataset_name',        type=str,      default='aves_small')
    parser.add_argument('--prop_train_labels',   type=float,    default=0.5)
    parser.add_argument('--use_ssb_splits',      type=str2bool, default=True)
    parser.add_argument('--num_novel_classes',   type=int,      default=None)
    parser.add_argument('--hier_split_type',     type=str,      default=None)

    # Coarse labels (required)
    parser.add_argument('--coarse_labels_path',  type=str, required=True,
                        help='JSON mapping fine_class_idx -> coarse_class_idx')

    # Model
    parser.add_argument('--base_model',          type=str,   default='vit_dino')
    parser.add_argument('--model_name',          type=str,   default='vit_dino')
    parser.add_argument('--grad_from_block',     type=int,   default=10)
    parser.add_argument('--warmup_model_dir',    type=str,   default=None)
    parser.add_argument('--use_dino_v1',         type=str2bool, default=False)

    # Training
    parser.add_argument('--batch_size',          type=int,   default=128)
    parser.add_argument('--num_workers',         type=int,   default=4)
    parser.add_argument('--lr',                  type=float, default=0.1)
    parser.add_argument('--momentum',            type=float, default=0.9)
    parser.add_argument('--weight_decay',        type=float, default=1e-4)
    parser.add_argument('--epochs',              type=int,   default=200)
    parser.add_argument('--transform',           type=str,   default='imagenet')
    parser.add_argument('--n_views',             type=int,   default=2)
    parser.add_argument('--seed',                type=int,   default=1)

    # Loss
    parser.add_argument('--temperature',         type=float,    default=1.0)
    parser.add_argument('--sup_con_weight',      type=float,    default=0.35)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--unsupervised_smoothing', type=float, default=1.0)

    # Cluster momentum / confusion factor
    parser.add_argument('--strategy',            type=str,   default='zero_one',
                        choices=['zero_one', 'pair_dist', 'pair_cluster', 'none'])
    parser.add_argument('--cluster_momentum',    type=float, default=1.0)
    parser.add_argument('--distance',            type=str,   default='euclidean',
                        choices=['euclidean', 'cosine'])
    parser.add_argument('--prototype_extraction_interval', type=int, default=1)

    # Evaluation
    parser.add_argument('--eval_funcs',          nargs='+',  default=['v1', 'v2'])
    parser.add_argument('--train_report_interval', type=int, default=200)
    parser.add_argument('--gpu_clustering',      type=str2bool, default=True)
    parser.add_argument('--unbalanced',          type=str2bool, default=False)

    # Misc
    parser.add_argument('--exp_root',            type=str,   default=exp_root)
    parser.add_argument('--exp_id',              type=str,   default=None)
    parser.add_argument('--gpu_id',              type=int,   default=0)
    parser.add_argument('--num_unlabeled_classes', type=int, default=None)

    args = parser.parse_args()

    device = torch.device('cuda:0')
    args   = get_class_splits(args)
    args   = restrict_unlabeled_classes(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.num_labeled_classes   = len(args.train_classes)
    if args.num_unlabeled_classes is None:
        args.num_unlabeled_classes = len(args.unlabeled_classes)

    # ---- Load coarse info ----
    coarse_map = load_coarse_map(args.coarse_labels_path)
    coarse_of  = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)

    k_per_coarse   = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)
    known_coarses  = {coarse_of[i] for i in range(args.num_labeled_classes)}
    known_coarse_k = Counter(coarse_of[i] for i in range(args.num_labeled_classes))
    full_allocation = {int(c): int(k) for c, k in k_per_coarse.items()}
    full_allocation.update(known_coarse_k)
    total_k_unknown = sum(k_per_coarse.values())

    init_experiment(args, runner_name=['contrastive_coarse', args.dataset_name],
                    exp_id=args.exp_id)
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    print(f'Labeled classes: {args.num_labeled_classes} | '
          f'Unlabeled classes: {args.num_unlabeled_classes}')
    print(f'K: labeled={args.num_labeled_classes} | unknown={total_k_unknown} | '
          f'total={args.num_labeled_classes + total_k_unknown}')
    print(f'Known coarses: {len(known_coarses)} | Unknown coarses: {len(k_per_coarse)}')
    print(f'Unique coarses: {len(np.unique(coarse_of))}')

    # ---- Base model ----
    if args.base_model == 'vit_dino':
        args.interpolation  = 3
        args.crop_pct       = 0.875
        args.image_size     = 224
        args.feat_dim       = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim    = 65536

        if args.use_dino_v1:
            print('Loading DINOv1 vitb16 backbone')
            model = torch.hub.load(
                'facebookresearch/dino:main', 'dino_vitb16', pretrained=False
            )
            torch.cuda.empty_cache()
            state_dict = torch.load(dino_v1_vitb16_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            print('Loading DINOv2 vitb14-reg4 backbone')
            model      = vits.__dict__['vit_base']()
            state_dict = torch.load(dino_pretrain_path2, map_location='cpu')
            model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(
                torch.load(args.warmup_model_dir + 'model_best.pt', map_location='cpu'),
                strict=False
            )
        model.to(device)

        for m in model.parameters():
            m.requires_grad = False
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    else:
        raise NotImplementedError(f'Unknown base_model: {args.base_model}')

    # ---- Transforms ----
    train_transform, test_transform = get_transform(
        args.transform, image_size=args.image_size, args=args
    )
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=args.n_views
    )

    # ---- Datasets ----
    _, test_dataset, unlabelled_train_examples_test, raw_datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )

    # Both train_dataset and merge_train_dataset use CoarseMergedDataset for 5-tuple output
    train_dataset = CoarseMergedDataset(
        labelled_dataset=raw_datasets['train_labelled'],
        unlabelled_dataset=raw_datasets['train_unlabelled'],
        coarse_of=coarse_of,
    )

    # ---- Sampler ----
    label_len      = len(raw_datasets['train_labelled'])
    unlabelled_len = len(raw_datasets['train_unlabelled'])
    sample_weights = [
        1 if i < label_len else label_len / (unlabelled_len + label_len)
        for i in range(len(train_dataset))
    ]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset)
    )

    # ---- Dataloaders ----
    train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False,
        sampler=sampler, drop_last=True,
    )
    merge_train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False
    )
    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False
    )
    test_loader_labelled = DataLoader(
        test_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False
    )

    # ---- Projection head ----
    projection_head = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    )
    if args.warmup_model_dir is not None:
        print(f'Loading projection head weights from {args.warmup_model_dir}')
        projection_head.load_state_dict(
            torch.load(args.warmup_model_dir + 'model_proj_head_best.pt', map_location='cpu'),
            strict=False
        )
    projection_head.to(device)

    # ---- Train ----
    if not os.path.exists('Plots'):
        os.mkdir('Plots')

    train(projection_head, model,
          train_loader, test_loader_labelled,
          test_loader_unlabelled, merge_train_loader,
          full_allocation, known_coarses, coarse_of, args)

    torch.cuda.empty_cache()
    print(args.model_path)
