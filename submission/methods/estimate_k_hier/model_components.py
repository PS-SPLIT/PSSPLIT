"""
model_components.py — Self-contained Set Transformer model classes for K estimation.

Adapted from train_transformer_approach_direct_k_ensemble_4.py.
No project-specific imports. Usable standalone with standard PyTorch + sklearn + numpy.

Key components:
  SetTransformerForKAndSupCon  — the main model
  PerSplitSequenceDataset      — dataset where each item is one coarse group
  SupConLoss                   — supervised contrastive loss
  EMA                          — exponential moving average for model weights
  pad_collate                  — collate function for variable-length sequences
  predict_ensemble_and_cluster — ensemble inference returning K and token embeddings
  train_one_model              — train a single model on train/val sequences
"""

import copy
import os
import json
import time
import warnings as _warnings

import numpy as np
import math

# pyclustering compat shim
if not hasattr(np, "warnings"):
    np.warnings = _warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, device="cuda"):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions required')
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
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        den = mask.sum(1).clamp_min(1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / den

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def apply_token_dropout(tokens, labels, key_padding_mask, p):
    """Drop random non-padded tokens with probability p (ensure ≥1 token survives)."""
    if p <= 0:
        return tokens, labels, key_padding_mask
    B, N, D = tokens.shape
    kpm = key_padding_mask.clone()
    lab = labels.clone()
    keep_mask = ~kpm
    drop = torch.zeros_like(kpm, dtype=torch.bool)
    drop[keep_mask] = torch.rand_like(tokens[..., 0], dtype=tokens.dtype).lt(p)[keep_mask]
    all_dropped = ((~kpm & drop).sum(dim=1) == (~kpm).sum(dim=1))
    if all_dropped.any():
        for b in torch.where(all_dropped)[0].tolist():
            real_idxs = torch.where(~kpm[b])[0]
            j = real_idxs[torch.randint(len(real_idxs), (1,))].item()
            drop[b, j] = False
    kpm = kpm | drop
    lab[drop] = -1
    tokens = tokens.masked_fill(drop.unsqueeze(-1), 0.0)
    return tokens, lab, kpm


def apply_same_class_mixup(tokens, labels, key_padding_mask, alpha, prob, rng=None):
    """Same-batch, same-sequence, same-label mixup (labels unchanged)."""
    if alpha <= 0 or prob <= 0:
        return tokens
    B, N, D = tokens.shape
    out = tokens.clone()
    device = tokens.device
    if rng is None:
        rng = torch.Generator(device=device)
    for b in range(B):
        real = ~key_padding_mask[b] & (labels[b] != -1)
        if real.sum() < 2:
            continue
        idx = torch.where(real)[0]
        yb = labels[b, idx]
        classes = torch.unique(yb)
        for c in classes.tolist():
            cls_mask = (yb == c)
            cls_idx = idx[cls_mask]
            if cls_idx.numel() < 2:
                continue
            m = torch.rand(cls_idx.numel(), device=device) < prob
            if not m.any():
                continue
            perm = torch.randperm(cls_idx.numel(), generator=rng, device=device)
            partners = cls_idx[perm]
            lam = torch.distributions.Beta(alpha, alpha).sample((cls_idx.numel(),)).to(device)
            lam = lam.clamp(1e-6, 1.0 - 1e-6)
            src = cls_idx[m]
            prt = partners[m]
            lam_m = lam[m].unsqueeze(-1)
            out[b, src] = lam_m * tokens[b, src] + (1.0 - lam_m) * tokens[b, prt]
    return out


# ---------------------------------------------------------------------------
# SupCon loss across a batch (same dataset + same label = positive)
# ---------------------------------------------------------------------------

def _supcon_loss_across_batch(token_embs, labels, device, sup):
    B, N, C = token_embs.shape
    nonpad = (labels != -1)
    if not nonpad.any():
        return torch.zeros((), device=device)
    z = F.normalize(token_embs[nonpad], dim=-1)
    y = labels[nonpad]
    groups = torch.arange(B, device=device).unsqueeze(1).expand(B, N)[nonpad]
    with torch.no_grad():
        same_group = groups.unsqueeze(1).eq(groups.unsqueeze(0))
        same_label = y.unsqueeze(1).eq(y.unsqueeze(0))
        mask = (same_group & same_label).float().to(device)
    return sup(z.unsqueeze(1), labels=None, mask=mask, device=device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=1024, dropout=0.2, activation='relu', batch_first=True):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask, mask=src_mask)


class SetTransformerForKAndSupCon(nn.Module):
    """
    Set Transformer that jointly:
    - predicts K (number of fine classes in a coarse group) via a [CLS] token
    - shapes token embeddings via SupCon (same fine class = positive pair)

    Forward returns: (k_logits [B, Kmax], tok_b [B, N, bottleneck_dim], cls_b [B, bottleneck_dim])
    """
    def __init__(self, input_dim, Kmax, d_model=256, nhead=8, num_layers=4,
                 ff_dim=1024, dropout=0.2, bottleneck_dim=256,
                 proj_hidden=None, token_norm=True, cls_norm=True):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.enc = TransformerEncoderModel(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            dim_feedforward=ff_dim, dropout=dropout, activation='relu', batch_first=True
        )

        hidden = proj_hidden or max(bottleneck_dim * 2, d_model)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model) if cls_norm else nn.Identity(),
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden, bottleneck_dim)
        )
        self.tok_head = nn.Sequential(
            nn.LayerNorm(d_model) if token_norm else nn.Identity(),
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden, bottleneck_dim)
        )
        self.k_head = nn.Linear(bottleneck_dim, Kmax)

    def forward(self, tokens, key_padding_mask=None):
        B, N, _ = tokens.shape
        x = self.proj(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if key_padding_mask is not None:
            pad = torch.zeros((B, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            kpm = torch.cat([pad, key_padding_mask], dim=1)
        else:
            kpm = None
        z = self.enc(x, src_key_padding_mask=kpm)
        cls_emb = z[:, 0, :]
        tok_emb = z[:, 1:, :]
        cls_b = self.cls_head(cls_emb)
        tok_b = self.tok_head(tok_emb)
        k_logits = self.k_head(cls_b)
        return k_logits, tok_b, cls_b


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerSplitSequenceDataset(Dataset):
    """
    Each item is one coarse group (a "split"). Returns a dict with:
      tokens [N, D], labels [N], obs_ids [N], k_true (scalar), split_id (int)

    seqs: list of dicts with keys:
      "Xp" [N, D] float32, "y" [N] int64, "k_true" int, "split_id" int, "obs_ids" [N] int64

    subsample_range: (lo, hi) fraction of tokens to use each epoch. None = all tokens.
    """
    def __init__(self, seqs, max_tokens_per_split=None, seed=0,
                 subsample_range=(0.1, 1.0)):
        self.seqs = seqs
        self.max_tokens = max_tokens_per_split
        self.rng = np.random.default_rng(seed)
        self.subsample_range = subsample_range

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        X = s["Xp"].copy(); y = s["y"].copy(); obs = s["obs_ids"].copy()
        k_true = s["k_true"]; split_id = s["split_id"]

        if self.subsample_range is not None:
            lo, hi = self.subsample_range
            frac = float(self.rng.uniform(lo, hi))
            n_take = max(1, int(round(frac * X.shape[0])))
            take_idx = self.rng.choice(X.shape[0], size=n_take, replace=False)
            X, y, obs = X[take_idx], y[take_idx], obs[take_idx]

        if self.max_tokens is not None and X.shape[0] > self.max_tokens:
            idxs = self.rng.choice(X.shape[0], size=self.max_tokens, replace=False)
            X, y, obs = X[idxs], y[idxs], obs[idxs]

        perm = self.rng.permutation(len(X))
        X, y, obs = X[perm], y[perm], obs[perm]

        return {
            "tokens": torch.from_numpy(X).float(),
            "labels": torch.from_numpy(y).long(),
            "obs_ids": torch.from_numpy(obs).long(),
            "k_true": torch.tensor(k_true, dtype=torch.long),
            "split_id": split_id
        }


def pad_collate(batch):
    """Collate variable-length sequences into padded tensors."""
    B = len(batch)
    Din = batch[0]["tokens"].shape[1]
    Ns = [b["tokens"].shape[0] for b in batch]
    Nmax = max(Ns)

    tokens = torch.zeros(B, Nmax, Din, dtype=batch[0]["tokens"].dtype)
    labels = torch.full((B, Nmax), -1, dtype=torch.long)
    obs_ids = torch.full((B, Nmax), -1, dtype=torch.long)
    kpm = torch.ones(B, Nmax, dtype=torch.bool)  # True = padded
    k_trues = torch.zeros(B, dtype=torch.long)
    sids = []

    for i, b in enumerate(batch):
        n = b["tokens"].shape[0]
        tokens[i, :n] = b["tokens"]
        labels[i, :n] = b["labels"]
        obs_ids[i, :n] = b["obs_ids"]
        kpm[i, :n] = False
        k_trues[i] = b["k_true"]
        sids.append(b["split_id"])

    return {"tokens": tokens, "labels": labels, "obs_ids": obs_ids,
            "key_padding_mask": kpm, "k_true": k_trues, "split_ids": sids}


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def _epoch(model, ema, loader, optimizer, device, train,
           supcon_lambda=0.3, log_every=50, jitter=0.02,
           token_dropout_p=0.05, mixup_alpha=0.2, mixup_p=0.5):
    model.train() if train else model.eval()
    agg = {"loss": 0.0, "ceK": 0.0, "sup": 0.0, "accK": 0.0}
    steps = 0

    ceK = nn.CrossEntropyLoss()
    sup = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)

    for step, batch in enumerate(loader):
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        k_true = batch["k_true"].to(device)
        kpm = batch["key_padding_mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if jitter > 0:
                std = tokens.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
                tokens = tokens + torch.randn_like(tokens) * (std * jitter)
            tokens, labels, kpm = apply_token_dropout(tokens, labels, kpm, token_dropout_p)
            tokens = apply_same_class_mixup(tokens, labels, kpm, alpha=mixup_alpha, prob=mixup_p)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            k_logits, token_embs, _ = model(tokens, key_padding_mask=kpm)
            if train:
                loss_ce = ceK(k_logits, k_true - 1)
                loss_sup = _supcon_loss_across_batch(token_embs, labels, device, sup) * supcon_lambda
                loss = loss_ce + loss_sup
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update(model)

        with torch.no_grad():
            predK = (k_logits.argmax(1) + 1)
            accK = (predK == k_true).float().mean().item()

        if train:
            agg["loss"] += float(loss.item())
            agg["ceK"] += float(loss_ce.item())
            agg["sup"] += float(loss_sup.item())
        agg["accK"] += accK
        steps += 1

        if (step + 1) % log_every == 0:
            print(f"  {'train' if train else 'val'} step {step+1}: "
                  f"loss {agg['loss']/steps:.4f} | ceK {agg['ceK']/steps:.4f} | "
                  f"sup {agg['sup']/steps:.4f} | accK {agg['accK']/steps:.3f}")

    for k in agg:
        agg[k] /= max(steps, 1)
    return agg


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def train_one_model(model_id, input_dim, Kmax_train, train_seqs, val_seqs,
                    ckpt_dir, epochs, batch_size, max_tokens_per_split,
                    lr, momentum, weight_decay, supcon_lambda,
                    jitter_std, token_dropout_p, mixup_alpha, mixup_p,
                    d_model, nhead, num_layers, ff_dim, dropout, bottleneck_dim,
                    device, seed=1000):
    """Train a single SetTransformerForKAndSupCon model. Returns path to model dir."""
    mdir = os.path.join(ckpt_dir, f"model_{model_id:02d}")
    os.makedirs(mdir, exist_ok=True)

    # If epochs=0, skip training entirely. Requires best_ema.pt to already exist.
    if epochs <= 0:
        ema_path = os.path.join(mdir, "best_ema.pt")
        if os.path.exists(ema_path):
            print(f"[M{model_id:02d}] epochs=0 and best_ema.pt found — skipping training.")
            return mdir
        raise ValueError(
            f"epochs=0 but no best_ema.pt found in {mdir}. "
            "Provide a valid --ckpt_dir with pre-trained checkpoints."
        )
    mlog = os.path.join(mdir, "log.jsonl")

    base = int(seed) + 1000 * int(model_id)
    np.random.seed(base)
    torch.manual_seed(base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base)

    train_ds = PerSplitSequenceDataset(train_seqs, max_tokens_per_split=max_tokens_per_split,
                                        seed=base, subsample_range=(0.1, 1.0))
    val_ds = PerSplitSequenceDataset(val_seqs, max_tokens_per_split=None,
                                      seed=base + 10_000, subsample_range=None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=pad_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=pad_collate, pin_memory=True)

    model = SetTransformerForKAndSupCon(
        input_dim=input_dim, Kmax=Kmax_train,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        ff_dim=ff_dim, dropout=dropout, bottleneck_dim=bottleneck_dim
    ).to(device)
    ema = EMA(model, decay=0.999)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                             weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 1e-3)

    best_val_accK_ema = -1.0

    log_f = open(mlog, "a", encoding="utf-8")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = _epoch(model, ema, train_loader, optim, device, train=True,
                    supcon_lambda=supcon_lambda, log_every=max(10, len(train_loader)),
                    jitter=jitter_std, token_dropout_p=token_dropout_p,
                    mixup_alpha=mixup_alpha, mixup_p=mixup_p)

        ema_model = copy.deepcopy(model).to(device)
        ema.copy_to(ema_model)
        va_ema = _epoch(ema_model, ema, val_loader, optim, device, train=False,
                        supcon_lambda=supcon_lambda, log_every=max(10, len(val_loader)),
                        jitter=0.0, token_dropout_p=0.0, mixup_alpha=0.0, mixup_p=0.0)
        scheduler.step()

        rec = {"model_id": model_id, "epoch": epoch, "sec": round(time.time() - t0, 2),
               "train": tr, "val_ema": va_ema}
        print(f"[M{model_id:02d}] Ep {epoch:03d}/{epochs} | tr accK={tr['accK']:.3f} | "
              f"val_ema accK={va_ema['accK']:.3f} | {rec['sec']}s")
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        if va_ema["accK"] >= best_val_accK_ema:
            best_val_accK_ema = va_ema["accK"]
            torch.save({"model": ema_model.state_dict(), "epoch": epoch,
                        "val_stats_ema": va_ema}, os.path.join(mdir, "best_ema.pt"))
            print(f"[M{model_id:02d}][best_ema] ep {epoch} | val_ema accK={best_val_accK_ema:.3f}")

    torch.save({"model": model.state_dict(), "epoch": epochs}, os.path.join(mdir, "final.pt"))
    log_f.close()
    return mdir


# ---------------------------------------------------------------------------
# Load ensemble
# ---------------------------------------------------------------------------

def load_ensemble(model_dirs, which, input_dim, Kmax_train,
                  d_model, nhead, num_layers, ff_dim, dropout, bottleneck_dim, device):
    """Load an ensemble of models. which ∈ {"best_ema", "final"}."""
    models = []
    for mdir in model_dirs:
        ckpt_path = os.path.join(mdir, f"{which}.pt")
        if not os.path.exists(ckpt_path):
            alt = os.path.join(mdir, "final.pt")
            ckpt_path = alt if os.path.exists(alt) else None
        if ckpt_path is None:
            continue
        model = SetTransformerForKAndSupCon(
            input_dim=input_dim, Kmax=Kmax_train,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout, bottleneck_dim=bottleneck_dim
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_ensemble(models, seqs, device, batch_size=8, max_tokens_per_split=None, seed=0):
    """
    Run ensemble forward pass on a list of coarse-group sequences.
    Returns list of dicts with:
      split_id, k_hat (int), k_true (int), p_split (float),
      token_embs (np.float32 [N, bottleneck_dim]), obs_ids (np.int64 [N]),
      y_true_tokens (np.int64 [N])
    """
    ds = PerSplitSequenceDataset(seqs, max_tokens_per_split=max_tokens_per_split,
                                  seed=seed, subsample_range=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=pad_collate)

    out = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].cpu().numpy()
        obs_ids = batch["obs_ids"].cpu().numpy()
        kpm = batch["key_padding_mask"].to(device)
        k_true = batch["k_true"].cpu().numpy()
        split_ids = batch["split_ids"]

        sum_logits = None
        sum_tok = None
        for m in models:
            k_logits, tok_embs, _ = m(tokens, key_padding_mask=kpm)
            if sum_logits is None:
                sum_logits = k_logits.detach().clone()
                sum_tok = tok_embs.detach().clone()
            else:
                sum_logits += k_logits
                sum_tok += tok_embs

        avg_logits = sum_logits / max(1, len(models))
        avg_tok = sum_tok / max(1, len(models))
        avg_tok = F.normalize(avg_tok, dim=-1)

        probs = torch.softmax(avg_logits, dim=1)
        p_k1 = probs[:, 0]
        p_split = (1.0 - p_k1).cpu().numpy()
        khat = (avg_logits.argmax(1) + 1).cpu().numpy()

        B, N, C = avg_tok.shape
        avg_tok = avg_tok.cpu().numpy()

        for i in range(B):
            sid = split_ids[i]
            true_k = int(k_true[i])
            real = (labels[i] != -1)
            Zi = avg_tok[i, real, :]
            yi = labels[i, real]
            oi = obs_ids[i, real]
            Kh = int(khat[i])

            out.append({
                "split_id": sid,
                "k_true": true_k,
                "k_hat": Kh,
                "p_split": float(p_split[i]),
                "token_embs": Zi,       # [n_tokens, bottleneck_dim]
                "obs_ids": oi,
                "y_true_tokens": yi
            })
    return out


def predict_ensemble_per_model(models, seqs, device, batch_size=8,
                                max_tokens_per_split=None, seed=0):
    """
    Like predict_ensemble but returns per-model token embeddings separately.
    K prediction still uses averaged logits across all models.

    Each result dict contains:
      split_id, k_hat, k_true, p_split, obs_ids,
      per_model_embs: list of [N, bottleneck_dim] float32 arrays (one per model,
                       each L2-normalized independently)
    """
    ds = PerSplitSequenceDataset(seqs, max_tokens_per_split=max_tokens_per_split,
                                  seed=seed, subsample_range=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=pad_collate)

    out = []
    for batch in loader:
        tokens   = batch["tokens"].to(device)
        labels   = batch["labels"].cpu().numpy()
        obs_ids  = batch["obs_ids"].cpu().numpy()
        kpm      = batch["key_padding_mask"].to(device)
        k_true   = batch["k_true"].cpu().numpy()
        split_ids = batch["split_ids"]
        B = tokens.shape[0]

        sum_logits = None
        per_model_tok = []        # list of [B, N, C] per model (cpu numpy)

        for m in models:
            with torch.no_grad():
                k_logits, tok_embs, _ = m(tokens, key_padding_mask=kpm)
            tok_norm = F.normalize(tok_embs, dim=-1).cpu().numpy()
            per_model_tok.append(tok_norm)
            if sum_logits is None:
                sum_logits = k_logits.detach().clone()
            else:
                sum_logits += k_logits

        avg_logits = sum_logits / max(1, len(models))
        p_split = (1.0 - torch.softmax(avg_logits, dim=1)[:, 0]).cpu().numpy()
        khat = (avg_logits.argmax(1) + 1).cpu().numpy()

        for i in range(B):
            real = (labels[i] != -1)
            oi = obs_ids[i, real]
            out.append({
                "split_id": split_ids[i],
                "k_true":   int(k_true[i]),
                "k_hat":    int(khat[i]),
                "p_split":  float(p_split[i]),
                "obs_ids":  oi,
                # One [N, C] L2-normed array per model
                "per_model_embs": [tok[i, real, :] for tok in per_model_tok],
            })

    return out


def get_val_accK_per_model(model_dirs):
    """
    Load the best-EMA validation K-accuracy for each model from its checkpoint.
    Returns a list of floats (same order as model_dirs). Missing checkpoints → 0.0.
    """
    vals = []
    for mdir in model_dirs:
        ck_path = os.path.join(mdir, "best_ema.pt")
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location="cpu")
            vals.append(float(ck["val_stats_ema"]["accK"]))
        else:
            vals.append(0.0)
    return vals
