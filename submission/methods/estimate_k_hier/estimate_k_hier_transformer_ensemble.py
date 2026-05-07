"""
estimate_k_hier_transformer_ensemble.py — Set Transformer K estimation with
comprehensive ensemble embedding strategies for clustering evaluation.

This is an extended copy of estimate_k_hier_transformer.py.  K prediction is
identical (averaged logits across all models).  Clustering evaluation is run
with five strategies for the transformer embeddings:

  avg        — arithmetic mean of all models' embeddings, re-normalised (original)
  best       — embeddings from the single model with highest val K-accuracy
  concat     — all models' embeddings concatenated then re-normalised
  dist_avg   — averaged pairwise cosine-distance matrix; clustered with
                agglomerative (average linkage) and spectral methods
  vote       — KMeans per model, Hungarian-aligned to a reference, majority vote

All strategies share the same K estimate (averaged logits).
Results for every strategy are saved in a single *_metrics.json.

Run standalone (epochs=0 to load existing checkpoints):
    python -m methods.estimate_k_hier.estimate_k_hier_transformer_ensemble \\
        --dataset_name aves_small \\
        --coarse_labels_path data/coarse_labels/aves_small.json \\
        --hier_split_type 5050 --prop_train_labels 1.0 \\
        --n_models 5 --epochs 0 \\
        --ckpt_dir osr_novel_categories/k_estimation_results/aves_small_dinov2/transformer_k_transformer_ckpts \\
        --k_results_path osr_novel_categories/k_estimation_results/aves_small_dinov2/transformer_ensemble_k.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from config import feature_extract_dir
from data.data_utils import build_coarse_of, load_coarse_map, oracle_k_per_coarse
from data.get_datasets import get_class_splits, get_datasets, restrict_unlabeled_classes
from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from methods.estimate_k_hier.estimate_k_hier import load_all_features
from methods.estimate_k_hier.model_components import (
    SetTransformerForKAndSupCon,
    get_val_accK_per_model,
    load_ensemble,
    predict_ensemble,
    predict_ensemble_per_model,
    train_one_model,
)
from project_utils.cluster_utils import cluster_acc
from project_utils.general_utils import str2bool

# Reuse helpers from the original transformer module
from methods.estimate_k_hier.estimate_k_hier_transformer import (
    build_coarse_sequences,
    build_test_coarse_sequences,
    load_test_features,
    reassemble_features,
)


# ---------------------------------------------------------------------------
# Feature-array builders for the three feature-based strategies
# ---------------------------------------------------------------------------

def _build_strategy_feats(per_model_results, N, bottleneck_dim, best_idx):
    """
    Build flat [N, D] arrays for the avg / best / concat strategies.
    All returned arrays are L2-normalised float32.
    Unknown-coarse positions are filled; known-coarse positions stay zero
    (eval_per_coarse only iterates coarses present in k_per_coarse).
    """
    n_models = len(per_model_results[0]["per_model_embs"])
    avg_f    = np.zeros((N, bottleneck_dim),          dtype=np.float32)
    best_f   = np.zeros((N, bottleneck_dim),          dtype=np.float32)
    concat_f = np.zeros((N, n_models * bottleneck_dim), dtype=np.float32)

    for r in per_model_results:
        oids = r["obs_ids"]
        embs = r["per_model_embs"]   # list of [n, D] L2-normed arrays

        # avg: mean then re-normalise
        avg_emb = np.mean(np.stack(embs, axis=0), axis=0)  # [n, D]
        norms = np.linalg.norm(avg_emb, axis=1, keepdims=True).clip(1e-8)
        avg_f[oids] = (avg_emb / norms).astype(np.float32)

        # best: already L2-normed from predict_ensemble_per_model
        best_f[oids] = embs[best_idx].astype(np.float32)

        # concat: stack and re-normalise
        cat_emb = np.concatenate(embs, axis=1)             # [n, n_models*D]
        norms_c = np.linalg.norm(cat_emb, axis=1, keepdims=True).clip(1e-8)
        concat_f[oids] = (cat_emb / norms_c).astype(np.float32)

    return avg_f, best_f, concat_f


# ---------------------------------------------------------------------------
# Distance-matrix averaging strategy
# ---------------------------------------------------------------------------

def _eval_dist_avg(per_model_results, all_labels, k_per_coarse, oracle_k):
    """
    For each unknown coarse:
      1. Compute pairwise cosine-distance matrix for each model's embeddings.
      2. Average the distance matrices across models.
      3. Cluster with agglomerative (average linkage) and spectral methods.

    Returns {int(coarse_id): {dist_avg_agglo_acc, dist_avg_agglo_ami,
                               dist_avg_spectral_acc, dist_avg_spectral_ami}}
    """
    results = {}
    for r in per_model_results:
        sid = str(r["split_id"])
        if sid not in k_per_coarse:
            continue

        c     = int(sid)
        oids  = r["obs_ids"]
        embs  = r["per_model_embs"]  # list of [n, D] L2-normed
        n     = len(oids)
        k_hat = max(1, int(k_per_coarse[sid]))
        k_true = oracle_k.get(c, len(np.unique(all_labels[oids])))
        entry = {"k_true": k_true, "k_hat": k_hat}

        if n < 2 or k_hat < 2:
            for key in ("dist_avg_agglo_acc", "dist_avg_agglo_ami",
                        "dist_avg_spectral_acc", "dist_avg_spectral_ami"):
                entry[key] = float("nan")
            results[c] = entry
            continue

        # Average cosine distance matrices  (cos_dist = 1 - cos_sim; embs are L2-normed)
        dist_mat = np.zeros((n, n), dtype=np.float64)
        for emb in embs:
            sim = emb.astype(np.float64) @ emb.astype(np.float64).T
            dist_mat += 1.0 - np.clip(sim, -1.0, 1.0)
        dist_mat /= max(1, len(embs))
        np.fill_diagonal(dist_mat, 0.0)
        dist_mat = np.clip(dist_mat, 0.0, None)

        k_eff = min(k_hat, n)

        # Agglomerative — average linkage on precomputed distances
        try:
            preds = AgglomerativeClustering(
                n_clusters=k_eff, metric="precomputed", linkage="average"
            ).fit_predict(dist_mat)
            entry["dist_avg_agglo_acc"] = float(cluster_acc(all_labels[oids].astype(int), preds.astype(int)))
            entry["dist_avg_agglo_ami"] = float(adjusted_mutual_info_score(all_labels[oids], preds))
        except Exception as exc:
            print(f"  WARNING dist_avg agglo coarse {sid}: {exc}")
            entry["dist_avg_agglo_acc"] = float("nan")
            entry["dist_avg_agglo_ami"] = float("nan")

        # Spectral — convert distance → affinity then cluster
        try:
            max_d = dist_mat.max()
            affinity = np.clip(1.0 - dist_mat / max(max_d, 1e-8), 0.0, 1.0)
            np.fill_diagonal(affinity, 1.0)
            preds = SpectralClustering(
                n_clusters=k_eff, affinity="precomputed",
                assign_labels="cluster_qr", random_state=0
            ).fit_predict(affinity)
            entry["dist_avg_spectral_acc"] = float(cluster_acc(all_labels[oids].astype(int), preds.astype(int)))
            entry["dist_avg_spectral_ami"] = float(adjusted_mutual_info_score(all_labels[oids], preds))
        except Exception as exc:
            print(f"  WARNING dist_avg spectral coarse {sid}: {exc}")
            entry["dist_avg_spectral_acc"] = float("nan")
            entry["dist_avg_spectral_ami"] = float("nan")

        results[c] = entry

    return results


# ---------------------------------------------------------------------------
# Hungarian-aligned majority-vote strategy
# ---------------------------------------------------------------------------

def _eval_hungarian_vote(per_model_results, all_labels, k_per_coarse, oracle_k):
    """
    For each unknown coarse:
      1. KMeans-cluster each model's embeddings independently at k_hat.
      2. Hungarian-match every model's assignment to a reference (first valid model).
      3. Majority-vote across aligned assignments for each sample.

    Returns {int(coarse_id): {vote_acc, vote_ami}}
    """
    results = {}
    for r in per_model_results:
        sid = str(r["split_id"])
        if sid not in k_per_coarse:
            continue

        c     = int(sid)
        oids  = r["obs_ids"]
        embs  = r["per_model_embs"]
        n     = len(oids)
        k_hat = max(1, int(k_per_coarse[sid]))
        k_true = oracle_k.get(c, len(np.unique(all_labels[oids])))
        entry = {"k_true": k_true, "k_hat": k_hat}

        if n < 2 or k_hat < 2:
            entry["vote_acc"] = float("nan")
            entry["vote_ami"] = float("nan")
            results[c] = entry
            continue

        k_eff = min(k_hat, n)

        # Per-model KMeans
        preds_list = []
        for emb in embs:
            try:
                preds_list.append(
                    KMeans(n_clusters=k_eff, n_init=10, random_state=0)
                    .fit_predict(emb.astype(np.float64)).astype(np.int64)
                )
            except Exception:
                preds_list.append(None)

        valid = [p for p in preds_list if p is not None]
        if not valid:
            entry["vote_acc"] = float("nan")
            entry["vote_ami"] = float("nan")
            results[c] = entry
            continue

        # Hungarian-align every model to the reference (first valid)
        ref = valid[0]
        aligned = [ref]
        for preds in valid[1:]:
            cost = np.zeros((k_eff, k_eff), dtype=np.int64)
            for t, p in zip(ref, preds):
                cost[t, p] += 1
            row, col = linear_sum_assignment(-cost)
            remap = {int(old): int(new) for old, new in zip(col, row)}
            aligned.append(np.array([remap.get(int(p), int(p)) for p in preds], dtype=np.int64))

        # Majority vote
        votes = np.stack(aligned, axis=0)   # [n_valid, n_samples]
        final = np.array([
            np.bincount(votes[:, i].astype(np.int64), minlength=k_eff).argmax()
            for i in range(n)
        ], dtype=np.int64)

        try:
            entry["vote_acc"] = float(cluster_acc(all_labels[oids].astype(int), final.astype(int)))
            entry["vote_ami"] = float(adjusted_mutual_info_score(all_labels[oids], final))
        except Exception as exc:
            print(f"  WARNING vote coarse {sid}: {exc}")
            entry["vote_acc"] = float("nan")
            entry["vote_ami"] = float("nan")

        results[c] = entry

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _nanmean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def _aggregate_ensemble(per_coarse, cluster_methods):
    """
    Aggregate per-coarse metrics over 'all' coarses and 'multi_class' (k_true > 1).
    Handles all strategy keys including dist_avg and vote.
    """
    all_entries   = list(per_coarse.values())
    multi_entries = [e for e in all_entries if e.get("k_true", 1) > 1]

    # Collect all metric key names (excluding bookkeeping keys)
    skip = {"k_true", "k_hat", "k_score"}
    sample_entry = all_entries[0] if all_entries else {}
    metric_keys = [k for k in sample_entry if k not in skip]

    agg = {}
    for split_name, entries in [("all", all_entries), ("multi_class", multi_entries)]:
        d = {"n_coarses": len(entries)}
        for key in metric_keys:
            d[key] = _nanmean([e.get(key, float("nan")) for e in entries])
        # k_accuracy from k_score
        if "k_score" in sample_entry:
            d["k_accuracy"] = _nanmean([e.get("k_score", float("nan")) for e in entries])
        agg[split_name] = d

    return agg


def _print_ensemble_summary(agg, cluster_methods):
    """Print a tidy per-strategy summary table."""
    strategies = [
        ("backbone",  [f"{m}_acc" for m in cluster_methods], [f"{m}_ami" for m in cluster_methods]),
        ("avg",       [f"avg_{m}_acc" for m in cluster_methods], [f"avg_{m}_ami" for m in cluster_methods]),
        ("best",      [f"best_{m}_acc" for m in cluster_methods], [f"best_{m}_ami" for m in cluster_methods]),
        ("concat",    [f"concat_{m}_acc" for m in cluster_methods], [f"concat_{m}_ami" for m in cluster_methods]),
        ("dist_avg",  ["dist_avg_agglo_acc", "dist_avg_spectral_acc"],
                      ["dist_avg_agglo_ami", "dist_avg_spectral_ami"]),
        ("vote",      ["vote_acc"], ["vote_ami"]),
    ]
    for split_name in ("all", "multi_class"):
        d = agg.get(split_name, {})
        n = d.get("n_coarses", 0)
        kacc = d.get("k_accuracy", float("nan"))
        print(f"\n  [{split_name} | n={n}]  k_acc={kacc:.3f}")
        for strat_name, acc_keys, ami_keys in strategies:
            for ak, amk in zip(acc_keys, ami_keys):
                a = d.get(ak, float("nan"))
                am = d.get(amk, float("nan"))
                label = ak.replace("_acc", "")
                na = "nan" if np.isnan(a) else f"{a:.3f}"
                nam = "nan" if np.isnan(am) else f"{am:.3f}"
                print(f"    {label:<35s}: acc={na}  ami={nam}")


# ---------------------------------------------------------------------------
# Main clustering eval (all strategies)
# ---------------------------------------------------------------------------

def _run_clustering_eval_ensemble(args, all_feats, all_labels, coarse_of,
                                   k_per_coarse, oracle_k_int, model_dirs,
                                   models, unknown_seqs):
    """
    Run per-coarse clustering evaluation for all five ensemble strategies.
    Saves a comprehensive *_metrics.json alongside the K results JSON.
    """
    from methods.estimate_k_hier.hier_eval_utils import (
        eval_per_coarse, save_hier_metrics,
    )

    oracle_k = {int(k): int(v) for k, v in oracle_k_int.items()}
    cluster_methods_str = getattr(args, "cluster_methods", "kmeans,agglo,spectral,gmm_tied")
    cluster_methods = [m.strip() for m in cluster_methods_str.split(",")]

    feats_normed = normalize(all_feats, norm="l2", axis=1).astype(np.float32)

    # ---- 1. Val accuracy per model ----
    val_accKs = get_val_accK_per_model(model_dirs)
    best_idx = int(np.argmax(val_accKs))
    print(f"\nVal K-acc per model: {[f'{v:.3f}' for v in val_accKs]}")
    print(f"Best model: model_{best_idx:02d} (val_accK={val_accKs[best_idx]:.3f})")

    # ---- 2. Per-model inference ----
    device = next(models[0].parameters()).device
    print("\nRunning per-model inference for ensemble strategies...")
    per_model_results = predict_ensemble_per_model(
        models=models,
        seqs=unknown_seqs,
        device=device,
        batch_size=args.transformer_batch_size,
        max_tokens_per_split=None,
        seed=args.seed,
    )

    # ---- 3. Build feature arrays ----
    N = len(all_feats)
    avg_feats, best_feats, concat_feats = _build_strategy_feats(
        per_model_results, N, args.bottleneck_dim, best_idx
    )

    # ---- 4. Backbone + avg transformer (first eval_per_coarse call) ----
    print("\nEvaluating backbone + avg-embedding strategy...")
    per_coarse = eval_per_coarse(
        feats_normed, all_labels, coarse_of, k_per_coarse, oracle_k,
        methods=cluster_methods, alt_feats=avg_feats, alt_name="avg",
    )

    # ---- 5. Best-model strategy ----
    print("Evaluating best-model strategy...")
    pc_best = eval_per_coarse(
        feats_normed, all_labels, coarse_of, k_per_coarse, oracle_k,
        methods=cluster_methods, alt_feats=best_feats, alt_name="best",
    )
    for c, d in pc_best.items():
        per_coarse[c].update({k: v for k, v in d.items() if k.startswith("best_")})

    # ---- 6. Concatenate strategy ----
    print("Evaluating concatenated-embeddings strategy...")
    pc_concat = eval_per_coarse(
        feats_normed, all_labels, coarse_of, k_per_coarse, oracle_k,
        methods=cluster_methods, alt_feats=concat_feats, alt_name="concat",
    )
    for c, d in pc_concat.items():
        per_coarse[c].update({k: v for k, v in d.items() if k.startswith("concat_")})

    # ---- 7. Distance-matrix averaging ----
    print("Evaluating distance-averaged strategy...")
    dist_results = _eval_dist_avg(per_model_results, all_labels, k_per_coarse, oracle_k)
    for c, d in dist_results.items():
        if c in per_coarse:
            per_coarse[c].update({k: v for k, v in d.items() if k not in ("k_true", "k_hat")})

    # ---- 8. Hungarian majority-vote ----
    print("Evaluating Hungarian-vote strategy...")
    vote_results = _eval_hungarian_vote(per_model_results, all_labels, k_per_coarse, oracle_k)
    for c, d in vote_results.items():
        if c in per_coarse:
            per_coarse[c].update({k: v for k, v in d.items() if k not in ("k_true", "k_hat")})

    # ---- 9. Aggregate + print ----
    agg = _aggregate_ensemble(per_coarse, cluster_methods)
    print("\n===== Ensemble strategy comparison =====")
    _print_ensemble_summary(agg, cluster_methods)

    # ---- 10. Save ----
    metrics_path = args.k_results_path.replace(".json", "_metrics.json")
    save_hier_metrics(k_per_coarse, per_coarse, agg, "transformer_ensemble",
                      args.dataset_name, metrics_path)


# ---------------------------------------------------------------------------
# Args / Main (identical to original except clustering eval section)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Set Transformer K estimation — ensemble embedding strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name",      type=str, default="scars")
    parser.add_argument("--model_name",        type=str, default="vit_dino")
    parser.add_argument("--prop_train_labels", type=float, default=1.0)
    parser.add_argument("--use_ssb_splits",    type=str2bool, default=True)
    parser.add_argument("--hier_split_type",   type=str, default=None)
    parser.add_argument("--num_novel_classes", type=int, default=None)
    parser.add_argument("--root_dir",          type=str, default=feature_extract_dir)
    parser.add_argument("--batch_size",        type=int, default=128)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--coarse_labels_path", type=str, required=True)
    parser.add_argument("--k_results_path",    type=str, required=True)
    parser.add_argument("--use_oracle_k",      type=str2bool, default=False)
    parser.add_argument("--n_models",          type=int, default=5)
    parser.add_argument("--epochs",            type=int, default=50)
    parser.add_argument("--transformer_batch_size", type=int, default=32)
    parser.add_argument("--max_tokens_per_split",   type=int, default=1024)
    parser.add_argument("--d_model",           type=int, default=128)
    parser.add_argument("--nhead",             type=int, default=8)
    parser.add_argument("--num_layers",        type=int, default=2)
    parser.add_argument("--ff_dim",            type=int, default=256)
    parser.add_argument("--dropout",           type=float, default=0.2)
    parser.add_argument("--bottleneck_dim",    type=int, default=64)
    parser.add_argument("--supcon_lambda",     type=float, default=0.3)
    parser.add_argument("--lr",                type=float, default=0.1)
    parser.add_argument("--momentum",          type=float, default=0.9)
    parser.add_argument("--weight_decay",      type=float, default=1e-4)
    parser.add_argument("--jitter_std",        type=float, default=0.02)
    parser.add_argument("--token_dropout_p",   type=float, default=0.05)
    parser.add_argument("--mixup_alpha",       type=float, default=0.2)
    parser.add_argument("--mixup_p",           type=float, default=0.5)
    parser.add_argument("--val_prop",          type=float, default=0.2)
    parser.add_argument("--seed",              type=int, default=1000)
    parser.add_argument("--device",            type=str, default="cuda:0")
    parser.add_argument("--ckpt_dir",          type=str, default=None)
    parser.add_argument("--cluster_methods",   type=str,
                        default="kmeans,agglo,spectral,gmm_tied")

    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    args = get_class_splits(args)
    args = restrict_unlabeled_classes(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    coarse_map  = load_coarse_map(args.coarse_labels_path)
    coarse_of   = build_coarse_of(coarse_map, args.train_classes, args.unlabeled_classes)
    oracle_k_int = oracle_k_per_coarse(coarse_map, args.unlabeled_classes)

    # ---- Load backbone features ----
    save_dir = os.path.join(args.root_dir, f"{args.model_name}_{args.dataset_name}")
    print(f"Loading features from {save_dir} ...")
    train_dataset, _, _, _ = get_datasets(
        args.dataset_name, train_transform=None, test_transform=None, args=args
    )
    all_feats, all_labels, all_mask = load_all_features(
        train_dataset, save_dir, args.batch_size, args.num_workers, device
    )
    print(f"Loaded {len(all_feats)} train feature vectors (dim={all_feats.shape[1]})")
    print(f"  Labeled: {all_mask.sum()} | Unlabeled: {(~all_mask).sum()}")

    # ---- Build coarse-group sequences ----
    known_seqs, unknown_seqs, Kmax, input_dim = build_coarse_sequences(
        all_feats, all_labels, all_mask, coarse_of
    )
    print(f"Known coarses: {len(known_seqs)} | Unknown coarses: {len(unknown_seqs)}")
    print(f"Kmax: {Kmax} | Feature dim: {input_dim}")

    rng = np.random.default_rng(args.seed)
    n_val = max(1, round(args.val_prop * len(known_seqs)))
    perm = rng.permutation(len(known_seqs)).tolist()
    val_seqs   = [known_seqs[i] for i in perm[:n_val]]
    train_seqs = [known_seqs[i] for i in perm[n_val:]]
    print(f"Train coarses: {len(train_seqs)} | Val coarses: {len(val_seqs)}")

    # ---- Set up checkpoint dir ----
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        base = os.path.splitext(os.path.abspath(args.k_results_path))[0]
        ckpt_dir = base + "_transformer_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Train / load ensemble ----
    print(f"\nTraining {args.n_models} model(s) for {args.epochs} epochs ...")
    model_dirs = []
    for m in range(args.n_models):
        mdir = train_one_model(
            model_id=m, input_dim=input_dim, Kmax_train=Kmax,
            train_seqs=train_seqs, val_seqs=val_seqs, ckpt_dir=ckpt_dir,
            epochs=args.epochs, batch_size=args.transformer_batch_size,
            max_tokens_per_split=args.max_tokens_per_split,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            supcon_lambda=args.supcon_lambda, jitter_std=args.jitter_std,
            token_dropout_p=args.token_dropout_p,
            mixup_alpha=args.mixup_alpha, mixup_p=args.mixup_p,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            ff_dim=args.ff_dim, dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim, device=device, seed=args.seed,
        )
        model_dirs.append(mdir)

    models = load_ensemble(
        model_dirs=model_dirs, which="best_ema", input_dim=input_dim,
        Kmax_train=Kmax, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, ff_dim=args.ff_dim, dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim, device=device,
    )
    print(f"Loaded {len(models)} models.")

    # ---- Predict K (averaged logits — identical to original) ----
    print("\nRunning ensemble inference on unknown coarses ...")
    results = predict_ensemble(
        models=models, seqs=unknown_seqs, device=device,
        batch_size=args.transformer_batch_size, max_tokens_per_split=None, seed=args.seed,
    )

    k_per_coarse = {}
    for r in results:
        sid = str(r["split_id"])
        k_hat = max(1, int(r["k_hat"]))
        print(f"  Coarse {sid}: k_hat={k_hat}  k_true={r['k_true']}  p_split={r['p_split']:.3f}")
        k_per_coarse[sid] = k_hat

    # Val K-accuracy check
    val_results = predict_ensemble(
        models=models, seqs=val_seqs, device=device,
        batch_size=args.transformer_batch_size, max_tokens_per_split=None, seed=args.seed,
    )
    k_hats  = np.array([r["k_hat"] for r in val_results])
    k_trues = np.array([r["k_true"] for r in val_results])
    print(f"Val K-accuracy: exact={np.mean(k_hats == k_trues):.3f}  "
          f"±1={np.mean(np.abs(k_hats - k_trues) <= 1):.3f} (n={len(val_results)})")

    # ---- Save K results JSON ----
    os.makedirs(os.path.dirname(os.path.abspath(args.k_results_path)), exist_ok=True)
    with open(args.k_results_path, "w") as f:
        json.dump(k_per_coarse, f, indent=2)
    print(f"\nTotal unknown clusters: {sum(k_per_coarse.values())}")

    # ---- All ensemble clustering strategies ----
    _run_clustering_eval_ensemble(
        args, all_feats, all_labels, coarse_of,
        k_per_coarse, oracle_k_int,
        model_dirs=model_dirs, models=models, unknown_seqs=unknown_seqs,
    )

    print(f"\nK estimation complete: {args.k_results_path}")


if __name__ == "__main__":
    main()
