"""Write per-sample SINR location features for PS-SPLIT from the released embedding table.

Input: ``PSSPLITS/sinr/sinr_obs_embeddings.npz`` from the Kaggle dataset (one 256-d
SINR embedding per iNaturalist observation_id; see the README in that folder).

Output, in the layout used by every feature-based script in this repo:

    {feature_root}/{model_name}_{dataset}/{train|test}/{class_idx}/{uq_idx}.npy

Each file is the L2-normalised float32 embedding of the sample's observation,
saved with ``torch.save(numpy_array, path)`` like methods/clustering/extract_features.py.
Sample enumeration (class_idx, uq_idx, train/test assignment) reuses the repo's
data loaders, so the files line up with ``vit_dino_{dataset}`` image features.
With the default ``--model_name sinr_loc`` this reproduces the ``sinr_loc_aves_*``
feature folders behind the "SINR" rows of the paper.

Usage (run from submission/):
    python scripts/fanout_sinr_features.py \
        --sinr_npz <kaggle>/PSSPLITS/sinr/sinr_obs_embeddings.npz \
        --dataset_name both
    # optional: --dataset_root <kaggle>/PSSPLITS  (instead of the aves_* paths in config.py)
    #           --feature_root <dir>               (default: config.feature_extract_dir)

Then use ``--model_name sinr_loc`` with the K-estimation scripts, or
scripts/concat_features.py to build visual+geo features.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinr_common import (AVES_DATASETS, enumerate_aves_samples, load_sinr_npz,  # noqa: E402
                         override_aves_paths)


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def write_fold(records, fold_dir, embeddings, on_existing):
    n_written = n_skipped = 0
    for cls_idx, uq_idx, taxon_id, obs_id in records:
        out_path = os.path.join(fold_dir, str(cls_idx), f'{uq_idx}.npy')
        if os.path.exists(out_path):
            if on_existing == 'skip':
                n_skipped += 1
                continue
            if on_existing == 'error':
                raise FileExistsError(
                    f'{out_path} already exists. Use --on_existing skip|overwrite, or a new '
                    f'--feature_root.')
        if obs_id not in embeddings:
            raise KeyError(
                f'No SINR embedding for observation_id={obs_id} (taxon {taxon_id}, '
                f'class_idx {cls_idx}, uq_idx {uq_idx}). The embedding file does not match '
                f'this version of the dataset.')
        emb = np.asarray(embeddings[obs_id], dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f'SINR embedding of observation_id={obs_id} has norm {norm}')
        emb = emb / norm
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(emb, out_path)
        n_written += 1
    return n_written, n_skipped


def main():
    from config import feature_extract_dir

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sinr_npz', required=True,
                        help='PSSPLITS/sinr/sinr_obs_embeddings.npz (keys observation_id, emb)')
    parser.add_argument('--dataset_name', choices=list(AVES_DATASETS) + ['both'], default='both')
    parser.add_argument('--feature_root', default=feature_extract_dir,
                        help='Parent folder of the per-dataset feature folders '
                             '(default: config.feature_extract_dir = %(default)s)')
    parser.add_argument('--model_name', default='sinr_loc',
                        help='Output folder is {feature_root}/{model_name}_{dataset}')
    parser.add_argument('--dataset_root', default=None,
                        help='Kaggle PSSPLITS/ folder; overrides the aves_* paths in config.py')
    parser.add_argument('--on_existing', choices=['error', 'skip', 'overwrite'], default='error',
                        help='What to do if an output file already exists')
    parser.add_argument('--no_count_check', action='store_true',
                        help='Allow sample counts that differ from the released PS-SPLIT data')
    args = parser.parse_args()

    if args.dataset_root is not None:
        override_aves_paths(args.dataset_root)

    print(f'SINR embeddings: {args.sinr_npz} (md5 {_md5(args.sinr_npz)})')
    embeddings = load_sinr_npz(args.sinr_npz)

    datasets = list(AVES_DATASETS) if args.dataset_name == 'both' else [args.dataset_name]
    for ds in datasets:
        folds = enumerate_aves_samples(ds, check_counts=not args.no_count_check)
        ds_root = os.path.join(args.feature_root, f'{args.model_name}_{ds}')
        print(f'Writing {ds} SINR features to {ds_root}')
        for fold in ('train', 'test'):
            n_w, n_s = write_fold(folds[fold], os.path.join(ds_root, fold), embeddings,
                                  args.on_existing)
            print(f'  [{fold}] wrote {n_w} files, skipped {n_s} existing')
    print('Done.')


if __name__ == '__main__':
    main()
