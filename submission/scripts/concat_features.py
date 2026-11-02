"""Concatenate two per-sample feature folders (e.g. image features + SINR features).

Both inputs use the layout written by methods/clustering/extract_features.py and
scripts/fanout_sinr_features.py:

    {feat_dir}/{train|test}/{class_idx}/{uq_idx}.npy     (torch.save(numpy_array, path))

For every file in feat_dir_a the file with the same relative path in feat_dir_b is
loaded, the two vectors are concatenated as [a | b] and (by default) L2-normalised
again, and the result is saved under output_dir with the same relative path.
Every file in A must have a partner in B and the two folders must contain the same
number of files per split; otherwise the script stops.

Paper feature sets (run from submission/; F = osr_novel_categories/extracted_features):
    "DINOv2 SINR" / "DINOv2, w/ Geo":
        python scripts/concat_features.py --feat_dir_a F/vit_dino_aves_small \
            --feat_dir_b F/sinr_loc_aves_small --output_dir F/vit_dino_sinr_loc_aves_small
    "SelEx, w/ Geo" (SelEx features from bash_scripts/estimate_k_trained_backbone.sh):
        python scripts/concat_features.py \
            --feat_dir_a osr_novel_categories/extracted_features_trained/selex_hier_ptl1_aves_small/vit_dino_aves_small \
            --feat_dir_b F/sinr_loc_aves_small --output_dir F/selex_sinr_loc_aves_small

DINOv2 (768-d, unit norm) + SINR (256-d, unit norm) gives 1024-d unit-norm vectors,
i.e. each part contributes 1/sqrt(2) of the norm. The K-estimation scripts find the
output with --model_name vit_dino_sinr_loc (or selex_sinr_loc) --root_dir F.
"""
import argparse
import os

import numpy as np
import torch


def _load(path):
    # Feature files are pickled numpy arrays written by this repo's own scripts;
    # weights_only=False is needed to read them with torch >= 2.6.
    arr = torch.load(path, weights_only=False)
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    return np.asarray(arr, dtype=np.float32)


def _count_npy(split_dir):
    n = 0
    for cls_name in os.listdir(split_dir):
        cls_dir = os.path.join(split_dir, cls_name)
        if os.path.isdir(cls_dir):
            n += sum(1 for f in os.listdir(cls_dir) if f.endswith('.npy'))
    return n


def concat_split(split_dir_a, split_dir_b, split_dir_out, normalize, on_existing):
    n_written = n_skipped = 0
    dims = set()
    for cls_name in sorted(os.listdir(split_dir_a), key=int):
        cls_dir_a = os.path.join(split_dir_a, cls_name)
        if not os.path.isdir(cls_dir_a):
            continue
        cls_dir_b = os.path.join(split_dir_b, cls_name)
        if not os.path.isdir(cls_dir_b):
            raise FileNotFoundError(
                f'Class folder missing in feat_dir_b: {cls_dir_b}. The two feature sets were '
                f'built from different dataset versions.')
        cls_dir_out = os.path.join(split_dir_out, cls_name)
        os.makedirs(cls_dir_out, exist_ok=True)

        for fname in sorted(os.listdir(cls_dir_a)):
            if not fname.endswith('.npy'):
                continue
            out_path = os.path.join(cls_dir_out, fname)
            if os.path.exists(out_path):
                if on_existing == 'skip':
                    n_skipped += 1
                    continue
                if on_existing == 'error':
                    raise FileExistsError(
                        f'{out_path} already exists. Use --on_existing skip|overwrite, or a new '
                        f'--output_dir.')
            path_a = os.path.join(cls_dir_a, fname)
            path_b = os.path.join(cls_dir_b, fname)
            if not os.path.exists(path_b):
                raise FileNotFoundError(
                    f'{path_b} missing (partner of {path_a}). The two feature sets were built '
                    f'from different dataset versions.')
            vec_a = _load(path_a)
            vec_b = _load(path_b)
            if vec_a.ndim != 1 or vec_b.ndim != 1:
                raise ValueError(f'Expected 1-D vectors, got {vec_a.shape} and {vec_b.shape} '
                                 f'for {path_a}')
            combined = np.concatenate([vec_a, vec_b], axis=0)
            if normalize:
                norm = float(np.linalg.norm(combined))
                if not np.isfinite(norm) or norm < 1e-12:
                    raise ValueError(f'Combined vector for {path_a} has norm {norm}')
                combined = combined / norm
            dims.add((vec_a.shape[0], vec_b.shape[0]))
            torch.save(combined, out_path)
            n_written += 1
    if len(dims) > 1:
        raise ValueError(f'Inconsistent feature dimensions in {split_dir_a}: {sorted(dims)}')
    return n_written, n_skipped, dims


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--feat_dir_a', required=True,
                        help='First feature folder (e.g. vit_dino_aves_small)')
    parser.add_argument('--feat_dir_b', required=True,
                        help='Second feature folder (e.g. sinr_loc_aves_small)')
    parser.add_argument('--output_dir', required=True,
                        help='Output folder (e.g. vit_dino_sinr_loc_aves_small)')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--no_normalize', action='store_true',
                        help='Do not L2-normalise the concatenated vector (paper: normalised)')
    parser.add_argument('--on_existing', choices=['error', 'skip', 'overwrite'], default='error',
                        help='What to do if an output file already exists')
    args = parser.parse_args()

    for split in args.splits:
        split_dir_a = os.path.join(args.feat_dir_a, split)
        split_dir_b = os.path.join(args.feat_dir_b, split)
        if not os.path.isdir(split_dir_a):
            raise FileNotFoundError(f'{split_dir_a} not found')
        if not os.path.isdir(split_dir_b):
            raise FileNotFoundError(f'{split_dir_b} not found')
        n_a, n_b = _count_npy(split_dir_a), _count_npy(split_dir_b)
        if n_a != n_b:
            raise RuntimeError(
                f'[{split}] {n_a} files in {split_dir_a} but {n_b} in {split_dir_b}. Re-extract '
                f'one of the two feature sets.')
        split_dir_out = os.path.join(args.output_dir, split)
        os.makedirs(split_dir_out, exist_ok=True)
        n_w, n_s, dims = concat_split(split_dir_a, split_dir_b, split_dir_out,
                                      not args.no_normalize, args.on_existing)
        dim_str = ', '.join(f'{a}+{b}={a + b}' for a, b in sorted(dims)) or 'n/a'
        print(f'  [{split}] wrote {n_w} files, skipped {n_s} existing (dims {dim_str})')
    print('Done.')


if __name__ == '__main__':
    main()
