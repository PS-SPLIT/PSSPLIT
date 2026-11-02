"""Shared helpers for the geographic (SINR) feature scripts in this folder.

* ``override_aves_paths(dataset_root)`` points the PS-SPLIT data loaders at a
  Kaggle download (``.../PSSPLITS``) instead of the paths in ``config.py``.
* ``enumerate_aves_samples(dataset_name)`` lists every train/test sample exactly
  as ``methods/clustering/extract_features.py`` does (same data loaders, same
  arguments), so the per-sample files written by the SINR scripts line up
  one-to-one with the image features:

      {feature_root}/{model_name}_{dataset}/{train|test}/{class_idx}/{uq_idx}.npy

* ``load_sinr_npz(path)`` loads the observation_id -> SINR embedding table.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

AVES_DATASETS = ('aves_small', 'aves_full')

# Sample counts of the released PS-SPLIT v1 data (dataset_pass=True rows; per-taxon
# 80/20 train/test split with seed 0). A mismatch means the metadata CSVs or the
# splits CSVs differ from the ones used in the paper.
EXPECTED_COUNTS = {
    'aves_small': {'train': 20230, 'test': 5055},
    'aves_full':  {'train': 63239, 'test': 15822},
}


def _set_module_attr(module, name, value):
    if not hasattr(module, name):
        raise AttributeError(
            f'{module.__name__} has no attribute {name!r}; the data loader changed and '
            f'scripts/sinr_common.py:override_aves_paths needs updating.')
    setattr(module, name, value)


def override_aves_paths(dataset_root):
    """Use ``{dataset_root}/aves_*`` instead of the aves_* paths in config.py.

    ``dataset_root`` is the Kaggle ``PSSPLITS/`` folder (it contains
    ``aves_splits_{small,full}.csv``, ``aves_metadata_{small,full}/`` and
    ``aves_photos_cropped/``). Images are never opened by the SINR scripts; only
    their file names are used.
    """
    import data.aves_small as aves_small
    import data.aves_full as aves_full

    p = {
        'aves_small_splits_csv':   os.path.join(dataset_root, 'aves_splits_small.csv'),
        'aves_small_metadata_dir': os.path.join(dataset_root, 'aves_metadata_small'),
        'aves_small_images_dir':   os.path.join(dataset_root, 'aves_photos_cropped'),
        'aves_full_splits_csv':    os.path.join(dataset_root, 'aves_splits_full.csv'),
        'aves_full_metadata_dir':  os.path.join(dataset_root, 'aves_metadata_full'),
        'aves_full_images_dir':    os.path.join(dataset_root, 'aves_photos_cropped'),
    }
    for key in ('aves_small_splits_csv', 'aves_full_splits_csv'):
        if not os.path.isfile(p[key]):
            raise FileNotFoundError(f'--dataset_root {dataset_root}: {p[key]} not found.')
    for key in ('aves_small_metadata_dir', 'aves_full_metadata_dir'):
        if not os.path.isdir(p[key]):
            raise FileNotFoundError(f'--dataset_root {dataset_root}: {p[key]} not found.')

    for name in ('aves_small_splits_csv', 'aves_small_metadata_dir', 'aves_small_images_dir'):
        _set_module_attr(aves_small, name, p[name])
    for name in ('aves_small_splits_csv', 'aves_full_splits_csv',
                 'aves_full_metadata_dir', 'aves_full_images_dir'):
        _set_module_attr(aves_full, name, p[name])
    print(f'Using PS-SPLIT files under {dataset_root} (config.py aves_* paths ignored)')


def _check_config_paths(dataset_name):
    import data.aves_small as aves_small
    import data.aves_full as aves_full
    if dataset_name == 'aves_small':
        paths = [aves_small.aves_small_splits_csv, aves_small.aves_small_metadata_dir]
    else:
        paths = [aves_full.aves_small_splits_csv, aves_full.aves_full_splits_csv,
                 aves_full.aves_full_metadata_dir]
    for path in paths:
        if str(path).startswith('<PATH_TO') or not os.path.exists(path):
            raise FileNotFoundError(
                f'{dataset_name}: {path!r} does not exist. Set the aves_* paths in config.py '
                f'or pass --dataset_root <kaggle download>/PSSPLITS.')


def _check_metadata_complete(dataset_name):
    """data/aves_*.py skips a taxon whose metadata CSV is missing; fail instead."""
    import data.aves_small as aves_small
    import data.aves_full as aves_full
    if dataset_name == 'aves_small':
        df = aves_small._load_splits_csv()
        taxa, _, _, _ = aves_small._build_taxon_index(df)
        meta_dir = aves_small.aves_small_metadata_dir
    else:
        import ast
        import pandas as pd
        df = pd.read_csv(aves_full.aves_full_splits_csv)
        taxa = sorted(set(t for ids in df['generated_species_taxon_ids'].apply(ast.literal_eval)
                          for t in ids))
        meta_dir = aves_full.aves_full_metadata_dir
    missing = [t for t in taxa
               if not os.path.isfile(os.path.join(meta_dir, f'metadata_{t}_updated.csv'))]
    if missing:
        raise FileNotFoundError(
            f'{dataset_name}: {len(missing)} of {len(taxa)} metadata CSVs missing in {meta_dir} '
            f'(first: metadata_{missing[0]}_updated.csv).')


def enumerate_aves_samples(dataset_name, check_counts=True):
    """Return {'train': [...], 'test': [...]} with one record per sample.

    Each record is ``(class_idx, uq_idx, taxon_id, observation_id)``, in the order
    the data loader yields them. Built with the same calls as
    methods/clustering/extract_features.py (all fine classes as train_classes,
    prop_train_labels=1.0, target_transform=None), so ``class_idx``/``uq_idx``
    match the image-feature folders.
    """
    if dataset_name not in AVES_DATASETS:
        raise ValueError(f'dataset_name must be one of {AVES_DATASETS}, got {dataset_name!r}')
    _check_config_paths(dataset_name)
    _check_metadata_complete(dataset_name)

    if dataset_name == 'aves_small':
        from data.aves_small import (get_aves_small_datasets, _load_splits_csv,
                                     _build_taxon_index)
        all_taxon_ids, taxon_to_idx, _, _ = _build_taxon_index(_load_splits_csv())
        datasets = get_aves_small_datasets(
            train_transform=None, test_transform=None,
            train_classes=list(range(len(all_taxon_ids))), prop_train_labels=1.0)
    else:
        from data.aves_full import get_aves_full_datasets, _build_extended_aves_index
        all_taxon_ids, taxon_to_idx, _, _ = _build_extended_aves_index()
        datasets = get_aves_full_datasets(
            train_transform=None, test_transform=None,
            train_classes=list(range(len(all_taxon_ids))), prop_train_labels=1.0)

    out = {}
    for fold, key in (('train', 'train_labelled'), ('test', 'test')):
        ds = datasets[key]
        records = []
        for (path, cls_idx), uq_idx in zip(ds.samples, ds.uq_idxs):
            taxon_id = int(os.path.basename(os.path.dirname(path)))
            stem, ext = os.path.splitext(os.path.basename(path))
            if ext != '.jpg':
                raise RuntimeError(f'Unexpected image file name {path!r}')
            obs_id = int(stem)
            if taxon_to_idx[taxon_id] != cls_idx:
                raise RuntimeError(
                    f'class index mismatch for {path!r}: loader says {cls_idx}, '
                    f'taxon index says {taxon_to_idx[taxon_id]}')
            records.append((int(cls_idx), int(uq_idx), taxon_id, obs_id))
        out[fold] = records

    counts = {fold: len(r) for fold, r in out.items()}
    print(f'[{dataset_name}] samples: train={counts["train"]} test={counts["test"]}')
    if check_counts and counts != EXPECTED_COUNTS[dataset_name]:
        raise RuntimeError(
            f'[{dataset_name}] sample counts {counts} differ from the released PS-SPLIT data '
            f'{EXPECTED_COUNTS[dataset_name]}. Check the metadata/splits files, or pass '
            f'--no_count_check if this is intentional.')
    return out


def load_sinr_npz(path):
    """Load ``{observation_id: embedding}`` from an npz with keys observation_id, emb."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'SINR embedding file not found: {path}')
    z = np.load(path, allow_pickle=False)
    for key in ('observation_id', 'emb'):
        if key not in z.files:
            raise KeyError(f'{path}: expected keys observation_id, emb; found {z.files}')
    ids = z['observation_id']
    emb = z['emb']
    if emb.ndim != 2 or len(ids) != len(emb):
        raise ValueError(f'{path}: observation_id {ids.shape} / emb {emb.shape} do not match')
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate observation_ids')
    print(f'Loaded {len(ids)} SINR embeddings (dim {emb.shape[1]}, {emb.dtype}) from {path}')
    return {int(k): v for k, v in zip(ids.tolist(), emb)}
