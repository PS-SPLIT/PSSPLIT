import ast
import os
import numpy as np
import pandas as pd
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import (aves_small_splits_csv, aves_full_splits_csv,
                    aves_full_images_dir, aves_full_metadata_dir)


def _build_extended_aves_index():
    """
    Build taxon→idx and split→coarse_idx for aves_full using an extended index
    that preserves all existing aves_small indices unchanged.

    Layout:
      0 .. N_small-1  : aves_small taxon IDs (sorted, same as aves_small's own index)
      N_small ..      : new aves_full-only taxon IDs (sorted)

    Coarse IDs follow the same pattern:
      0 .. C_small-1  : aves_small split_ids (sorted)
      C_small ..      : new aves_full-only split_ids (sorted)
    """
    def _parse(path):
        df = pd.read_csv(path)
        df['generated_species_taxon_ids'] = df['generated_species_taxon_ids'].apply(ast.literal_eval)
        return df

    df_small = _parse(aves_small_splits_csv)
    df_full  = _parse(aves_full_splits_csv)

    small_taxon_ids = sorted(set(tid for ids in df_small['generated_species_taxon_ids'] for tid in ids))
    full_taxon_ids  = set(tid for ids in df_full['generated_species_taxon_ids'] for tid in ids)
    new_taxon_ids   = sorted(full_taxon_ids - set(small_taxon_ids))
    all_taxon_ids   = small_taxon_ids + new_taxon_ids
    taxon_to_idx    = {tid: i for i, tid in enumerate(all_taxon_ids)}

    small_split_ids = sorted(df_small['split_id'].unique())
    full_split_ids  = set(df_full['split_id'].unique())
    new_split_ids   = sorted(full_split_ids - set(small_split_ids))
    all_split_ids   = small_split_ids + new_split_ids
    split_to_coarse_idx = {sid: i for i, sid in enumerate(all_split_ids)}

    return all_taxon_ids, taxon_to_idx, all_split_ids, split_to_coarse_idx


class AvesFullDataset(Dataset):
    """
    Aves-full GCD dataset.

    samples: list of (abs_image_path, int_class_idx)
    """

    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.uq_idxs = np.array(range(len(self)))

    @property
    def targets(self):
        return [t for _, t in self.samples]

    @targets.setter
    def targets(self, value):
        self.samples = [(p, t) for (p, _), t in zip(self.samples, value)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.uq_idxs[index]


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True
    dataset.samples = [s for i, s in enumerate(dataset.samples) if mask[i]]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    return dataset


def subsample_classes(dataset, include_classes):
    include_set = set(include_classes)
    cls_idxs = [i for i, (_, t) in enumerate(dataset.samples) if t in include_set]
    target_xform_dict = {k: i for i, k in enumerate(sorted(include_classes))}
    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]
    return dataset


def get_aves_full_datasets(train_transform, test_transform, train_classes,
                            prop_train_labels=0.8, split_train_val=False, seed=0):
    """
    Build aves-full GCD datasets using the extended joint index.

    Image-level split: random 80% train pool / 20% test pool, per taxon, seeded.
    Only observations with dataset_pass=True are included.
    train_classes: integer indices from the extended aves_full index.
    """
    rng = np.random.default_rng(seed)

    _, taxon_to_idx, _, _ = _build_extended_aves_index()

    df = pd.read_csv(aves_full_splits_csv)
    df['generated_species_taxon_ids'] = df['generated_species_taxon_ids'].apply(ast.literal_eval)
    aves_full_taxon_ids = sorted(set(tid for ids in df['generated_species_taxon_ids'] for tid in ids))

    train_pool_samples = []
    test_pool_samples = []

    for taxon_id in aves_full_taxon_ids:
        cls_idx = taxon_to_idx[taxon_id]
        meta_path = os.path.join(aves_full_metadata_dir, f'metadata_{taxon_id}_updated.csv')
        if not os.path.exists(meta_path):
            continue
        meta_df = pd.read_csv(meta_path, usecols=['observation_id', 'dataset_pass'])
        obs_ids = meta_df.loc[meta_df['dataset_pass'] == True, 'observation_id'].tolist()
        imgs = [os.path.join(aves_full_images_dir, str(taxon_id), f'{obs_id}.jpg')
                for obs_id in obs_ids]
        if not imgs:
            continue
        perm = rng.permutation(len(imgs))
        n_train = max(1, int(round(len(imgs) * 0.8)))
        for i in perm[:n_train]:
            train_pool_samples.append((imgs[i], cls_idx))
        for i in perm[n_train:]:
            test_pool_samples.append((imgs[i], cls_idx))

    whole_training_set = AvesFullDataset(train_pool_samples, transform=train_transform)

    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set),
                                               include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled,
                                            prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    labelled_uq = set(train_dataset_labelled.uq_idxs)
    unlabelled_indices = np.array([i for i in whole_training_set.uq_idxs
                                   if i not in labelled_uq], dtype=np.int64)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), unlabelled_indices)

    test_dataset = AvesFullDataset(test_pool_samples, transform=test_transform)

    return {
        'train_labelled':   train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val':              None,
        'test':             test_dataset,
    }
