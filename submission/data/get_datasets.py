from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.pets import get_pets_datasets
from data.inat_subspecies import get_inat_datasets
from data.aves_small import get_aves_small_datasets
from data.aves_full import get_aves_full_datasets


from data.cifar import subsample_classes as subsample_dataset_cifar
from data.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_air
from data.pets import subsample_classes as subsample_dataset_pets
from data.inat_subspecies import subsample_classes as subsample_dataset_inat
from data.aves_small import subsample_classes as subsample_dataset_aves_small
from data.aves_full import subsample_classes as subsample_dataset_aves_full



from copy import deepcopy
import numpy as np
import pickle
import os

from config import osr_split_dir

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars,
    'pets': subsample_dataset_pets,
    'inat_subspecies': subsample_dataset_inat,
    'aves_small': subsample_dataset_aves_small,
    'aves_full': subsample_dataset_aves_full,
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'pets': get_pets_datasets,
    'inat_subspecies': get_inat_datasets,
    'aves_small': get_aves_small_datasets,
    'aves_full': get_aves_full_datasets,
}


def restrict_unlabeled_classes(args):
    """
    If args.num_novel_classes is set, truncate args.unlabeled_classes to that many
    entries. Call this immediately after get_class_splits(args) in every entry point
    that uses the dataset, so all steps see the same restricted class set.
    """
    if getattr(args, 'num_novel_classes', None) is not None:
        args.unlabeled_classes = list(args.unlabeled_classes)[:args.num_novel_classes]
    return args


def _filter_dataset_by_classes(dataset, keep_classes):
    """
    In-place filter: keep only samples whose target is in keep_classes.
    Works with datasets that expose .targets and any of .data / .samples / .imgs.
    """
    keep_set = set(keep_classes)
    indices = [i for i, t in enumerate(dataset.targets) if t in keep_set]
    indices_arr = np.array(indices, dtype=np.intp)
    # Save targets before any mutation (targets may be a property derived from samples)
    original_targets = list(dataset.targets)

    if hasattr(dataset, 'data') and len(dataset.data) > 0:
        if isinstance(dataset.data, np.ndarray):
            dataset.data = dataset.data[indices_arr]
        elif hasattr(dataset.data, 'iloc'):  # pandas DataFrame
            dataset.data = dataset.data.iloc[indices].reset_index(drop=True)
        else:
            dataset.data = [dataset.data[i] for i in indices]
    if hasattr(dataset, 'samples'):
        dataset.samples = [dataset.samples[i] for i in indices]
    if hasattr(dataset, 'imgs'):
        dataset.imgs = [dataset.imgs[i] for i in indices]
    dataset.targets = [original_targets[i] for i in indices]
    if hasattr(dataset, 'uq_idxs'):
        dataset.uq_idxs = dataset.uq_idxs[indices_arr]
    return dataset


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)

    # Filter datasets to the classes actually in use (harmless when unlabeled_classes
    # is the full split; necessary when restrict_unlabeled_classes() has truncated it).
    all_valid_classes = set(list(args.train_classes) + list(args.unlabeled_classes))
    if datasets.get('train_unlabelled') is not None:
        _filter_dataset_by_classes(datasets['train_unlabelled'], all_valid_classes)
    if datasets.get('test') is not None:
        _filter_dataset_by_classes(datasets['test'], all_valid_classes)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            v = args.use_ssb_splits
            use_ssb_splits = v if isinstance(v, bool) else str(v).strip().lower() not in ('false', '0', 'no')
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'tinyimagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if getattr(args, 'hier_split_type', None) is not None:

            split_path = os.path.join(osr_split_dir, f'scars_hier_{args.hier_split_type}.pkl')
            with open(split_path, 'rb') as handle:
                hier_info = pickle.load(handle)
            args.train_classes = hier_info['train_classes']
            args.unlabeled_classes = hier_info['unlabeled_classes']

        elif use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']


        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224

        if getattr(args, 'hier_split_type', None) is not None:

            split_path = os.path.join(osr_split_dir, f'aircraft_hier_{args.hier_split_type}.pkl')
            with open(split_path, 'rb') as handle:
                hier_info = pickle.load(handle)
            args.train_classes = hier_info['train_classes']
            args.unlabeled_classes = hier_info['unlabeled_classes']

        elif use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if getattr(args, 'hier_split_type', None) is not None:

            split_path = os.path.join(osr_split_dir, f'cub_hier_{args.hier_split_type}.pkl')
            with open(split_path, 'rb') as handle:
                hier_info = pickle.load(handle)
            args.train_classes = hier_info['train_classes']
            args.unlabeled_classes = hier_info['unlabeled_classes']

        elif use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'pets':

        args.image_size = 224
        args.train_classes = range(19)
        args.unlabeled_classes = range(19, 37)

    elif args.dataset_name == 'chinese_traffic_signs':

        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    elif args.dataset_name == 'inat_subspecies':

        import ast
        import pandas as pd
        from config import inat_splits_csv

        args.image_size = 224

        df = pd.read_csv(inat_splits_csv)
        df['generated_species_taxon_ids'] = df['generated_species_taxon_ids'].apply(ast.literal_eval)
        all_taxon_ids = sorted(set(tid for ids in df['generated_species_taxon_ids'] for tid in ids))
        taxon_to_idx = {tid: i for i, tid in enumerate(all_taxon_ids)}

        split_type = getattr(args, 'hier_split_type', None)

        if split_type == '5050':
            split_path = os.path.join(osr_split_dir, 'inat_5050.pkl')
            with open(split_path, 'rb') as handle:
                hier_info = pickle.load(handle)
            args.train_classes = hier_info['train_classes']
            args.unlabeled_classes = hier_info['unlabeled_classes']

        elif split_type == 'train_val':
            known_taxons = [tid for ids in df[df['set'] == 'train']['generated_species_taxon_ids'] for tid in ids]
            novel_taxons = [tid for ids in df[df['set'] == 'val']['generated_species_taxon_ids'] for tid in ids]
            args.train_classes = sorted(set(taxon_to_idx[t] for t in known_taxons))
            args.unlabeled_classes = sorted(set(taxon_to_idx[t] for t in novel_taxons))

        else:
            # Default: train_test
            known_taxons = [tid for ids in df[df['set'] == 'train']['generated_species_taxon_ids'] for tid in ids]
            novel_taxons = [tid for ids in df[df['set'] == 'test']['generated_species_taxon_ids'] for tid in ids]
            args.train_classes = sorted(set(taxon_to_idx[t] for t in known_taxons))
            args.unlabeled_classes = sorted(set(taxon_to_idx[t] for t in novel_taxons))

    elif args.dataset_name == 'aves_small':

        import ast
        import pandas as pd
        from config import aves_small_splits_csv

        args.image_size = 224

        df = pd.read_csv(aves_small_splits_csv)
        df['generated_species_taxon_ids'] = df['generated_species_taxon_ids'].apply(ast.literal_eval)
        all_taxon_ids = sorted(set(tid for ids in df['generated_species_taxon_ids'] for tid in ids))
        taxon_to_idx = {tid: i for i, tid in enumerate(all_taxon_ids)}

        split_type = getattr(args, 'hier_split_type', None)
        # All rows have set=train so only pre-computed pkl splits are supported.
        if split_type == 'all_known':
            args.train_classes = list(range(len(all_taxon_ids)))
            args.unlabeled_classes = []
        elif split_type in (None, '5050'):
            split_path = os.path.join(osr_split_dir, 'aves_small_5050.pkl')
            with open(split_path, 'rb') as handle:
                hier_info = pickle.load(handle)
            args.train_classes = hier_info['train_classes']
            args.unlabeled_classes = hier_info['unlabeled_classes']
        elif split_type in ('gcd_paired', 'gcd_single'):
            split_path = os.path.join(osr_split_dir, f'aves_small_{split_type}.pkl')
            with open(split_path, 'rb') as handle:
                info = pickle.load(handle)
            args.train_classes = info['train_classes']
            args.unlabeled_classes = info['unlabeled_classes']
        else:
            raise ValueError(
                f"aves_small: unrecognised hier_split_type='{split_type}'. "
                f"Valid values: None, '5050', 'all_known', 'gcd_paired', 'gcd_single'."
            )

    elif args.dataset_name == 'aves_full':

        from data.aves_full import _build_extended_aves_index

        args.image_size = 224

        all_taxon_ids, taxon_to_idx, all_split_ids, split_to_coarse_idx = _build_extended_aves_index()
        n_fine = len(all_taxon_ids)

        split_type = getattr(args, 'hier_split_type', None)

        if split_type == 'all_known':
            # All fine classes treated as known — for feature extraction and supervised probing.
            args.train_classes = list(range(n_fine))
            args.unlabeled_classes = []
        elif split_type in (None, '5050'):
            split_path = os.path.join(osr_split_dir, 'aves_full_5050.pkl')
            with open(split_path, 'rb') as handle:
                hier_info = pickle.load(handle)
            args.train_classes = hier_info['train_classes']
            args.unlabeled_classes = hier_info['unlabeled_classes']
        elif split_type in ('gcd_paired', 'gcd_single'):
            split_path = os.path.join(osr_split_dir, f'aves_full_{split_type}.pkl')
            with open(split_path, 'rb') as handle:
                info = pickle.load(handle)
            args.train_classes = info['train_classes']
            args.unlabeled_classes = info['unlabeled_classes']
        else:
            raise ValueError(
                f"aves_full: unrecognised hier_split_type='{split_type}'. "
                f"Valid values: None, '5050', 'all_known', 'gcd_paired', 'gcd_single'."
            )

    else:

        raise NotImplementedError

    return args