import numpy as np
import torch
from torch.utils.data import Dataset


def load_coarse_map(path):
    """Load a coarse label JSON file and return {int(fine_idx): int(coarse_idx)}."""
    import json
    with open(path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items() if not k.startswith('_')}


def build_coarse_of(coarse_map, train_classes, unlabeled_classes):
    """
    Build a numpy array coarse_of such that coarse_of[remapped_label] = coarse_class_id.
    all_classes[i] is the original class index that maps to remapped label i after get_datasets().
    """
    # Must match the order used by target_transform_dict in get_datasets.py:
    # enumerate(list(train_classes) + list(unlabeled_classes)) — NOT sorted.
    all_classes = list(train_classes) + list(unlabeled_classes)
    coarse_of = np.array([coarse_map[c] for c in all_classes], dtype=np.int64)
    return coarse_of


def oracle_k_per_coarse(coarse_map, unlabeled_classes):
    """
    Count fine classes per coarse category for unknown coarse categories.
    Returns {coarse_id: k} where k = number of unlabeled fine classes in that coarse.
    """
    from collections import Counter
    unlabeled_set = set(unlabeled_classes)
    counts = Counter()
    for orig_idx, coarse_id in coarse_map.items():
        if orig_idx in unlabeled_set:
            counts[coarse_id] += 1
    return dict(counts)

def verify_coarse_constraint(train_classes, unlabeled_classes, coarse_map):
    """
    Assert that no coarse category spans both train_classes and unlabeled_classes.
    Raises ValueError listing any mixed coarses if the constraint is violated.
    """
    train_set = set(train_classes)
    unlabeled_set = set(unlabeled_classes)

    coarse_train = {}
    coarse_unlabeled = {}
    for fine, coarse in coarse_map.items():
        if fine in train_set:
            coarse_train.setdefault(coarse, []).append(fine)
        if fine in unlabeled_set:
            coarse_unlabeled.setdefault(coarse, []).append(fine)

    mixed = set(coarse_train) & set(coarse_unlabeled)
    if mixed:
        details = []
        for c in sorted(mixed):
            details.append(
                f"  coarse {c}: train={coarse_train[c]}, unlabeled={coarse_unlabeled[c]}"
            )
        raise ValueError(
            f"Coarse constraint violated: {len(mixed)} coarse(s) span both splits:\n"
            + '\n'.join(details)
        )


def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


class CoarseMergedDataset(Dataset):

    """
    Like MergedDataset but also returns a coarse label for each sample.
    Requires coarse_of: np.ndarray of shape [num_classes] where
    coarse_of[remapped_fine_label] = coarse_class_id.
    Build with build_coarse_of() after get_class_splits().

    Returns: (img, label, uq_idx, coarse_label, np.array([labeled_or_not]))
    """

    def __init__(self, labelled_dataset, unlabelled_dataset, coarse_of):
        self.labelled_dataset   = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.coarse_of          = coarse_of
        self.target_transform   = None

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        coarse_label = int(self.coarse_of[label])
        return img, label, uq_idx, coarse_label, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


class CoarseBalancedBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that gives stable coarse-class representation throughout an epoch.

    Each batch contains `samples_per_coarse` samples from `coarses_per_batch` coarses.
    Coarse appearances are spread evenly through the epoch using jittered LDS scheduling
    so a coarse with little data does not have all its batches clustered at one end of
    the epoch. Exhausted coarses are NOT reshuffled — their quota of batches simply ends.

    Parameters
    ----------
    coarse_indices : dict {coarse_id -> list of int}
        Sample indices in the train dataset belonging to each coarse class.
    batch_size : int
    min_coarse_samples : int
        Minimum samples drawn per coarse per batch. Actual value is
        max(min_coarse_samples, batch_size // num_coarses).
    """

    def __init__(self, coarse_indices: dict, batch_size: int,
                 min_coarse_samples: int = 4):
        super().__init__(None)
        self.coarse_indices     = {c: np.asarray(v) for c, v in coarse_indices.items()}
        self.coarses            = sorted(self.coarse_indices.keys())
        self.num_coarses        = len(self.coarses)
        self.batch_size         = batch_size
        self.samples_per_coarse = max(min_coarse_samples, batch_size // self.num_coarses)
        self.coarses_per_batch  = batch_size // self.samples_per_coarse

        # How many batches each coarse contributes to per epoch (at least 1)
        self.appearances = {
            c: max(1, len(v) // self.samples_per_coarse)
            for c, v in self.coarse_indices.items()
        }
        total_slots       = sum(self.appearances.values())
        self._num_batches = total_slots // self.coarses_per_batch

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = np.random.default_rng()

        # Jittered LDS: assign each coarse-slot a fractional epoch position so
        # appearances are spread uniformly rather than clustered.
        slots = []
        for c in self.coarses:
            n = self.appearances[c]
            positions = (np.arange(n) + rng.uniform(0.0, 1.0, n)) / n
            slots.extend(zip(positions.tolist(), [c] * n))
        slots.sort(key=lambda x: x[0])
        schedule_flat = [c for _, c in slots]

        # Pre-shuffled queues — no reshuffling mid-epoch.
        queues = {c: rng.permutation(idxs).tolist()
                  for c, idxs in self.coarse_indices.items()}
        ptrs = {c: 0 for c in self.coarses}

        for b in range(self._num_batches):
            batch_coarses = schedule_flat[
                b * self.coarses_per_batch : (b + 1) * self.coarses_per_batch
            ]
            batch = []
            for c in batch_coarses:
                end = min(ptrs[c] + self.samples_per_coarse, len(queues[c]))
                batch.extend(queues[c][ptrs[c]:end])
                ptrs[c] = end
            rng.shuffle(batch)
            yield batch
