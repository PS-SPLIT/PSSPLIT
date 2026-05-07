import json

from project_utils.cluster_utils import cluster_acc, np, linear_assignment
from sklearn.metrics import adjusted_mutual_info_score
from torch.utils.tensorboard import SummaryWriter
from typing import List


def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask]) if mask.any() else np.nan
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask]) if (~mask).any() else np.nan

    if np.isnan(old_acc):
        total_acc = new_acc
    elif np.isnan(new_acc):
        total_acc = old_acc
    else:
        total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc


def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)

    if len(y_true) == 0:
        return float('nan'), float('nan'), float('nan')

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc = old_acc / total_old_instances if total_old_instances > 0 else np.nan

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc = new_acc / total_new_instances if total_new_instances > 0 else np.nan

    return total_acc, old_acc, new_acc


def split_ami_score(y_true, y_pred, mask):
    """
    Evaluate Adjusted Mutual Information on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param y_true: All ground truth labels
    :param y_pred: All predictions
    :param mask: Mask defining two subsets (True = Old classes)
    :return: (all_ami, old_ami, new_ami)
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_ami = adjusted_mutual_info_score(y_true[mask], y_pred[mask]) if mask.any() else np.nan
    new_ami = adjusted_mutual_info_score(y_true[~mask], y_pred[~mask]) if (~mask).any() else np.nan
    all_ami = adjusted_mutual_info_score(y_true, y_pred)

    return all_ami, old_ami, new_ami


EVAL_FUNCS = {
    'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
    'ami': split_ami_score,
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, writer: SummaryWriter=None,
                        print_output=False, save_path: str=None):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log results for ALL functions.

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :param save_path: Optional path to a .jsonl file; each call appends one JSON line with all results
    :return: dict mapping each eval func name to (all_score, old_score, new_score)
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    results = {}
    for f_name in eval_funcs:

        acc_f = EVAL_FUNCS[f_name]
        all_score, old_score, new_score = acc_f(y_true, y_pred, mask)
        results[f_name] = (all_score, old_score, new_score)
        log_name = f'{save_name}_{f_name}'

        if writer is not None:
            writer.add_scalars(log_name,
                               {'Old': old_score, 'New': new_score,
                                'All': all_score}, T)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_score:.4f} | Old {old_score:.4f} | New {new_score:.4f}'
            print(print_str)

    if save_path is not None:
        entry = {'epoch': T, 'save_name': save_name, 'results': {
            k: {'all': v[0], 'old': v[1], 'new': v[2]} for k, v in results.items()
        }}
        with open(save_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    return results