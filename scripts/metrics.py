import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(x, y, axis=-1):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[np.isnan(iou_)] = 1.
    return iou_


# exclude background
def batched_distance(x, y):
    try:
        per_class_iou = iou(x[:, :, None], y[:, None, :], axis=-2)
    except MemoryError:
        raise NotImplementedError

    return 1 - per_class_iou[..., 1:].mean(-1)


def calc_batched_generalised_energy_distance(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.astype(np.int64)
    samples_dist_1 = samples_dist_1.astype(np.int64)

    samples_dist_0 = samples_dist_0.reshape(*samples_dist_0.shape[:2], -1)
    samples_dist_1 = samples_dist_1.reshape(*samples_dist_1.shape[:2], -1)

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(bool)
    samples_dist_1 = eye[samples_dist_1].astype(bool)

    cross = np.mean(batched_distance(samples_dist_0, samples_dist_1), axis=(1, 2))
    diversity_0 = np.mean(batched_distance(samples_dist_0, samples_dist_0), axis=(1, 2))
    diversity_1 = np.mean(batched_distance(samples_dist_1, samples_dist_1), axis=(1, 2))
    return 2 * cross - diversity_0 - diversity_1


def batched_hungarian_matching(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.astype(np.int64)
    samples_dist_1 = samples_dist_1.astype(np.int64)

    samples_dist_0 = samples_dist_0.reshape((*samples_dist_0.shape[:2], -1))
    samples_dist_1 = samples_dist_1.reshape((*samples_dist_1.shape[:2], -1))

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(bool)
    samples_dist_1 = eye[samples_dist_1].astype(bool)

    cost_matrix = batched_distance(samples_dist_0, samples_dist_1)

    h_scores = []
    for i in range(samples_dist_0.shape[0]):
        h_scores.append((1 - cost_matrix[i])[linear_sum_assignment(cost_matrix[i])].mean())

    return h_scores

