import numpy as np


def jaccard_index(y_true, y_pred, smooth=0):
    """
    :param y_true: (B x H x W)
    :param y_pred: (B x H x W)
    :param smooth: smoothing value
    """
    B = y_true.shape[0]
    assert y_true.shape == y_pred.shape
    intersection = np.zeros(B,)
    for i in range(B):
        intersection[i] = np.dot(y_true[i].ravel(), y_pred[i].ravel())
    union_data = np.zeros_like(intersection)
    for i in range(B,):
        union_data[i] = np.sum(np.abs(y_true[i].ravel()) + np.abs(y_pred[i].ravel())) - intersection[i]
    iou = np.zeros_like(intersection)
    for i in range(B):
        if union_data[i] + smooth != 0:
            iou[i] = (intersection[i] + smooth) / (union_data[i] + smooth)
    return iou

