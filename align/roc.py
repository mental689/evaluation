import numpy as np
from align.jaccard import jaccard_index


def precision_recall(y_true, y_pred, iou=0.5):
    B = y_true.shape[0]
    assert y_true.shape == y_pred.shape
    scores = np.zeros((B,2))
    ious = jaccard_index(y_true, y_pred, smooth=0.)
    for i in range(B):
        iou_score = ious[i]
        if iou_score >= iou:
            yt, yp = y_true[i].ravel(), y_pred[i].ravel()
            tp = np.dot(yt,yp).sum()
            fp = np.dot(1-yt,yp).sum()
            fn = np.dot(yt, 1-yp).sum()
            if tp + fp != 0:
                scores[i,0] = tp / (tp+ fp)
            if tp + fn != 0:
                scores[i,1] = tp / (tp + fn)
    return scores


def compute_ap(precisions_recalls_matrix, num_points=101):
    points = np.arange(num_points,dtype=np.float32) / num_points
    precisions = np.zeros((num_points-1,))
    ap = 0.
    for i in range(num_points-1):
        pmax = 0.
        for j in range(precisions_recalls_matrix.shape[0]):
            if precisions_recalls_matrix[j,1] >= points[i+1]:
                if pmax <  precisions_recalls_matrix[j,0]:
                    pmax = precisions_recalls_matrix[j,0]
        ap += (points[i+1]-points[i]) * pmax
    return ap

def compute_map(y_true, y_pred, iou=(0.05, 0.95), iou_step=0.05, num_points=101):
    ap = []
    iou_points = []
    i = 0
    while True:
        if iou[0] + iou_step * i < iou[1]:
            iou_points.append(iou[0] + iou_step*i)
            i+=1
        else:
            break
    for i in iou_points:
        pr_mat = precision_recall(y_true, y_pred, iou=i)
        ap.append(compute_ap(pr_mat, num_points=num_points))
    return np.array(ap).mean()
