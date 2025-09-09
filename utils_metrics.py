import numpy as np
import torch

@torch.no_grad()
def confusion_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

@torch.no_grad()
def mean_iou(preds, gts, n_classes):
    hist = np.zeros((n_classes, n_classes))
    for p, g in zip(preds, gts):
        hist += confusion_hist(g.flatten(), p.flatten(), n_classes)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-7)
    return float(np.nanmean(iu))

def compute_multiclass_fscore(mask_gt, mask_pred, beta=1):
    f_scores = []
    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))
        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-7)
        f_scores.append(f)
    return float(np.mean(f_scores))