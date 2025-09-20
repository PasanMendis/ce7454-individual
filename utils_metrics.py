import numpy as np

def compute_multiclass_fscore(mask_gt, mask_pred, beta=1):
    f = []
    for cid in np.unique(mask_gt):
        tp = ((mask_gt==cid)&(mask_pred==cid)).sum()
        fp = ((mask_gt!=cid)&(mask_pred==cid)).sum()
        fn = ((mask_gt==cid)&(mask_pred!=cid)).sum()
        pre = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f.append((1+beta**2)*pre*rec/((beta**2*pre)+rec + 1e-7))
    return float(np.mean(f)) if f else 0.0

def mean_iou(preds, gts, num_classes):
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    for p,g in zip(preds, gts):
        for c in np.unique(g):
            i = ((p==c)&(g==c)).sum()
            u = ((p==c)|(g==c)).sum()
            inter[c] += i; union[c] += u
    iou = inter / (union + 1e-7)
    valid = union > 0
    return float(iou[valid].mean()) if valid.any() else 0.0