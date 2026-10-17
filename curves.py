"""ROC / precision-recall curves and prevalence weighting in plain NumPy (no scikit-learn needed)."""
import numpy as np


def _cum(y, s, w):
    o = np.argsort(-s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(w * y)[idx]; fp = np.cumsum(w * (1 - y))[idx]
    return tp, fp, s[idx]


def roc_curve(y, s, w=None):
    y = np.asarray(y, float); s = np.asarray(s, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    tp, fp, _ = _cum(y, s, w); return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


def auroc(y, s):
    f, t = roc_curve(y, s); return float(np.sum(np.diff(f) * (t[1:] + t[:-1]) / 2))


def pr_curve(y, s, w=None):
    y = np.asarray(y, float); s = np.asarray(s, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    tp, fp, _ = _cum(y, s, w); prec = tp / (tp + fp); rec = tp / tp[-1]
    return np.r_[rec, 0][::-1], np.r_[prec, 1][::-1]


def average_precision(y, s, w=None):
    y = np.asarray(y, float); s = np.asarray(s, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    tp, fp, _ = _cum(y, s, w); prec = tp / (tp + fp); rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def prevalence_weights(y, prevalence):
    y = np.asarray(y); n_pos, n_neg = y.sum(), (1 - y).sum()
    return np.where(y == 1, 1.0, ((1 - prevalence) / prevalence) * (n_pos / max(n_neg, 1)))
