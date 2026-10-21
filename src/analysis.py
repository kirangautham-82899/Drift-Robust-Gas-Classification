"""Bootstrap confidence intervals and confusion-matrix helpers for the per-batch drift analysis (Step 10).

Every test batch is resampled separately (rows drawn with replacement) and the SAME resampled rows are
used for every model, so differences between models are paired. Pooled results over several batches
are built by adding up the batches' resampled confusion matrices.

Caveat: rows inside a batch were recorded close together in time and are not independent, so intervals
from resampling rows are narrower than the true uncertainty.
"""
import numpy as np

from src.evaluation import CLASSES

K = len(CLASSES)


def confusion_counts(y_true, y_pred):
    """K x K count matrix: rows = true gas, columns = predicted gas (labels 1..K)."""
    codes = (np.asarray(y_true) - 1) * K + (np.asarray(y_pred) - 1)
    return np.bincount(codes, minlength=K * K).reshape(K, K)


def bootstrap_confusions(y_true, preds, n_boot=2000, rng=None):
    """Bootstrap confusion matrices of several models on the same resampled rows.

    preds: {model name: predicted labels}. Returns {model name: array of shape (n_boot, K, K)}.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    y_true = np.asarray(y_true)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    offset = (np.arange(n_boot) * K * K)[:, None]
    out = {}
    for name, y_pred in preds.items():
        codes = (y_true - 1) * K + (np.asarray(y_pred) - 1)
        out[name] = np.bincount((codes[idx] + offset).ravel(), minlength=n_boot * K * K).reshape(n_boot, K, K)
    return out


def metrics_from_confusion(cm):
    """Accuracy and macro-F1 from confusion matrices of shape (..., K, K).

    Macro-F1 averages over the gases that are present in the true labels, like evaluation.compute_metrics.
    """
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    row, col = cm.sum(axis=-1), cm.sum(axis=-2)
    accuracy = tp.sum(axis=-1) / cm.sum(axis=(-2, -1))
    denom = row + col
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = row > 0
    return accuracy, (f1 * present).sum(axis=-1) / present.sum(axis=-1)


def interval(values, level=0.95):
    """Percentile interval (low, high) of bootstrap values."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def pooled(per_batch, batches):
    """Add up the confusion matrices of several batches. per_batch: {batch: array(..., K, K)}."""
    return sum(per_batch[b] for b in batches)
