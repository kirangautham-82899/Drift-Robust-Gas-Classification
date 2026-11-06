"""Robustness checks for the main findings (Step 14).

1. Batch-level bootstrap. Recordings in one batch share sensor state and recording conditions, so they are not
   independent and the row bootstrap of Step 10 gives intervals that are too narrow. Rows inside the batch files are not
   in time order either (batches 1-6 are sorted by gas and batch 10 is shuffled), so a moving-block bootstrap over rows
   has no meaning here. Resampling whole test batches (a cluster bootstrap) is the honest alternative: it asks how much
   the result depends on which batches happened to be tested. With 7 test batches the intervals are wide.
2. Standardising a new batch from a small sample. Per-batch standardisation (Step 13) needs the whole new batch before it
   can predict anything. Here each test batch is scaled with the mean and spread of only n of its recordings, drawn at
   random (the files do not keep time order, so "the first n recordings" cannot be simulated).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.adaptation import per_batch_standardise


def batch_bootstrap(correct, batches, n_boot=10000, rng=None):
    """Paired cluster bootstrap over test batches.

    correct: {model name: boolean array, True where the prediction is right}; batches: the batch of every row.
    Each resample draws len(unique batches) batches with replacement, the SAME draw for every model.
    Returns {model: {"pooled": array(n_boot), "mean": array(n_boot)}}: accuracy over all rows of the drawn batches, and
    the plain average of the drawn batches' accuracies.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    batches = np.asarray(batches)
    ids = np.unique(batches)
    n_rows = np.array([(batches == b).sum() for b in ids], dtype=float)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    counts = np.stack([np.bincount(d, minlength=len(ids)) for d in draws])          # times each batch is drawn
    out = {}
    for name, ok in correct.items():
        ok = np.asarray(ok, dtype=bool)
        hits = np.array([ok[batches == b].sum() for b in ids], dtype=float)
        out[name] = {"pooled": (counts @ hits) / (counts @ n_rows),
                     "mean": (counts @ (hits / n_rows)) / len(ids)}
    return out


def subsample_standardise(X, batches, n, rng):
    """Scale every batch with the mean and standard deviation of n of its rows drawn at random (all rows if fewer)."""
    X, batches = np.asarray(X, dtype=float), np.asarray(batches)
    out = np.empty_like(X)
    for b in np.unique(batches):
        rows = np.flatnonzero(batches == b)
        sample = rng.choice(rows, min(n, len(rows)), replace=False)
        out[rows] = StandardScaler().fit(X[sample]).transform(X[rows])
    return out


def predict_subsample_std(make_model, X_tr, y_tr, b_tr, X_te, b_te, n, seed=0):
    """Tier 2: training batches are standardised on their own in full (they are all available when training); each test
    batch is standardised with the statistics of only n of its recordings. Labels of test rows are never used."""
    rng = np.random.default_rng(seed)
    model = make_model().fit(per_batch_standardise(X_tr, b_tr), y_tr)
    return model.predict(subsample_standardise(X_te, b_te, n, rng))
