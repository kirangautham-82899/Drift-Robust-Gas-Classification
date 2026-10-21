"""Random splits for the leakage check (Step 11).

Two ways of ignoring time, to compare with the time-aware protocol A:
  random_split        - the usual random 80/20 split of all rows (stratified by gas)
  size_matched_split  - random training set with exactly as many rows as protocol A's training
                        set (batches 1-3); everything else is the test set. Only the time order
                        differs from protocol A, not the amount of data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import evaluation as ev
from src.data_loader import FEATURE_COLS


def random_split(df, seed, test_size=0.2):
    """Stratified (by gas) random split. Returns positional (train_idx, test_idx)."""
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed,
                                           stratify=df["label"].to_numpy())
    return np.sort(train_idx), np.sort(test_idx)


def size_matched_split(df, seed, n_train):
    """Random training set of n_train rows; all other rows are the test set."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def evaluate_rows(make_model, df, train_idx, test_idx):
    """Fit on the training rows, test on the test rows.

    Returns pooled accuracy / macro-F1 and the accuracy inside each batch (NaN when a batch
    has no test rows).
    """
    if set(train_idx) & set(test_idx):
        raise ValueError("train and test rows overlap")
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    model = make_model()
    model.fit(train[FEATURE_COLS].to_numpy(), train["label"].to_numpy())
    pred = model.predict(test[FEATURE_COLS].to_numpy())
    out = dict(ev.compute_metrics(test["label"], pred), n_train=len(train), n_test=len(test))
    correct = pd.Series(pred == test["label"].to_numpy(), index=test["batch"].to_numpy())
    per_batch = correct.groupby(level=0).mean()
    for b in range(1, 11):
        out[f"acc_batch{b}"] = float(per_batch.get(b, np.nan))
    return out
