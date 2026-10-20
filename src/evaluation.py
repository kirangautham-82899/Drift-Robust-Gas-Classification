"""Time-aware evaluation protocols, scaling pipeline and metrics (Step 4).

A model is always fitted on the training batches only, and the scaler lives
inside the pipeline, so it is fitted on the training batches only as well.
"""
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURE_COLS

SEED = 42
CLASSES = [1, 2, 3, 4, 5]   # Toluene (6) has been removed
LAST_BATCH = 10

Split = namedtuple("Split", ["protocol", "name", "train_batches", "test_batches"])


def make_pipeline(model, reducer=None):
    """Scaler (+ optional PCA/LDA reducer) + model.

    Fitting this pipeline fits the scaler and the reducer on training data only.
    """
    steps = [("scaler", StandardScaler())]
    if reducer is not None:
        steps.append(("reducer", reducer))
    steps.append(("model", model))
    return Pipeline(steps)


def get_splits(protocol):
    """Return the list of train/test splits for protocol 'A', 'B' or 'C'.

    A: train on batches 1-3, test on batches 4-10 (one split, per-batch scores reported)
    B: train on batch 1 only, test on each of batches 2-10 separately
    C: rolling - train on batches 1..k-1, test on batch k (k = 2..10)
    """
    if protocol == "A":
        return [Split("A", "train 1-3 / test 4-10", (1, 2, 3), tuple(range(4, LAST_BATCH + 1)))]
    if protocol == "B":
        return [Split("B", f"train 1 / test {k}", (1,), (k,)) for k in range(2, LAST_BATCH + 1)]
    if protocol == "C":
        return [Split("C", f"train 1-{k - 1} / test {k}", tuple(range(1, k)), (k,))
                for k in range(2, LAST_BATCH + 1)]
    raise ValueError("protocol must be 'A', 'B' or 'C'")


def check_split(split):
    """Raise if train and test overlap or if test is not later in time than train."""
    if set(split.train_batches) & set(split.test_batches):
        raise ValueError(f"train and test batches overlap in {split.name}")
    if max(split.train_batches) >= min(split.test_batches):
        raise ValueError(f"test batches are not later than train batches in {split.name}")


def compute_metrics(y_true, y_pred):
    """Accuracy and macro precision / recall / F1.

    Macro averages use the gases present in y_true, so a gas absent from a
    batch cannot create an undefined score.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = sorted(np.unique(y_true))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": precision, "macro_recall": recall, "macro_f1": f1}


def confusion(y_true, y_pred):
    """Confusion matrix with rows = true gas and columns = predicted gas (labels 1-5)."""
    return confusion_matrix(y_true, y_pred, labels=CLASSES)


def evaluate_split(make_model, df, split, return_model=False):
    """Fit on the split's training batches, predict its test batches.

    Returns (metrics DataFrame, predictions DataFrame[, fitted model]).
    The metrics have one row per test batch, plus a 'pooled' row when the
    split has more than one test batch.
    """
    check_split(split)
    train = df[df["batch"].isin(split.train_batches)]
    test = df[df["batch"].isin(split.test_batches)]

    model = make_model()
    model.fit(train[FEATURE_COLS].to_numpy(), train["label"].to_numpy())
    pred = model.predict(test[FEATURE_COLS].to_numpy())

    preds = pd.DataFrame({"protocol": split.protocol, "split": split.name,
                          "batch": test["batch"].to_numpy(),
                          "y_true": test["label"].to_numpy(), "y_pred": pred})

    rows = []
    for batch, g in preds.groupby("batch"):
        rows.append({"protocol": split.protocol, "split": split.name, "test_batch": batch,
                     "n_train": len(train), "n_test": len(g),
                     **compute_metrics(g["y_true"], g["y_pred"])})
    if len(split.test_batches) > 1:
        rows.append({"protocol": split.protocol, "split": split.name, "test_batch": "pooled",
                     "n_train": len(train), "n_test": len(preds),
                     **compute_metrics(preds["y_true"], preds["y_pred"])})
    metrics = pd.DataFrame(rows)
    return (metrics, preds, model) if return_model else (metrics, preds)


def run_protocol(make_model, df, protocol):
    """Run every split of a protocol. Returns (metrics, predictions) DataFrames."""
    all_metrics, all_preds = [], []
    for split in get_splits(protocol):
        m, p = evaluate_split(make_model, df, split)
        all_metrics.append(m)
        all_preds.append(p)
    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_preds, ignore_index=True)


def summarize(metrics):
    """One-line summary of a protocol run from its per-batch rows.

    mean_* = plain average over the test batches; weighted_accuracy = accuracy
    over all test samples together (batches weighted by their size).
    """
    per_batch = metrics[metrics["test_batch"] != "pooled"]
    return {"mean_accuracy": per_batch["accuracy"].mean(),
            "mean_macro_f1": per_batch["macro_f1"].mean(),
            "weighted_accuracy": np.average(per_batch["accuracy"], weights=per_batch["n_test"]),
            "min_accuracy": per_batch["accuracy"].min()}
