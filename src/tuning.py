"""Hyperparameter tuning with time-aware, batch-wise cross-validation (Step 8).

Tuning only ever sees the training batches (1-3). Two forward-chaining folds are used,
which mimic the real task of predicting a later batch from earlier ones:
    fold 1: train on batch 1,      validate on batch 2
    fold 2: train on batches 1-2,  validate on batch 3
The configuration with the best mean macro-F1 over the folds is selected.
The test batches (4-10) are never loaded into any function in this module.
"""
import itertools
import json
import time

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src import models
from src.data_loader import FEATURE_COLS
from src.evaluation import SEED, compute_metrics

TRAIN_BATCHES = (1, 2, 3)
FOLDS = [((1,), 2), ((1, 2), 3)]   # (training batches, validation batch)

GRIDS = {
    "kNN": {"n_neighbors": [1, 3, 5, 9, 15, 25], "weights": ["uniform", "distance"]},
    "SVM": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1]},
    "Decision Tree": {"max_depth": [3, 5, 10, 20, None], "min_samples_leaf": [1, 5, 20]},
    "Random Forest": {"n_estimators": [100, 300], "max_depth": [10, None],
                      "min_samples_leaf": [1, 5], "max_features": ["sqrt", "log2"]},
    "Naive Bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]},
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0],
                 "base_depth": [1, 2]},
    "Gradient Boosting": {"n_estimators": [100, 200], "max_depth": [2, 3, 5]},
}
ALL_MODELS = list(GRIDS)


def param_combos(model):
    """All parameter dictionaries of the model's grid."""
    grid = GRIDS[model]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def make_tuned(model, reduction, params):
    """Build a fresh pipeline for the model with the given hyperparameters."""
    params = dict(params)
    factory = (models.get_boosting if model in models.BOOSTING_NAMES else models.get_baselines)(reduction)[model]
    pipe = factory()
    if model == "AdaBoost":
        depth = params.pop("base_depth")
        pipe.set_params(model__estimator=DecisionTreeClassifier(max_depth=depth, random_state=SEED))
    pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
    return pipe


def cv_score(model, reduction, params, train_df):
    """Forward-chaining cross-validation score of one configuration."""
    out = {"fit_seconds": 0.0}
    f1s, accs = [], []
    for i, (tr, va) in enumerate(FOLDS, start=1):
        tr_rows = train_df[train_df["batch"].isin(tr)]
        va_rows = train_df[train_df["batch"] == va]
        t0 = time.perf_counter()
        pipe = make_tuned(model, reduction, params)
        pipe.fit(tr_rows[FEATURE_COLS].to_numpy(), tr_rows["label"].to_numpy())
        pred = pipe.predict(va_rows[FEATURE_COLS].to_numpy())
        out["fit_seconds"] += time.perf_counter() - t0
        m = compute_metrics(va_rows["label"], pred)
        f1s.append(m["macro_f1"])
        accs.append(m["accuracy"])
        out[f"fold{i}_macro_f1"] = m["macro_f1"]
        out[f"fold{i}_accuracy"] = m["accuracy"]
    out["mean_macro_f1"] = sum(f1s) / len(f1s)
    out["mean_accuracy"] = sum(accs) / len(accs)
    return out


def tune(model, reduction, train_df):
    """Score every configuration of the model's grid. Returns one row per configuration."""
    if not set(train_df["batch"].unique()) <= set(TRAIN_BATCHES):
        raise ValueError("tuning data must contain training batches 1-3 only")
    rows = []
    for params in param_combos(model):
        rows.append({"model": model, "reduction": reduction,
                     "params": json.dumps(params), **cv_score(model, reduction, params, train_df)})
    return pd.DataFrame(rows)


def select_best(results):
    """Best configuration per (model, reduction): highest mean macro-F1, then mean accuracy,
    then the first one in grid order."""
    results = results.reset_index(drop=True).assign(_order=lambda d: d.index)
    ranked = results.sort_values(["mean_macro_f1", "mean_accuracy", "_order"],
                                 ascending=[False, False, True], kind="stable")
    best = ranked.groupby(["model", "reduction"], sort=False).head(1)
    return best.drop(columns="_order").reset_index(drop=True)
