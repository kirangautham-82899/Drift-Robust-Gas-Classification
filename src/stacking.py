"""Stacking ensemble with batch-wise out-of-fold predictions (Step 9).

Base models use the tuned settings from Step 8. For every base family the reduction
(none / PCA / LDA) with the best *validation* score on batches 1-3 is used, so nothing
is chosen with test data.

The meta-model (logistic regression) is trained on out-of-fold probabilities that come
from leave-one-batch-out folds: the base models that predict batch b never saw batch b.
Everything (scaler, PCA, LDA, base models) is refitted inside each fold.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src import evaluation as ev
from src.data_loader import FEATURE_COLS
from src.repeats import set_all_seeds
from src.tuning import make_tuned

BASE_FAMILIES = ["kNN", "SVM", "Random Forest", "Gradient Boosting", "Naive Bayes"]


def choose_base_configs(best_params, families=BASE_FAMILIES):
    """For each family take the reduction with the best Step 8 validation macro-F1.

    best_params needs the columns model, reduction, params (JSON text), mean_macro_f1.
    """
    configs = {}
    for family in families:
        rows = best_params[best_params["model"] == family]
        top = rows.sort_values("mean_macro_f1", ascending=False, kind="stable").iloc[0]
        configs[family] = {"reduction": top["reduction"], "params": json.loads(top["params"])}
    return configs


def build_base_estimators(configs):
    """List of (name, fitted-from-scratch pipeline). SVM gets probability outputs switched on."""
    estimators = []
    for family, cfg in configs.items():
        pipe = make_tuned(family, cfg["reduction"], cfg["params"])
        if family == "SVM":
            pipe.set_params(model__probability=True)
        estimators.append((family, pipe))
    return estimators


def leave_one_batch_out(batches):
    """Folds that hold out one whole batch at a time (a partition of all rows)."""
    batches = np.asarray(batches)
    idx = np.arange(len(batches))
    return [(idx[batches != b], idx[batches == b]) for b in sorted(np.unique(batches))]


def make_stack(configs, batches):
    """Stacking classifier whose meta-model is trained on leave-one-batch-out predictions."""
    return StackingClassifier(
        estimators=build_base_estimators(configs),
        final_estimator=LogisticRegression(max_iter=2000, random_state=ev.SEED),
        cv=leave_one_batch_out(batches),
        stack_method="predict_proba",
        n_jobs=1,
    )


def soft_vote(stack, X):
    """Average the base models' probabilities (no learning) and take the most likely gas."""
    probs = np.mean([est.predict_proba(X) for est in stack.estimators_], axis=0)
    return stack.classes_[probs.argmax(axis=1)]


def _metric_rows(split, n_train, preds, label_col_name):
    rows = []
    for batch in sorted(set(preds["batch"])):
        g = preds[preds["batch"] == batch]
        rows.append({"protocol": split.protocol, "split": split.name, "test_batch": batch,
                     "n_train": n_train, "n_test": len(g), "model": label_col_name,
                     **ev.compute_metrics(g["y_true"], g["y_pred"])})
    if preds["batch"].nunique() > 1:
        rows.append({"protocol": split.protocol, "split": split.name, "test_batch": "pooled",
                     "n_train": n_train, "n_test": len(preds), "model": label_col_name,
                     **ev.compute_metrics(preds["y_true"], preds["y_pred"])})
    return rows


def evaluate_stack_split(df, split, configs, return_model=False, seed=None):
    """Fit the stack on the split's training batches, then score the stack, a soft vote and
    each base model on the test batches. Needs at least two training batches.

    seed: if given, every random_state inside the stack is set to it (used by the repeated runs)."""
    ev.check_split(split)
    if len(split.train_batches) < 2:
        raise ValueError("batch-wise folds need at least two training batches")
    train = df[df["batch"].isin(split.train_batches)]
    test = df[df["batch"].isin(split.test_batches)]
    X_train, y_train = train[FEATURE_COLS].to_numpy(), train["label"].to_numpy()
    X_test = test[FEATURE_COLS].to_numpy()

    stack = make_stack(configs, train["batch"].to_numpy())
    if seed is not None:
        set_all_seeds(stack, seed)
    stack.fit(X_train, y_train)

    predictions = {"Stack (LR)": stack.predict(X_test), "Soft vote": soft_vote(stack, X_test)}
    for name, est in zip(configs, stack.estimators_):
        # StackingClassifier fits its base models on labels renumbered 0..4, so map back to gas labels
        predictions[name] = stack.classes_[est.predict(X_test)]

    rows = []
    for model_name, y_pred in predictions.items():
        preds = pd.DataFrame({"batch": test["batch"].to_numpy(),
                              "y_true": test["label"].to_numpy(), "y_pred": y_pred})
        rows += _metric_rows(split, len(train), preds, model_name)
    metrics = pd.DataFrame(rows)
    return (metrics, stack) if return_model else metrics
