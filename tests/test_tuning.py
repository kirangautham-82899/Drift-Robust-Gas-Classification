import json

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import models
from src import tuning


@pytest.fixture(scope="module")
def train_df():
    df = dl.remove_toluene(dl.load_all())
    return df[df["batch"].isin(tuning.TRAIN_BATCHES)].reset_index(drop=True)


def test_folds_are_forward_in_time_and_inside_training_batches():
    for tr, va in tuning.FOLDS:
        assert max(tr) < va
        assert set(tr) | {va} <= set(tuning.TRAIN_BATCHES)


def test_every_model_has_a_grid_and_a_builder():
    assert set(tuning.ALL_MODELS) == set(models.BASELINE_NAMES) | set(models.BOOSTING_NAMES)
    for name in tuning.ALL_MODELS:
        combos = tuning.param_combos(name)
        assert len(combos) >= 6
        assert len(combos) == len({json.dumps(c) for c in combos})   # no duplicates
        for red in ["None", "PCA", "LDA"]:
            tuning.make_tuned(name, red, combos[0])                   # builds without error


def test_params_are_applied():
    pipe = tuning.make_tuned("kNN", "None", {"n_neighbors": 9, "weights": "distance"})
    assert pipe.named_steps["model"].n_neighbors == 9
    assert pipe.named_steps["model"].weights == "distance"
    ada = tuning.make_tuned("AdaBoost", "PCA", {"n_estimators": 50, "learning_rate": 0.5, "base_depth": 2})
    assert ada.named_steps["model"].estimator.max_depth == 2
    assert ada.named_steps["model"].n_estimators == 50
    assert list(ada.named_steps) == ["scaler", "reducer", "model"]


def test_tune_refuses_test_batches():
    df = dl.remove_toluene(dl.load_all())
    with pytest.raises(ValueError):
        tuning.tune("Naive Bayes", "None", df)


def test_cv_score_is_valid_and_ignores_other_batches(train_df):
    params = {"n_neighbors": 5, "weights": "uniform"}
    a = tuning.cv_score("kNN", "None", params, train_df)
    assert 0 <= a["mean_macro_f1"] <= 1 and 0 <= a["mean_accuracy"] <= 1
    # extra later batches in the frame must not change the score (folds only touch batches 1-3)
    full = dl.remove_toluene(dl.load_all())
    b = tuning.cv_score("kNN", "None", params, full)
    assert np.isclose(a["mean_macro_f1"], b["mean_macro_f1"])


def test_tune_returns_one_row_per_config(train_df):
    res = tuning.tune("Naive Bayes", "None", train_df)
    assert len(res) == len(tuning.param_combos("Naive Bayes"))
    assert {"mean_macro_f1", "mean_accuracy", "params", "fold1_macro_f1", "fold2_macro_f1"} <= set(res.columns)


def test_select_best_rule():
    res = pd.DataFrame({
        "model": ["A", "A", "A", "B", "B"], "reduction": ["None"] * 5,
        "params": list("abcde"),
        "mean_macro_f1": [0.5, 0.7, 0.7, 0.4, 0.4],
        "mean_accuracy": [0.9, 0.6, 0.8, 0.5, 0.5],
    })
    best = tuning.select_best(res).set_index("model")
    assert best.loc["A", "params"] == "c"     # tie on F1 broken by accuracy
    assert best.loc["B", "params"] == "d"     # full tie -> first in grid order
