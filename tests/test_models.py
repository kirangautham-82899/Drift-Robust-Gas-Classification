import numpy as np
import pytest

from src import data_loader as dl
from src import evaluation as ev
from src import models


@pytest.fixture(scope="module")
def small():
    df = dl.remove_toluene(dl.load_all())
    return df[df["batch"].isin([1, 2, 3, 4])].reset_index(drop=True)


def test_all_five_baselines_exist():
    assert list(models.get_baselines()) == models.BASELINE_NAMES


@pytest.mark.parametrize("name", models.BASELINE_NAMES)
def test_baseline_has_scaler_and_predicts(name, small):
    make = models.get_baselines()[name]
    pipe = make()
    assert list(pipe.named_steps) == ["scaler", "model"]

    split = ev.Split("T", "train 1-3 / test 4", (1, 2, 3), (4,))
    metrics, preds = ev.evaluate_split(make, small, split)
    assert len(preds) == (small["batch"] == 4).sum()
    assert set(preds["y_pred"]) <= set(ev.CLASSES)
    assert 0.0 <= metrics.loc[0, "accuracy"] <= 1.0


def test_fresh_model_each_call():
    make = models.get_baselines()["Random Forest"]
    assert make() is not make()


def test_seeded_models_are_repeatable(small):
    split = ev.Split("T", "train 1-3 / test 4", (1, 2, 3), (4,))
    make = models.get_baselines()["Random Forest"]
    a, _ = ev.evaluate_split(make, small, split)
    b, _ = ev.evaluate_split(make, small, split)
    assert np.isclose(a.loc[0, "accuracy"], b.loc[0, "accuracy"])
