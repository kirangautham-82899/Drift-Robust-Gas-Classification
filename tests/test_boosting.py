import pytest

from src import data_loader as dl
from src import evaluation as ev
from src import models


@pytest.fixture(scope="module")
def small():
    df = dl.remove_toluene(dl.load_all())
    return df[df["batch"].isin([1, 2, 3, 4])].reset_index(drop=True)


def test_boosting_names():
    assert list(models.get_boosting()) == models.BOOSTING_NAMES


def test_unknown_reduction_rejected():
    with pytest.raises(ValueError):
        models.get_boosting("ICA")


@pytest.mark.parametrize("reduction", ["None", "PCA", "LDA"])
def test_pipeline_layout(reduction):
    for make in models.get_boosting(reduction).values():
        steps = list(make().named_steps)
        assert steps == (["scaler", "model"] if reduction == "None" else ["scaler", "reducer", "model"])


@pytest.mark.parametrize("name", models.BOOSTING_NAMES)
def test_boosting_trains_and_predicts(name, small):
    split = ev.Split("T", "train 1-3 / test 4", (1, 2, 3), (4,))
    metrics, preds = ev.evaluate_split(models.get_boosting("LDA")[name], small, split)
    assert len(preds) == (small["batch"] == 4).sum()
    assert set(preds["y_pred"]) <= set(ev.CLASSES)
    assert 0.0 <= metrics.loc[0, "accuracy"] <= 1.0
