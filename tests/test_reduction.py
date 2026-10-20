import numpy as np
import pytest

from src import data_loader as dl
from src import evaluation as ev
from src import models


@pytest.fixture(scope="module")
def clean():
    return dl.remove_toluene(dl.load_all())


def test_default_pipeline_unchanged():
    assert list(models.get_baselines()["SVM"]().named_steps) == ["scaler", "model"]


@pytest.mark.parametrize("reduction", ["PCA", "LDA"])
def test_pipeline_order_with_reducer(reduction):
    pipe = models.get_baselines(reduction)["kNN"]()
    assert list(pipe.named_steps) == ["scaler", "reducer", "model"]


def test_unknown_reduction_rejected():
    with pytest.raises(ValueError):
        models.get_baselines("ICA")


def test_pca_keeps_95_percent_variance_of_training_data(clean):
    split = ev.get_splits("A")[0]
    _, _, fitted = ev.evaluate_split(models.get_baselines("PCA")["kNN"], clean, split, return_model=True)
    pca = fitted.named_steps["reducer"]
    assert pca.explained_variance_ratio_.sum() >= 0.95
    assert pca.n_components_ < 128
    # one component fewer would not reach 95%
    assert pca.explained_variance_ratio_[:-1].sum() < 0.95


def test_lda_has_four_components(clean):
    split = ev.get_splits("A")[0]
    _, _, fitted = ev.evaluate_split(models.get_baselines("LDA")["kNN"], clean, split, return_model=True)
    assert fitted.named_steps["reducer"].transform(
        fitted.named_steps["scaler"].transform(clean[dl.FEATURE_COLS].to_numpy()[:5])).shape == (5, 4)


@pytest.mark.parametrize("reduction", ["PCA", "LDA"])
def test_reducer_never_sees_test_batches(reduction, clean):
    """Changing the test batches must not change the fitted PCA / LDA at all."""
    split = ev.get_splits("A")[0]
    altered = clean.copy()
    later = altered["batch"] >= 4
    altered.loc[later, dl.FEATURE_COLS] = altered.loc[later, dl.FEATURE_COLS] * 1000 + 5
    make = models.get_baselines(reduction)["kNN"]
    _, _, m1 = ev.evaluate_split(make, clean, split, return_model=True)
    _, _, m2 = ev.evaluate_split(make, altered, split, return_model=True)
    r1, r2 = m1.named_steps["reducer"], m2.named_steps["reducer"]
    key = "components_" if reduction == "PCA" else "scalings_"
    assert np.allclose(getattr(r1, key), getattr(r2, key))
