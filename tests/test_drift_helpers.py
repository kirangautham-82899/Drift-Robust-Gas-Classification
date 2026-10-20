import numpy as np
import pytest

from src import data_loader as dl
from src import drift


@pytest.fixture(scope="module")
def clean():
    return dl.remove_toluene(dl.load_all())


def test_scaler_and_pca_use_training_batches_only(clean):
    scaler, pca = drift.fit_scaler_pca(clean)
    train = clean[clean["batch"].isin(drift.TRAIN_BATCHES)]
    test = clean[~clean["batch"].isin(drift.TRAIN_BATCHES)]

    # scaler statistics equal the training means, not the all-data means
    assert np.allclose(scaler.mean_, train[dl.FEATURE_COLS].mean().to_numpy())
    assert not np.allclose(scaler.mean_, clean[dl.FEATURE_COLS].mean().to_numpy())

    # training data is centred; later batches are not (that is the drift)
    z_train = scaler.transform(train[dl.FEATURE_COLS])
    z_test = scaler.transform(test[dl.FEATURE_COLS])
    assert np.allclose(z_train.mean(axis=0), 0, atol=1e-8)
    assert np.abs(z_test.mean(axis=0)).max() > 0.1


def test_project_shape(clean):
    scaler, pca = drift.fit_scaler_pca(clean, n_components=2)
    assert drift.project(clean, scaler, pca).shape == (len(clean), 2)


def test_centroid_shift_shape_and_values(clean):
    scaler, _ = drift.fit_scaler_pca(clean)
    table, between = drift.centroid_shift(clean, scaler)
    assert table.shape == (10, 5)
    assert (table.to_numpy() >= 0).all()
    assert between > 0
