import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from src import data_loader as dl
from src import evaluation as ev


@pytest.fixture(scope="module")
def clean():
    return dl.remove_toluene(dl.load_all())


# ---------- protocol definitions ----------
def test_protocol_a():
    (split,) = ev.get_splits("A")
    assert split.train_batches == (1, 2, 3)
    assert split.test_batches == tuple(range(4, 11))


def test_protocol_b():
    splits = ev.get_splits("B")
    assert len(splits) == 9
    assert all(s.train_batches == (1,) for s in splits)
    assert [s.test_batches for s in splits] == [(k,) for k in range(2, 11)]


def test_protocol_c():
    splits = ev.get_splits("C")
    assert len(splits) == 9
    for k, s in zip(range(2, 11), splits):
        assert s.train_batches == tuple(range(1, k))
        assert s.test_batches == (k,)


@pytest.mark.parametrize("protocol", ["A", "B", "C"])
def test_no_overlap_and_time_order(protocol):
    for s in ev.get_splits(protocol):
        ev.check_split(s)                                   # must not raise
        assert not set(s.train_batches) & set(s.test_batches)
        assert max(s.train_batches) < min(s.test_batches)


def test_check_split_catches_bad_splits():
    with pytest.raises(ValueError):
        ev.check_split(ev.Split("X", "overlap", (1, 2), (2, 3)))
    with pytest.raises(ValueError):
        ev.check_split(ev.Split("X", "reversed", (5,), (2,)))


def test_unknown_protocol():
    with pytest.raises(ValueError):
        ev.get_splits("Z")


# ---------- no leakage ----------
class Spy(BaseEstimator, ClassifierMixin):
    """Records exactly how many rows it was fitted on."""
    def fit(self, X, y):
        self.n_fit_ = len(X)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        self.n_predict_ = len(X)
        return np.full(len(X), self.classes_[0])


def test_model_is_fitted_on_training_batches_only(clean):
    split = ev.get_splits("A")[0]
    _, _, model = ev.evaluate_split(lambda: Spy(), clean, split, return_model=True)
    assert model.n_fit_ == clean["batch"].isin([1, 2, 3]).sum()
    assert model.n_predict_ == clean["batch"].isin(range(4, 11)).sum()


def test_scaler_uses_training_statistics_only(clean):
    split = ev.get_splits("A")[0]
    _, _, model = ev.evaluate_split(lambda: ev.make_pipeline(Spy()), clean, split, return_model=True)
    scaler = model.named_steps["scaler"]
    train = clean[clean["batch"].isin([1, 2, 3])]
    assert np.allclose(scaler.mean_, train[dl.FEATURE_COLS].mean().to_numpy())
    assert not np.allclose(scaler.mean_, clean[dl.FEATURE_COLS].mean().to_numpy())


def test_changing_test_data_cannot_change_the_fitted_model(clean):
    split = ev.get_splits("A")[0]
    altered = clean.copy()
    later = altered["batch"] >= 4
    altered.loc[later, dl.FEATURE_COLS] = altered.loc[later, dl.FEATURE_COLS] * 1000 + 5

    make = lambda: ev.make_pipeline(KNeighborsClassifier(5))
    _, _, m1 = ev.evaluate_split(make, clean, split, return_model=True)
    _, _, m2 = ev.evaluate_split(make, altered, split, return_model=True)
    assert np.allclose(m1.named_steps["scaler"].mean_, m2.named_steps["scaler"].mean_)
    assert np.allclose(m1.named_steps["scaler"].scale_, m2.named_steps["scaler"].scale_)


def test_every_test_row_gets_a_prediction(clean):
    metrics, preds = ev.run_protocol(lambda: Spy(), clean, "B")
    assert len(preds) == clean["batch"].between(2, 10).sum()
    assert len(metrics) == 9


# ---------- metrics ----------
def test_compute_metrics_known_example():
    m = ev.compute_metrics([1, 1, 2, 2], [1, 2, 2, 2])
    assert np.isclose(m["accuracy"], 0.75)
    assert np.isclose(m["macro_precision"], (1.0 + 2 / 3) / 2)
    assert np.isclose(m["macro_recall"], (0.5 + 1.0) / 2)
    assert np.isclose(m["macro_f1"], (2 / 3 + 0.8) / 2)


def test_confusion_matrix_shape_and_counts():
    cm = ev.confusion([1, 1, 2, 5], [1, 2, 2, 4])
    assert cm.shape == (5, 5)
    assert cm.sum() == 4
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[4, 3] == 1


def test_perfect_prediction_scores_one():
    m = ev.compute_metrics([1, 2, 3], [1, 2, 3])
    assert all(np.isclose(v, 1.0) for v in m.values())
