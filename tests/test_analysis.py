import numpy as np
import pytest

from src import analysis
from src import evaluation as ev


@pytest.fixture
def example():
    rng = np.random.default_rng(0)
    y_true = rng.integers(1, 6, size=400)
    y_pred = np.where(rng.random(400) < 0.7, y_true, rng.integers(1, 6, size=400))
    return y_true, y_pred


def test_confusion_counts(example):
    y_true, y_pred = example
    cm = analysis.confusion_counts(y_true, y_pred)
    assert cm.shape == (5, 5) and cm.sum() == len(y_true)
    assert cm[0, 0] == ((y_true == 1) & (y_pred == 1)).sum()


def test_metrics_from_confusion_match_the_project_metric_function(example):
    y_true, y_pred = example
    acc, f1 = analysis.metrics_from_confusion(analysis.confusion_counts(y_true, y_pred))
    ref = ev.compute_metrics(y_true, y_pred)
    assert np.isclose(acc, ref["accuracy"])
    assert np.isclose(f1, ref["macro_f1"])


def test_metrics_ignore_gases_absent_from_the_true_labels():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 2, 2, 5])           # gas 5 is predicted but never true
    acc, f1 = analysis.metrics_from_confusion(analysis.confusion_counts(y_true, y_pred))
    ref = ev.compute_metrics(y_true, y_pred)
    assert np.isclose(f1, ref["macro_f1"]) and np.isclose(acc, ref["accuracy"])


def test_bootstrap_shapes_totals_and_pairing(example):
    y_true, y_pred = example
    perfect = y_true.copy()
    boot = analysis.bootstrap_confusions(y_true, {"a": y_pred, "perfect": perfect}, n_boot=200,
                                         rng=np.random.default_rng(1))
    assert boot["a"].shape == (200, 5, 5)
    assert (boot["a"].sum(axis=(1, 2)) == len(y_true)).all()
    # the perfect model is perfect in every resample (same rows are used for every model)
    acc, _ = analysis.metrics_from_confusion(boot["perfect"])
    assert np.allclose(acc, 1.0)
    # paired: the same resampled rows give the same true-label counts for both models
    assert (boot["a"].sum(axis=2) == boot["perfect"].sum(axis=2)).all()


def test_interval_contains_the_point_estimate_for_a_big_sample(example):
    y_true, y_pred = example
    boot = analysis.bootstrap_confusions(y_true, {"a": y_pred}, n_boot=1000, rng=np.random.default_rng(2))
    acc_boot, _ = analysis.metrics_from_confusion(boot["a"])
    point, _ = analysis.metrics_from_confusion(analysis.confusion_counts(y_true, y_pred))
    low, high = analysis.interval(acc_boot)
    assert low <= point <= high and high - low < 0.2


def test_interval_gets_wider_for_a_smaller_sample(example):
    y_true, y_pred = example
    widths = []
    for n in (400, 40):
        boot = analysis.bootstrap_confusions(y_true[:n], {"a": y_pred[:n]}, n_boot=1000, rng=np.random.default_rng(3))
        acc, _ = analysis.metrics_from_confusion(boot["a"])
        low, high = analysis.interval(acc)
        widths.append(high - low)
    assert widths[1] > widths[0]


def test_pooled_adds_batches():
    a = np.ones((3, 5, 5)); b = 2 * np.ones((3, 5, 5))
    assert (analysis.pooled({4: a, 5: b}, [4, 5]) == 3).all()
