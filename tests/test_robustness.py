import numpy as np
import pytest
from sklearn.svm import SVC

from src import robustness as rb


def test_batch_bootstrap_without_resampling_noise_returns_the_point_estimate():
    # one batch only: every resample draws that batch, so every value is its accuracy
    ok = np.array([True, True, False, True])
    out = rb.batch_bootstrap({"m": ok}, np.zeros(4), n_boot=50)
    assert np.allclose(out["m"]["pooled"], 0.75) and np.allclose(out["m"]["mean"], 0.75)


def test_batch_bootstrap_resamples_whole_batches_and_is_paired():
    batches = np.array([1] * 10 + [2] * 30)
    a = np.r_[np.ones(10), np.zeros(30)].astype(bool)            # perfect on batch 1, wrong on batch 2
    b = ~a
    out = rb.batch_bootstrap({"a": a, "b": b}, batches, n_boot=2000, rng=np.random.default_rng(0))
    pooled = out["a"]["pooled"]
    assert set(np.round(np.unique(pooled), 6)) <= {0.0, 0.25, 1.0}  # only whole-batch combinations are possible
    assert np.allclose(out["a"]["pooled"] + out["b"]["pooled"], 1.0)  # same draw for both models
    assert set(np.round(np.unique(out["a"]["mean"]), 6)) <= {0.0, 0.5, 1.0}


def test_subsample_standardise_uses_only_the_sample():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=(500, 3))
    b = np.repeat([1, 2], 250)
    full = rb.subsample_standardise(X, b, 10_000, np.random.default_rng(1))
    assert np.allclose(full[b == 1].mean(axis=0), 0, atol=1e-9)          # n >= batch size: whole batch
    small = rb.subsample_standardise(X, b, 10, np.random.default_rng(1))
    assert not np.allclose(small[b == 1].mean(axis=0), 0, atol=1e-3)     # 10 rows: estimate is noisy
    assert small.shape == X.shape


@pytest.mark.parametrize("n", [5, 50])
def test_subsample_std_never_reads_test_labels(n):
    rng = np.random.default_rng(0)
    X_tr, y_tr, b_tr = rng.normal(size=(60, 4)), np.repeat([1, 2, 3], 20), np.repeat([1, 2], 30)
    X_te, b_te = rng.normal(size=(40, 4)), np.repeat([3, 4], 20)
    # the method has no argument for test labels, and its result is repeatable for a given seed
    p1 = rb.predict_subsample_std(lambda: SVC(), X_tr, y_tr, b_tr, X_te, b_te, n=n, seed=3)
    p2 = rb.predict_subsample_std(lambda: SVC(), X_tr, y_tr, b_tr, X_te, b_te, n=n, seed=3)
    assert (p1 == p2).all() and len(p1) == 40
