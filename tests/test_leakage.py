import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src import data_loader as dl
from src import leakage


@pytest.fixture(scope="module")
def clean():
    return dl.remove_toluene(dl.load_all())


def test_random_split_is_a_disjoint_partition_with_80_20(clean):
    tr, te = leakage.random_split(clean, seed=0)
    assert not set(tr) & set(te)
    assert sorted(np.concatenate([tr, te])) == list(range(len(clean)))
    assert abs(len(te) / len(clean) - 0.2) < 0.001


def test_random_split_keeps_gas_proportions(clean):
    tr, te = leakage.random_split(clean, seed=0)
    full = clean["label"].value_counts(normalize=True).sort_index()
    test = clean.iloc[te]["label"].value_counts(normalize=True).sort_index()
    assert np.allclose(full, test, atol=0.003)


def test_random_split_mixes_batches_unlike_the_time_aware_split(clean):
    tr, te = leakage.random_split(clean, seed=0)
    assert set(clean.iloc[tr]["batch"]) == set(range(1, 11))
    assert set(clean.iloc[te]["batch"]) == set(range(1, 11))     # test rows come from every batch, even batch 1


def test_random_split_is_reproducible_and_seed_dependent(clean):
    a, _ = leakage.random_split(clean, seed=1)
    b, _ = leakage.random_split(clean, seed=1)
    c, _ = leakage.random_split(clean, seed=2)
    assert (a == b).all() and not (a == c).all()


def test_size_matched_split_has_protocol_a_sizes(clean):
    n_train = int(clean["batch"].isin([1, 2, 3]).sum())
    assert n_train == 3196
    tr, te = leakage.size_matched_split(clean, seed=0, n_train=n_train)
    assert len(tr) == 3196 and len(te) == len(clean) - 3196 == 8881
    assert not set(tr) & set(te)
    assert set(clean.iloc[tr]["batch"]) == set(range(1, 11))


class Spy(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.n_fit_ = len(X)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_evaluate_rows_trains_only_on_training_rows(clean):
    tr, te = leakage.random_split(clean, seed=0)
    out = leakage.evaluate_rows(lambda: Spy(), clean, tr, te)
    assert out["n_train"] == len(tr) and out["n_test"] == len(te)
    assert 0 <= out["accuracy"] <= 1
    assert all(f"acc_batch{b}" in out for b in range(1, 11))


def test_evaluate_rows_rejects_overlapping_rows(clean):
    with pytest.raises(ValueError):
        leakage.evaluate_rows(lambda: Spy(), clean, np.array([0, 1, 2]), np.array([2, 3]))
