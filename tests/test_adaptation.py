import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from src import adaptation as ad
from src import data_loader as dl
from src.data_loader import FEATURE_COLS


@pytest.fixture(scope="module")
def clean():
    return dl.remove_toluene(dl.load_all())


@pytest.fixture(scope="module")
def small():
    """Batches 1-2 for training and 3-5 for testing, as arrays."""
    df = dl.remove_toluene(dl.load_all())
    tr, te = df[df["batch"].isin([1, 2])], df[df["batch"].isin([3, 4, 5])]
    return (tr[FEATURE_COLS].to_numpy(float), tr["label"].to_numpy(), tr["batch"].to_numpy(),
            te[FEATURE_COLS].to_numpy(float), te["label"].to_numpy(), te["batch"].to_numpy())


def knn():
    return KNeighborsClassifier(5)


def test_per_batch_standardise_gives_each_batch_mean_0_and_std_1():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(5, 3, (50, 4)), rng.normal(-2, 0.5, (30, 4))])
    b = np.array([1] * 50 + [2] * 30)
    Z = ad.per_batch_standardise(X, b)
    for k in (1, 2):
        assert np.allclose(Z[b == k].mean(axis=0), 0) and np.allclose(Z[b == k].std(axis=0), 1)


def test_per_batch_standardise_of_one_batch_ignores_the_other_batches():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4)); b = np.array([1] * 40 + [2] * 40)
    X2 = X.copy(); X2[b == 2] = X2[b == 2] * 100 + 7
    assert np.allclose(ad.per_batch_standardise(X, b)[b == 1], ad.per_batch_standardise(X2, b)[b == 1])


def test_coral_matches_the_source_mean_and_covariance():
    rng = np.random.default_rng(2)
    Xs = rng.multivariate_normal([0, 0, 0], [[2, .5, 0], [.5, 1, 0], [0, 0, 3]], 4000)
    Xt = rng.multivariate_normal([4, -1, 2], [[1, 0, 0], [0, 5, 1], [0, 1, 1]], 4000)
    A = ad.coral(Xs, Xt, eps=0.0)
    assert np.allclose(A.mean(axis=0), Xs.mean(axis=0), atol=1e-8)
    assert np.allclose(np.cov(A, rowvar=False), np.cov(Xs, rowvar=False), atol=0.05)


@pytest.mark.parametrize("method", [ad.predict_per_batch_std, ad.predict_coral])
def test_tier2_methods_never_read_test_labels(small, method):
    # Tier 2 functions are not given the test labels at all; they must still predict one known gas per test row
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    p = method(knn, X_tr, y_tr, b_tr, X_te, b_te)
    assert p.shape == y_te.shape and set(p) <= set(y_tr)


def test_self_training_never_lets_a_later_batch_change_an_earlier_one(small):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    p = ad.predict_self_training(knn, X_tr, y_tr, b_tr, X_te, b_te)
    X_bad = X_te.copy()
    X_bad[b_te == 5] = np.random.default_rng(0).permutation(X_bad[b_te == 5])[:, ::-1]
    q = ad.predict_self_training(knn, X_tr, y_tr, b_tr, X_bad, b_te)
    assert (p[b_te < 5] == q[b_te < 5]).all()


def test_pick_labelled_takes_n_per_gas_per_batch_and_depends_on_seed(small):
    *_, y_te, b_te = small
    mask = ad.pick_labelled(y_te, b_te, 5, seed=0)
    for b in np.unique(b_te):
        for gas in np.unique(y_te[b_te == b]):
            n = ((b_te == b) & (y_te == gas)).sum()
            assert mask[(b_te == b) & (y_te == gas)].sum() == min(5, n)
    assert (mask == ad.pick_labelled(y_te, b_te, 5, seed=0)).all()
    assert not (mask == ad.pick_labelled(y_te, b_te, 5, seed=1)).all()


def test_few_labels_reads_only_the_labelled_recordings(small):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    labelled = ad.pick_labelled(y_te, b_te, 5, seed=0)
    p = ad.predict_few_labels(knn, X_tr, y_tr, b_tr, X_te, b_te, y_te, labelled)
    y_scrambled = y_te.copy()
    y_scrambled[~labelled] = np.random.default_rng(0).permutation(y_scrambled[~labelled])
    q = ad.predict_few_labels(knn, X_tr, y_tr, b_tr, X_te, b_te, y_scrambled, labelled)
    assert (p == q).all()


def test_few_labels_uses_no_labels_from_later_batches(small):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    labelled = ad.pick_labelled(y_te, b_te, 5, seed=0)
    p = ad.predict_few_labels(knn, X_tr, y_tr, b_tr, X_te, b_te, y_te, labelled)
    y_bad = y_te.copy(); y_bad[b_te == 5] = 1
    q = ad.predict_few_labels(knn, X_tr, y_tr, b_tr, X_te, b_te, y_bad, labelled)
    assert (p[b_te < 5] == q[b_te < 5]).all()


class Spy(BaseEstimator, ClassifierMixin):
    """Remembers the size of what it was fitted on and predicts the first class."""
    fitted_sizes = []

    def fit(self, X, y):
        Spy.fitted_sizes.append(len(X))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_evaluate_leaves_labelled_recordings_out_of_the_scores(clean):
    Spy.fitted_sizes = []
    m = ad.evaluate(None, Spy, clean, (1, 2, 3), (4, 5), n_per_gas=3, seed=0)
    n4, n5 = (clean["batch"] == 4).sum(), (clean["batch"] == 5).sum()
    assert list(m["test_batch"]) == ["4", "5", "pooled"]
    assert m.set_index("test_batch").loc["4", "n_test"] == n4 - 15       # 5 gases x 3 labels
    assert m.set_index("test_batch").loc["pooled", "n_test"] == n4 + n5 - 30
    assert Spy.fitted_sizes == [3196 + 15, 3196 + 30]                     # refit before each batch, earlier labels kept


def test_evaluate_strict_trains_on_training_batches_only(clean):
    Spy.fitted_sizes = []
    m = ad.evaluate(ad.predict_strict, Spy, clean, (1, 2, 3), (4, 5, 6))
    assert Spy.fitted_sizes == [3196]
    assert m.set_index("test_batch").loc["pooled", "n_test"] == 161 + 197 + 1833


# ------------------------------------------------------------------------------------------------ fixed label budget
from sklearn.svm import SVC   # noqa: E402


def svm_proba():
    return SVC(C=100, gamma=0.01, probability=True, random_state=0)


@pytest.mark.parametrize("strategy", ["random", "uncertain", "mixed"])
def test_label_budget_labels_exactly_budget_rows_per_batch(small, strategy):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    _, labelled = ad.predict_label_budget(svm_proba, X_tr, y_tr, b_tr, X_te, b_te, y_te, 10, strategy=strategy)
    assert all(labelled[b_te == b].sum() == 10 for b in np.unique(b_te))


@pytest.mark.parametrize("options", [{"strategy": "random"}, {"strategy": "uncertain", "pseudo": 0.5},
                                     {"strategy": "mixed", "label_weight": 5.0, "old_weight": 0.3}])
def test_label_budget_reads_only_the_labelled_recordings(small, options):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    p, labelled = ad.predict_label_budget(svm_proba, X_tr, y_tr, b_tr, X_te, b_te, y_te, 10, **options)
    y_scrambled = y_te.copy()
    y_scrambled[~labelled] = np.random.default_rng(0).permutation(y_scrambled[~labelled])
    q, labelled2 = ad.predict_label_budget(svm_proba, X_tr, y_tr, b_tr, X_te, b_te, y_scrambled, 10, **options)
    assert (labelled == labelled2).all() and (p == q).all()


def test_label_budget_never_lets_a_later_batch_change_an_earlier_one(small):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    p, _ = ad.predict_label_budget(svm_proba, X_tr, y_tr, b_tr, X_te, b_te, y_te, 10, strategy="uncertain", pseudo=0.5)
    y_bad, X_bad = y_te.copy(), X_te.copy()
    y_bad[b_te == 5] = 1
    X_bad[b_te == 5] = X_bad[b_te == 5][::-1] * 3
    q, _ = ad.predict_label_budget(svm_proba, X_tr, y_tr, b_tr, X_bad, b_te, y_bad, 10, strategy="uncertain", pseudo=0.5)
    assert (p[b_te < 5] == q[b_te < 5]).all()


def test_uncertain_strategy_picks_the_lowest_margin_rows(small):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    one = b_te == 3
    _, labelled = ad.predict_label_budget(svm_proba, X_tr, y_tr, b_tr, X_te[one], b_te[one], y_te[one], 10,
                                          strategy="uncertain")
    model = svm_proba().fit(ad.per_batch_standardise(X_tr, b_tr), y_tr)
    p = np.sort(model.predict_proba(ad.per_batch_standardise(X_te[one], b_te[one])), axis=1)
    margin = p[:, -1] - p[:, -2]
    assert margin[labelled].max() <= margin[~labelled].min()


def test_evaluate_with_a_budget_scores_only_unlabelled_rows(clean):
    m = ad.evaluate(None, svm_proba, clean, (1, 2, 3), (4, 5), budget=20, strategy="random")
    assert m.set_index("test_batch").loc["pooled", "n_test"] == 161 + 197 - 40


# ------------------------------------------------------------------------------------------------ Acetaldehyde
def test_pick_labelled_can_label_one_gas_more_than_the_others(small):
    *_, y_te, b_te = small
    mask = ad.pick_labelled(y_te, b_te, {1: 5, 2: 5, 3: 5, 4: 15, 5: 5}, seed=0)
    for b in np.unique(b_te):
        for gas in np.unique(y_te[b_te == b]):
            n = ((b_te == b) & (y_te == gas)).sum()
            assert mask[(b_te == b) & (y_te == gas)].sum() == min(15 if gas == 4 else 5, n)


def test_pick_labelled_with_equal_dict_matches_a_single_number(small):
    *_, y_te, b_te = small
    same = {g: 5 for g in range(1, 6)}
    assert (ad.pick_labelled(y_te, b_te, same, seed=3) == ad.pick_labelled(y_te, b_te, 5, seed=3)).all()


def test_pair_specialist_only_changes_rows_given_to_the_pair(small):
    X_tr, y_tr, b_tr, X_te, y_te, b_te = small
    base = ad.predict_per_batch_std(knn, X_tr, y_tr, b_tr, X_te, b_te)
    spec = ad.predict_pair_specialist(knn, X_tr, y_tr, b_tr, X_te, b_te, pair=(1, 4))
    outside = ~np.isin(base, (1, 4))
    assert (spec[outside] == base[outside]).all()
    assert set(spec[~outside]) <= {1, 4}


def test_focus_gas_scores_match_sklearn(clean):
    from sklearn.metrics import precision_score, recall_score
    m = ad.evaluate(ad.predict_strict, knn, clean, (1, 2, 3), (4, 5), focus_gas=4).set_index("test_batch")
    test = clean[clean["batch"].isin([4, 5])]
    pred = ad.predict_strict(knn, *[clean[clean["batch"] <= 3][c].to_numpy() for c in (FEATURE_COLS, "label", "batch")],
                             test[FEATURE_COLS].to_numpy(float), test["batch"].to_numpy())
    y = test["label"].to_numpy()
    assert np.isclose(m.loc["pooled", "focus_recall"], recall_score(y, pred, labels=[4], average="macro"))
    assert np.isclose(m.loc["pooled", "focus_precision"], precision_score(y, pred, labels=[4], average="macro"))
