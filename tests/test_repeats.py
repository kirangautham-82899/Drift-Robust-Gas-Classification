import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import data_loader as dl
from src import evaluation as ev
from src import stacking
from src.repeats import set_all_seeds

CHEAP = {
    "kNN": {"reduction": "None", "params": {"n_neighbors": 5, "weights": "uniform"}},
    "Naive Bayes": {"reduction": "PCA", "params": {"var_smoothing": 1e-9}},
    "Decision Tree": {"reduction": "None", "params": {"max_depth": 5, "min_samples_leaf": 5}},
}


@pytest.fixture(scope="module")
def noisy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    y = (X[:, 0] + rng.normal(scale=2.0, size=300) > 0).astype(int)
    return X, y


def test_set_all_seeds_reaches_every_random_state():
    pipe = Pipeline([("scaler", StandardScaler()), ("reducer", PCA(n_components=3, random_state=1)),
                     ("model", RandomForestClassifier(n_estimators=5, random_state=2))])
    set_all_seeds(pipe, 7)
    assert pipe.named_steps["reducer"].random_state == 7
    assert pipe.named_steps["model"].random_state == 7


def test_random_forest_changes_with_seed_but_is_repeatable(noisy):
    X, y = noisy
    def probs(seed):
        pipe = Pipeline([("model", RandomForestClassifier(n_estimators=10, max_features=2, random_state=0))])
        set_all_seeds(pipe, seed)
        return pipe.fit(X, y).predict_proba(X)
    assert np.allclose(probs(3), probs(3))
    assert not np.allclose(probs(3), probs(4))


def test_deterministic_models_do_not_depend_on_the_seed(noisy):
    X, y = noisy
    for make in (lambda: KNeighborsClassifier(5), lambda: SVC(random_state=0)):
        preds = []
        for seed in (0, 1, 2):
            model = set_all_seeds(make(), seed)
            preds.append(model.fit(X, y).predict(X))
        assert all((p == preds[0]).all() for p in preds)


def test_set_all_seeds_reaches_inside_a_stack():
    stack = stacking.make_stack(CHEAP, np.array([1, 1, 2, 2, 3, 3]))
    set_all_seeds(stack, 11)
    seeds = {k: v for k, v in stack.get_params(deep=True).items() if k.endswith("random_state")}
    assert len(seeds) >= 3 and all(v == 11 for v in seeds.values())


def test_seeded_stack_is_repeatable():
    df = dl.remove_toluene(dl.load_all())
    small = df[df["batch"].isin([1, 2, 3, 4])]
    split = ev.Split("T", "train 1-3 / test 4", (1, 2, 3), (4,))
    a = stacking.evaluate_stack_split(small, split, CHEAP, seed=5)
    b = stacking.evaluate_stack_split(small, split, CHEAP, seed=5)
    assert np.allclose(a["accuracy"], b["accuracy"])
