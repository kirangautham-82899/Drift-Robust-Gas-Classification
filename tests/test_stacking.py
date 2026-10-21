import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import evaluation as ev
from src import stacking

CHEAP = {   # fast base models, only for the tests
    "kNN": {"reduction": "None", "params": {"n_neighbors": 5, "weights": "uniform"}},
    "Naive Bayes": {"reduction": "PCA", "params": {"var_smoothing": 1e-9}},
    "Decision Tree": {"reduction": "None", "params": {"max_depth": 5, "min_samples_leaf": 5}},
}


@pytest.fixture(scope="module")
def clean():
    return dl.remove_toluene(dl.load_all())


def test_leave_one_batch_out_is_a_partition_without_leakage():
    batches = np.array([1, 1, 2, 2, 2, 3])
    folds = stacking.leave_one_batch_out(batches)
    assert len(folds) == 3
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test) == list(range(len(batches)))          # every row held out exactly once
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert len(set(batches[te])) == 1                          # one whole batch per fold
        assert batches[te][0] not in set(batches[tr])              # held-out batch is not in training


def test_choose_base_configs_picks_best_validation_reduction():
    table = pd.DataFrame({
        "model": ["kNN", "kNN", "kNN", "SVM", "SVM", "SVM"],
        "reduction": ["None", "PCA", "LDA"] * 2,
        "params": ['{"n_neighbors": 1}', '{"n_neighbors": 3}', '{"n_neighbors": 5}',
                   '{"C": 1}', '{"C": 10}', '{"C": 100}'],
        "mean_macro_f1": [0.5, 0.8, 0.6, 0.9, 0.7, 0.7],
    })
    cfg = stacking.choose_base_configs(table, families=["kNN", "SVM"])
    assert cfg["kNN"] == {"reduction": "PCA", "params": {"n_neighbors": 3}}
    assert cfg["SVM"] == {"reduction": "None", "params": {"C": 1}}


def test_real_step8_table_gives_five_diverse_bases():
    best = pd.read_csv("results/step8_best_params.csv", keep_default_na=False)
    cfg = stacking.choose_base_configs(best)
    assert list(cfg) == stacking.BASE_FAMILIES
    assert all(c["reduction"] in ("None", "PCA", "LDA") for c in cfg.values())
    names = [n for n, _ in stacking.build_base_estimators(cfg)]
    assert names == stacking.BASE_FAMILIES
    svm = dict(stacking.build_base_estimators(cfg))["SVM"]
    assert svm.named_steps["model"].probability is True


def test_stack_needs_two_training_batches(clean):
    split = ev.Split("T", "train 1 / test 2", (1,), (2,))
    with pytest.raises(ValueError):
        stacking.evaluate_stack_split(clean, split, CHEAP)


def test_stack_runs_and_reports_everything(clean):
    small = clean[clean["batch"].isin([1, 2, 3, 4])]
    split = ev.Split("T", "train 1-3 / test 4", (1, 2, 3), (4,))
    metrics = stacking.evaluate_stack_split(small, split, CHEAP)
    assert set(metrics["model"]) == {"Stack (LR)", "Soft vote", "kNN", "Naive Bayes", "Decision Tree"}
    assert (metrics["n_test"] == (small["batch"] == 4).sum()).all()
    assert metrics["accuracy"].between(0, 1).all()


def test_base_models_alone_match_the_same_pipeline_fitted_independently(clean):
    """Regression test: base predictions must use gas labels 1-5, not the internal 0-4 codes."""
    small = clean[clean["batch"].isin([1, 2, 3, 4])]
    split = ev.Split("T", "train 1-3 / test 4", (1, 2, 3), (4,))
    metrics = stacking.evaluate_stack_split(small, split, CHEAP)
    train, test = small[small["batch"] <= 3], small[small["batch"] == 4]
    for name, pipe in stacking.build_base_estimators(CHEAP):
        pipe.fit(train[dl.FEATURE_COLS].to_numpy(), train["label"].to_numpy())
        expected = (pipe.predict(test[dl.FEATURE_COLS].to_numpy()) == test["label"].to_numpy()).mean()
        got = metrics[(metrics["model"] == name) & (metrics["test_batch"] == 4)]["accuracy"].iloc[0]
        assert np.isclose(got, expected), (name, got, expected)
        assert got > 0.3      # far above the ~0 that shifted labels would give


def test_changing_test_batches_cannot_change_the_fitted_stack(clean):
    split = ev.Split("T", "train 1-3 / test 4", (1, 2, 3), (4,))
    small = clean[clean["batch"].isin([1, 2, 3, 4])].copy()
    altered = small.copy()
    later = altered["batch"] == 4
    altered.loc[later, dl.FEATURE_COLS] = altered.loc[later, dl.FEATURE_COLS] * 1000 + 5
    _, s1 = stacking.evaluate_stack_split(small, split, CHEAP, return_model=True)
    _, s2 = stacking.evaluate_stack_split(altered, split, CHEAP, return_model=True)
    assert np.allclose(s1.final_estimator_.coef_, s2.final_estimator_.coef_)
    assert np.allclose(s1.final_estimator_.intercept_, s2.final_estimator_.intercept_)
