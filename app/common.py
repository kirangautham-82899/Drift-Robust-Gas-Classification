"""Shared helpers for the dashboard: paths and cached data loaders.

The dashboard only *displays* things that the notebooks measured. Anything it computes
(PCA picture, batch averages) uses the same tested functions as the notebooks.
Colours and page components live in design.py and ui.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader as dl   # noqa: E402
# src.drift, src.stacking and joblib pull in scikit-learn (about 3 s to import), so they are imported inside the
# functions that need them: pages without models open without paying for it.

RESULTS = ROOT / "results"

from design import GASES  # noqa: E402  (re-exported: the five gases in their fixed order)

FEATURE_NAMES = {1: "dR (steady-state response)", 2: "normalised dR"}

FAMILIES = ["SVM", "kNN", "Decision Tree", "Random Forest", "Naive Bayes", "AdaBoost", "Gradient Boosting"]
REDUCTIONS = ["None", "PCA", "LDA"]
REDUCTION_NAMES = {"None": "All 128 features", "PCA": "PCA (95% of the variance)", "LDA": "LDA (4 components)"}
# The ten models analysed in Step 10 (family models with the reduction that scored best on validation, plus ensembles)
ANALYSIS_MODELS = ["Soft vote", "SVM", "SVM (untuned)", "Stack (LR)", "Random Forest + PCA", "Gradient Boosting + PCA",
                   "AdaBoost + PCA", "kNN + PCA", "Naive Bayes + PCA", "Decision Tree"]


@st.cache_data(show_spinner=False)
def read_result(name):
    """Read a CSV from results/ (cached: every page re-reads them on each click).

    keep_default_na=False: pandas would otherwise turn the text 'None' into NaN."""
    return pd.read_csv(RESULTS / name, keep_default_na=False)


RAW_CACHE = RESULTS / "dashboard_raw.pkl.xz"


@st.cache_data(show_spinner="Loading the dataset ...")
def load_raw():
    """All 13,910 samples of the 10 batches (6 gases).

    Read from a pre-parsed copy (written by scripts/make_dashboard_cache.py, about 3x faster than parsing the text files);
    a test checks it equals a fresh parse. Falls back to parsing the raw files when the copy is missing."""
    if RAW_CACHE.exists():
        return pd.read_pickle(RAW_CACHE)
    return dl.load_all()


@st.cache_data(show_spinner=False)
def load_data():
    """The cleaned data used in the project: Toluene removed, 12,077 samples, 5 gases."""
    return dl.remove_toluene(load_raw())


@st.cache_data(show_spinner=False)
def batch_counts():
    """Samples per batch before and after removing Toluene."""
    raw = load_raw()
    before = raw.groupby("batch").size()
    toluene = raw[raw["label"] == dl.TOLUENE_LABEL].groupby("batch").size().reindex(before.index, fill_value=0)
    return pd.DataFrame({"before": before, "Toluene removed": toluene, "after": before - toluene})


@st.cache_data(show_spinner="Fitting PCA on the training batches ...")
def pca_projection():
    """2-D PCA picture. The scaler and PCA are fitted on batches 1-3 only, then applied to every batch."""
    from src import drift
    df = load_data()
    scaler, pca = drift.fit_scaler_pca(df, n_components=2)
    coords = drift.project(df, scaler, pca)
    out = df[["batch", "gas"]].copy()
    out["PC1"], out["PC2"] = coords[:, 0], coords[:, 1]
    return out, [float(v) for v in pca.explained_variance_ratio_]


@st.cache_data(show_spinner="Measuring how far each gas has moved ...")
def centroid_shift():
    """Distance of each gas's batch average from its training average (see Step 3)."""
    from src import drift
    df = load_data()
    scaler, _ = drift.fit_scaler_pca(df)
    table, between = drift.centroid_shift(df, scaler)
    return table, between


def feature_column(sensor, offset):
    """Column name of feature `offset` (0 = dR, 1 = normalised dR, ...) of a sensor (1-16)."""
    return f"f{(sensor - 1) * 8 + 1 + offset}"


# ---------------------------------------------------------------------------------------------- model results
METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


@st.cache_data(show_spinner=False)
def all_results():
    """Every saved single-model result in one table: model, reduction, settings, protocol, test_batch, metrics.

    'Default' settings come from Steps 5-7 (library defaults); 'Tuned' settings from Step 8 (protocol A only).
    """
    s5 = read_result("step5_baseline_metrics.csv").assign(reduction="None")
    s6 = read_result("step6_reduction_metrics.csv")
    s6 = s6[s6["reduction"].isin(["PCA", "LDA"])]
    s7 = read_result("step7_boosting_metrics.csv")
    default = pd.concat([s5, s6, s7], ignore_index=True).assign(settings="Default")
    tuned = read_result("step8_tuned_protocolA.csv").assign(settings="Tuned")
    cols = ["model", "reduction", "settings", "protocol", "test_batch", "n_test"] + METRIC_COLUMNS
    out = pd.concat([default[cols], tuned[cols]], ignore_index=True)
    out["test_batch"] = out["test_batch"].astype(str)
    out[METRIC_COLUMNS] = out[METRIC_COLUMNS].astype(float) * 100
    return out


def leaderboard(results, protocol, reduction, settings):
    """Score of each model for one protocol / reduction / settings combination (in percent).

    Protocol A: pooled over the test batches 4-10. Protocols B and C: average of the per-batch scores.
    """
    d = results[(results["protocol"] == protocol) & (results["reduction"] == reduction) & (results["settings"] == settings)]
    if protocol == "A":
        table = d[d["test_batch"] == "pooled"].set_index("model")[METRIC_COLUMNS]
    else:
        table = d[d["test_batch"] != "pooled"].groupby("model")[METRIC_COLUMNS].mean()
    return table.reindex([f for f in FAMILIES if f in table.index])


# ---------------------------------------------------------------------------------------------- saved-model predictions
def analysis_models():
    """{label: how to load it} for the ten models of Step 10, chosen by the Step 8 validation rule."""
    best = read_result("step8_best_params.csv")
    models = {}
    for fam in FAMILIES:
        top = best[best["model"] == fam].sort_values("mean_macro_f1", ascending=False, kind="stable").iloc[0]
        red = top["reduction"]
        label = fam if red == "None" else f"{fam} + {red}"
        models[label] = RESULTS / "models" / f"tuned_{red.lower()}_{fam.replace(' ', '_').lower()}_protocolA.joblib"
    models["SVM (untuned)"] = RESULTS / "models" / "baseline_svm_protocolA.joblib"
    return models


@st.cache_resource(show_spinner="Loading the saved models ...")
def load_models():
    import joblib
    loaded = {label: joblib.load(path) for label, path in analysis_models().items()}
    loaded["Stack (LR)"] = joblib.load(RESULTS / "models" / "stack_protocolA.joblib")
    return loaded


PREDICTIONS_FILE = RESULTS / "dashboard_predictions.csv"


def compute_predictions():
    """True label and every model's prediction for each recording of the test batches 4-10 (protocol A).

    Written to results/dashboard_predictions.csv by scripts/make_dashboard_cache.py so that the pages open instantly.
    """
    df = load_data()
    test = df[df["batch"] >= 4].reset_index(drop=True)
    X = test[dl.FEATURE_COLS].to_numpy()
    models = load_models()
    out = test[["batch", "label"]].copy()
    for label, model in models.items():
        out[label] = model.predict(X)
    from src import stacking
    out["Soft vote"] = stacking.soft_vote(models["Stack (LR)"], X)
    return out


@st.cache_data(show_spinner="Predicting the test batches with the saved models ...")
def predictions():
    """The saved predictions if the cache file exists (fast), otherwise computed from the saved models."""
    if PREDICTIONS_FILE.exists():
        return pd.read_csv(PREDICTIONS_FILE, keep_default_na=False)
    return compute_predictions()


def confusion(y_true, y_pred):
    """5 x 5 confusion counts, rows = true gas, columns = predicted gas (gases 1-5)."""
    codes = (np.asarray(y_true) - 1) * 5 + (np.asarray(y_pred) - 1)
    return np.bincount(codes, minlength=25).reshape(5, 5)


# ---------------------------------------------------------------------------------------------- live demo
DEMO_MODELS = ["Soft vote", "Stack (LR)", "SVM", "Random Forest + PCA", "Gradient Boosting + PCA", "AdaBoost + PCA",
               "Naive Bayes + PCA", "kNN + PCA", "Decision Tree"]
MAX_UPLOAD_ROWS = 5000


def demo_predict(model_name, X):
    """Predicted gas (1-5) and a probability for each of the 5 gases, from a saved model.

    The SVM has no probabilities of its own, so its bars come from the same SVM refitted inside the stack with probability
    outputs (Platt scaling). In rare cases the top bar can then differ from the SVM's predicted gas.
    """
    models = load_models()
    stack = models["Stack (LR)"]
    from src import stacking
    base = dict(zip(stacking.BASE_FAMILIES, stack.estimators_))      # fitted base pipelines, in the stack's order
    if model_name == "Stack (LR)":
        return stack.predict(X), stack.predict_proba(X)
    if model_name == "Soft vote":
        probs = np.mean([est.predict_proba(X) for est in stack.estimators_], axis=0)
        return stack.classes_[probs.argmax(axis=1)], probs
    model = models[model_name]
    probs = base["SVM"].predict_proba(X) if model_name == "SVM" else model.predict_proba(X)
    return model.predict(X), probs


def parse_upload(frame):
    """Turn an uploaded table into (X, labels or None). Accepts 128 feature columns (named f1..f128 or not),
    optionally preceded by a gas label column (1-5). Raises ValueError with a plain message otherwise."""
    if len(frame) == 0:
        raise ValueError("The file has no rows.")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Please upload at most {MAX_UPLOAD_ROWS:,} rows (the file has {len(frame):,}).")
    labels = None
    if all(c in frame.columns for c in dl.FEATURE_COLS):
        features = frame[dl.FEATURE_COLS]
    elif frame.shape[1] == dl.N_FEATURES:
        features = frame
    elif frame.shape[1] == dl.N_FEATURES + 1:
        labels, features = frame.iloc[:, 0], frame.iloc[:, 1:]
    else:
        raise ValueError(f"Expected 128 feature columns (optionally with a gas label first), but the file has {frame.shape[1]} columns.")
    try:
        X = features.to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError("All feature columns must contain numbers only.")
    if not np.isfinite(X).all():
        raise ValueError("The features contain missing or infinite values.")
    if labels is not None:
        try:
            labels = labels.to_numpy(dtype=int)
        except (TypeError, ValueError):
            raise ValueError("The first column (the gas label) must be a whole number from 1 to 5.")
        if not set(np.unique(labels)) <= {1, 2, 3, 4, 5}:
            raise ValueError("Gas labels must be 1 (Ethanol), 2 (Ethylene), 3 (Ammonia), 4 (Acetaldehyde) or 5 (Acetone).")
    return X, labels
