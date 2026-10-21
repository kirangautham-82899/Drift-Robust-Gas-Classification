"""Shared helpers for the dashboard: paths, colours, cached data loaders and text boxes.

The dashboard only *displays* things that the notebooks measured. Anything it computes
(PCA picture, batch averages) uses the same tested functions as the notebooks.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader as dl   # noqa: E402
from src import drift                # noqa: E402
from src import stacking             # noqa: E402

RESULTS = ROOT / "results"

# One colour per gas on every page (colour-blind safe Okabe-Ito colours)
GAS_COLORS = {"Ethanol": "#0072B2", "Ethylene": "#E69F00", "Ammonia": "#009E73",
              "Acetaldehyde": "#D55E00", "Acetone": "#CC79A7"}
GASES = list(GAS_COLORS)

FEATURE_NAMES = {1: "dR (steady-state response)", 2: "normalised dR"}

# One colour per model family on every page
FAMILY_COLORS = {"SVM": "#0072B2", "kNN": "#666666", "Decision Tree": "#999933", "Random Forest": "#009E73",
                 "Naive Bayes": "#332288", "AdaBoost": "#882255", "Gradient Boosting": "#E69F00"}
FAMILIES = list(FAMILY_COLORS)
REDUCTIONS = ["None", "PCA", "LDA"]
REDUCTION_NAMES = {"None": "All 128 features", "PCA": "PCA (95% of the variance)", "LDA": "LDA (4 components)"}

# The ten models analysed in Step 10 (family models with the reduction that scored best on validation, plus ensembles)
MODEL_COLORS = {"Soft vote": "#D55E00", "SVM": "#0072B2", "SVM (untuned)": "#56B4E9", "Stack (LR)": "#CC79A7",
                "Random Forest + PCA": "#009E73", "Gradient Boosting + PCA": "#E69F00", "AdaBoost + PCA": "#882255",
                "kNN + PCA": "#666666", "Naive Bayes + PCA": "#332288", "Decision Tree": "#999933"}


def read_result(name):
    """Read a CSV from results/. keep_default_na=False: pandas would otherwise turn the text 'None' into NaN."""
    return pd.read_csv(RESULTS / name, keep_default_na=False)


@st.cache_data(show_spinner="Loading the dataset ...")
def load_raw():
    """All 13,910 samples of the 10 batches (6 gases)."""
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
    df = load_data()
    scaler, pca = drift.fit_scaler_pca(df, n_components=2)
    coords = drift.project(df, scaler, pca)
    out = df[["batch", "gas"]].copy()
    out["PC1"], out["PC2"] = coords[:, 0], coords[:, 1]
    return out, [float(v) for v in pca.explained_variance_ratio_]


@st.cache_data(show_spinner="Measuring how far each gas has moved ...")
def centroid_shift():
    """Distance of each gas's batch average from its training average (see Step 3)."""
    df = load_data()
    scaler, _ = drift.fit_scaler_pca(df)
    table, between = drift.centroid_shift(df, scaler)
    return table, between


def notice(text):
    """A short plain-English box under a chart."""
    st.info(text, icon=":material/lightbulb:")


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
    loaded = {label: joblib.load(path) for label, path in analysis_models().items()}
    loaded["Stack (LR)"] = joblib.load(RESULTS / "models" / "stack_protocolA.joblib")
    return loaded


@st.cache_data(show_spinner="Predicting the test batches with the saved models ...")
def predictions():
    """True label and every model's prediction for each recording of the test batches 4-10 (protocol A)."""
    df = load_data()
    test = df[df["batch"] >= 4].reset_index(drop=True)
    X = test[dl.FEATURE_COLS].to_numpy()
    models = load_models()
    out = test[["batch", "label"]].copy()
    for label, model in models.items():
        out[label] = model.predict(X)
    out["Soft vote"] = stacking.soft_vote(models["Stack (LR)"], X)
    return out


def confusion(y_true, y_pred):
    """5 x 5 confusion counts, rows = true gas, columns = predicted gas (gases 1-5)."""
    codes = (np.asarray(y_true) - 1) * 5 + (np.asarray(y_pred) - 1)
    return np.bincount(codes, minlength=25).reshape(5, 5)


def rgba(hex_colour, alpha):
    """'#RRGGBB' -> 'rgba(r, g, b, alpha)' for translucent bands."""
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"
