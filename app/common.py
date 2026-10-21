"""Shared helpers for the dashboard: paths, colours, cached data loaders and text boxes.

The dashboard only *displays* things that the notebooks measured. Anything it computes
(PCA picture, batch averages) uses the same tested functions as the notebooks.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader as dl   # noqa: E402
from src import drift                # noqa: E402

RESULTS = ROOT / "results"

# One colour per gas on every page (colour-blind safe Okabe-Ito colours)
GAS_COLORS = {"Ethanol": "#0072B2", "Ethylene": "#E69F00", "Ammonia": "#009E73",
              "Acetaldehyde": "#D55E00", "Acetone": "#CC79A7"}
GASES = list(GAS_COLORS)

FEATURE_NAMES = {1: "dR (steady-state response)", 2: "normalised dR"}


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
