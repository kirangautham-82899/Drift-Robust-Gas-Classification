"""Small helpers for the exploratory data analysis (Step 2)."""
import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLS

# The 8 features of each sensor, in file order (from the UCI documentation)
FEATURE_TYPES = [
    "dR", "norm_dR",
    "rise_a0.001", "rise_a0.01", "rise_a0.1",
    "decay_a0.001", "decay_a0.01", "decay_a0.1",
]


def feature_info():
    """One row per feature column: its sensor number (1-16) and feature type."""
    rows = []
    for i, col in enumerate(FEATURE_COLS):
        rows.append({"feature": col, "sensor": i // 8 + 1, "type": FEATURE_TYPES[i % 8]})
    return pd.DataFrame(rows)


def type_summary(df):
    """Pooled min / median / max / std of each feature type across all 16 sensors."""
    info = feature_info()
    out = []
    for t in FEATURE_TYPES:
        cols = info.loc[info["type"] == t, "feature"]
        vals = df[cols].to_numpy().ravel()
        out.append({"type": t, "min": vals.min(), "median": np.median(vals),
                    "max": vals.max(), "std": vals.std()})
    return pd.DataFrame(out).set_index("type")


def class_percent_per_batch(df):
    """Percentage of each gas inside every batch (rows add up to 100)."""
    counts = df.groupby(["batch", "gas"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def high_correlation_share(df, threshold=0.9):
    """Share of feature pairs whose absolute correlation is above the threshold."""
    corr = df[FEATURE_COLS].corr().abs().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float((upper > threshold).mean())
