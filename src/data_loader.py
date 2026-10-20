"""Load the UCI Gas Sensor Array Drift dataset and remove Toluene.

Each line of a batch file looks like:  label 1:value 2:value ... 128:value
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_FEATURES = 128
N_BATCHES = 10
TOLUENE_LABEL = 6

GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

# 16 sensors x 8 features each (feature layout from the UCI documentation)
FEATURE_COLS = [f"f{i}" for i in range(1, N_FEATURES + 1)]

# Expected sample counts, taken from counting the raw files
EXPECTED_BEFORE = {1: 445, 2: 1244, 3: 1586, 4: 161, 5: 197,
                   6: 2300, 7: 3613, 8: 294, 9: 470, 10: 3600}
EXPECTED_AFTER = {1: 371, 2: 1239, 3: 1586, 4: 161, 5: 197,
                  6: 1833, 7: 3045, 8: 276, 9: 369, 10: 3000}


def default_data_dir():
    return Path(__file__).resolve().parents[1] / "data_raw" / "Dataset"


def load_batch(path, batch_number):
    """Read one batch file into a DataFrame (batch, label, gas, f1..f128)."""
    labels, rows = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            labels.append(int(float(parts[0])))
            values = np.full(N_FEATURES, np.nan)
            for token in parts[1:]:
                idx, val = token.split(":")
                values[int(idx) - 1] = float(val)
            rows.append(values)
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    df.insert(0, "label", labels)
    df.insert(0, "batch", batch_number)
    return df


def load_all(data_dir=None):
    """Load batch1..batch10 into one DataFrame with a 'gas' name column."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    frames = [load_batch(data_dir / f"batch{b}.dat", b)
              for b in range(1, N_BATCHES + 1)]
    df = pd.concat(frames, ignore_index=True)
    df.insert(2, "gas", df["label"].map(GAS_NAMES))
    return df


def remove_toluene(df):
    """Drop every Toluene (label 6) row. The input is not modified."""
    return df[df["label"] != TOLUENE_LABEL].reset_index(drop=True)


def counts_per_batch(df):
    return df.groupby("batch").size()


def check_dataset(df, expected_counts, allowed_labels):
    """Run the integrity checks. Returns {check name: (passed, detail)}."""
    results = {}

    feats = df[FEATURE_COLS]
    results["128 feature columns"] = (
        feats.shape[1] == N_FEATURES, f"{feats.shape[1]} columns")
    results["no missing values (NaN)"] = (
        not feats.isna().any().any(), f"{int(feats.isna().sum().sum())} NaN")
    results["no infinite values"] = (
        bool(np.isfinite(feats.to_numpy()).all()), "all finite")

    total = len(df)
    results["total row count"] = (
        total == sum(expected_counts.values()),
        f"{total} rows (expected {sum(expected_counts.values())})")

    got = counts_per_batch(df).to_dict()
    results["per-batch row counts"] = (
        got == expected_counts, f"got {got}")

    labels = set(df["label"].unique())
    results["labels are allowed"] = (
        labels <= set(allowed_labels), f"labels present: {sorted(labels)}")
    return results
