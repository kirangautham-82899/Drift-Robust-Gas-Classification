"""Helpers for the drift visualisation (Step 3).

Everything that learns from data (scaler, PCA) is fitted on the training
batches only, then applied to all batches.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURE_COLS

TRAIN_BATCHES = (1, 2, 3)


def fit_scaler_pca(df, train_batches=TRAIN_BATCHES, n_components=2, seed=42):
    """Fit a StandardScaler and PCA on the training batches only."""
    train = df[df["batch"].isin(train_batches)]
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(scaler.transform(train[FEATURE_COLS]))
    return scaler, pca


def project(df, scaler, pca):
    """Scale then project every row of df (any batches) with the fitted objects."""
    return pca.transform(scaler.transform(df[FEATURE_COLS]))


def centroid_shift(df, scaler, train_batches=TRAIN_BATCHES):
    """Distance of each (batch, gas) centroid from that gas's training centroid.

    Distances are measured in the 128-D scaled feature space.
    Returns (table with batch rows and gas columns, mean distance between the
    training centroids of different gases).
    """
    z = pd.DataFrame(scaler.transform(df[FEATURE_COLS]), columns=FEATURE_COLS, index=df.index)
    z["batch"] = df["batch"].to_numpy()
    z["gas"] = df["gas"].to_numpy()

    train_centroid = (z[z["batch"].isin(train_batches)]
                      .groupby("gas")[FEATURE_COLS].mean())
    batch_centroid = z.groupby(["batch", "gas"])[FEATURE_COLS].mean()

    rows = {}
    for (batch, gas), centroid in batch_centroid.iterrows():
        rows.setdefault(batch, {})[gas] = np.linalg.norm(centroid.to_numpy() - train_centroid.loc[gas].to_numpy())
    table = pd.DataFrame(rows).T.sort_index()

    between = [np.linalg.norm(train_centroid.loc[a].to_numpy() - train_centroid.loc[b].to_numpy())
               for a, b in combinations(train_centroid.index, 2)]
    return table, float(np.mean(between))
