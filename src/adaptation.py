"""Drift adaptation that uses something from the new batches (Step 13, bonus steps B2 and B3).

Tiers (what a method may use from a test batch):
  1 strict       nothing (the setting of Steps 5-12)
  2 unlabelled   the batch's sensor readings, never its gas labels
  3 few labels   a few labelled recordings per gas from the batch; those recordings are not scored

Tier 2 and 3 numbers answer a different question from protocol A, so they are reported separately from the headline.
Every method returns one predicted gas per test row, and test batches are handled in time order, so a batch is never
helped by a later one.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURE_COLS
from src.evaluation import compute_metrics, make_pipeline


def per_batch_standardise(X, batches):
    """Scale every batch with its own mean and standard deviation (labels are never used)."""
    X, batches = np.asarray(X, dtype=float), np.asarray(batches)
    out = np.empty_like(X)
    for b in np.unique(batches):
        rows = batches == b
        out[rows] = StandardScaler().fit_transform(X[rows])
    return out


def coral(X_source, X_target, eps=1.0):
    """CORAL: give X_target the mean and covariance of X_source (Sun et al., 2016). eps regularises both covariances."""
    def power(C, p):
        w, V = np.linalg.eigh(C)
        return V @ np.diag(np.clip(w, 1e-8, None) ** p) @ V.T
    k = X_source.shape[1]
    C_s = np.cov(X_source, rowvar=False) + eps * np.eye(k)
    C_t = np.cov(X_target, rowvar=False) + eps * np.eye(k)
    return (X_target - X_target.mean(axis=0)) @ power(C_t, -0.5) @ power(C_s, 0.5) + X_source.mean(axis=0)


# ---------------------------------------------------------------------------------------------- methods
def predict_strict(make_model, X_tr, y_tr, b_tr, X_te, b_te):
    """Tier 1 reference: one scaler fitted on the training batches, as in Steps 5-12."""
    return make_pipeline(make_model()).fit(X_tr, y_tr).predict(X_te)


def predict_per_batch_std(make_model, X_tr, y_tr, b_tr, X_te, b_te):
    """Tier 2: every batch (training and test) is standardised on its own before the model sees it."""
    model = make_model().fit(per_batch_standardise(X_tr, b_tr), y_tr)
    return model.predict(per_batch_standardise(X_te, b_te))


def predict_coral(make_model, X_tr, y_tr, b_tr, X_te, b_te):
    """Tier 2: standardise with the training scaler, then CORAL-align each test batch to the training data."""
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
    for b in np.unique(b_te):
        rows = b_te == b
        Z_te[rows] = coral(Z_tr, Z_te[rows])
    return make_model().fit(Z_tr, y_tr).predict(Z_te)


def predict_self_training(make_model, X_tr, y_tr, b_tr, X_te, b_te, keep=0.5):
    """Tier 2: per-batch standardisation, then batch by batch in time order: predict the batch, and add its most
    confident `keep` share as pseudo-labelled training data for the next batch. make_model needs predict_proba."""
    Z_tr, Z_te = per_batch_standardise(X_tr, b_tr), per_batch_standardise(X_te, b_te)
    X_pool, y_pool = Z_tr, np.asarray(y_tr)
    pred = np.empty(len(Z_te), dtype=y_pool.dtype)
    for b in np.unique(b_te):
        rows = b_te == b
        model = make_model().fit(X_pool, y_pool)
        proba = model.predict_proba(Z_te[rows])
        pred[rows] = model.classes_[proba.argmax(axis=1)]
        confident = proba.max(axis=1) >= np.quantile(proba.max(axis=1), 1 - keep)
        X_pool = np.vstack([X_pool, Z_te[rows][confident]])
        y_pool = np.concatenate([y_pool, pred[rows][confident]])
    return pred


def pick_labelled(y_te, b_te, n_per_gas, seed):
    """Tier 3: choose n_per_gas random recordings of each gas in each test batch to be labelled (boolean mask)."""
    rng = np.random.default_rng(seed)
    y_te, b_te = np.asarray(y_te), np.asarray(b_te)
    mask = np.zeros(len(y_te), dtype=bool)
    for b in np.unique(b_te):
        for gas in np.unique(y_te[b_te == b]):
            rows = np.flatnonzero((b_te == b) & (y_te == gas))
            mask[rng.permutation(rows)[:n_per_gas]] = True
    return mask


def predict_few_labels(make_model, X_tr, y_tr, b_tr, X_te, b_te, y_te, labelled):
    """Tier 3: per-batch standardisation; before each test batch (in time order) the model is refitted on the training
    batches plus the labelled recordings of this and all earlier test batches. Only y_te[labelled] is ever read."""
    Z_tr, Z_te = per_batch_standardise(X_tr, b_tr), per_batch_standardise(X_te, b_te)
    y_known = np.where(labelled, np.asarray(y_te), -1)       # unlabelled rows are hidden from the method
    pred = np.empty(len(Z_te), dtype=np.asarray(y_tr).dtype)
    for b in np.unique(b_te):
        rows = b_te == b
        seen = labelled & (b_te <= b)
        model = make_model().fit(np.vstack([Z_tr, Z_te[seen]]), np.concatenate([y_tr, y_known[seen]]))
        pred[rows] = model.predict(Z_te[rows])
    return pred


# ---------------------------------------------------------------------------------------------- scoring
def evaluate(method, make_model, df, train_batches, test_batches, n_per_gas=0, seed=0):
    """Run one method on one split. Returns metric rows per test batch plus a 'pooled' row when there are several.

    For tier 3 (n_per_gas > 0) the labelled recordings are left out of the scores.
    """
    train = df[df["batch"].isin(train_batches)]
    test = df[df["batch"].isin(test_batches)]
    X_tr, y_tr, b_tr = train[FEATURE_COLS].to_numpy(float), train["label"].to_numpy(), train["batch"].to_numpy()
    X_te, y_te, b_te = test[FEATURE_COLS].to_numpy(float), test["label"].to_numpy(), test["batch"].to_numpy()

    if n_per_gas:
        labelled = pick_labelled(y_te, b_te, n_per_gas, seed)
        pred = predict_few_labels(make_model, X_tr, y_tr, b_tr, X_te, b_te, y_te, labelled)
    else:
        labelled = np.zeros(len(y_te), dtype=bool)
        pred = method(make_model, X_tr, y_tr, b_tr, X_te, b_te)

    scored = ~labelled
    rows = []
    for b in sorted(set(b_te)):
        m = scored & (b_te == b)
        rows.append({"test_batch": str(b), "n_test": int(m.sum()), **compute_metrics(y_te[m], pred[m])})
    if len(test_batches) > 1:
        rows.append({"test_batch": "pooled", "n_test": int(scored.sum()), **compute_metrics(y_te[scored], pred[scored])})
    return pd.DataFrame(rows)
