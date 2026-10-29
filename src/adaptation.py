"""Drift adaptation that uses something from the new batches (Step 13, bonus steps B2 and B3).

Tiers (what a method may use from a test batch):
  1 strict       nothing (the setting of Steps 5-12)
  2 unlabelled   the batch's sensor readings, never its gas labels
  3 few labels   a few labelled recordings from the batch (per gas, or a fixed budget); those are not scored

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
    """Tier 3: choose n_per_gas random recordings of each gas in each test batch to be labelled (boolean mask).

    n_per_gas is one number for every gas, or a {gas: number} dict to label some gases more than others.
    """
    rng = np.random.default_rng(seed)
    y_te, b_te = np.asarray(y_te), np.asarray(b_te)
    mask = np.zeros(len(y_te), dtype=bool)
    for b in np.unique(b_te):
        for gas in np.unique(y_te[b_te == b]):
            rows = np.flatnonzero((b_te == b) & (y_te == gas))
            n = n_per_gas[gas] if isinstance(n_per_gas, dict) else n_per_gas
            mask[rng.permutation(rows)[:n]] = True
    return mask


def predict_pair_specialist(make_model, X_tr, y_tr, b_tr, X_te, b_te, pair=(1, 4), make_specialist=None):
    """Tier 2: per-batch standardisation; a main model predicts every gas, then a second model trained only on the two
    gases in `pair` (default Ethanol and Acetaldehyde) re-decides every row the main model gave to one of them."""
    Z_tr, Z_te = per_batch_standardise(X_tr, b_tr), per_batch_standardise(X_te, b_te)
    y_tr = np.asarray(y_tr)
    pred = make_model().fit(Z_tr, y_tr).predict(Z_te)
    in_pair = np.isin(y_tr, pair)
    specialist = (make_specialist or make_model)().fit(Z_tr[in_pair], y_tr[in_pair])
    redo = np.isin(pred, pair)
    if redo.any():
        pred[redo] = specialist.predict(Z_te[redo])
    return pred


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


def predict_label_budget(make_model, X_tr, y_tr, b_tr, X_te, b_te, y_te, budget, strategy="random", seed=0,
                         label_weight=1.0, old_weight=1.0, pseudo=0.0):
    """Tier 3 with a fixed number of labels per batch, chosen without knowing the gas. Per-batch standardisation; test
    batches in time order. Returns (predictions, boolean mask of the labelled rows).

    strategy: 'random', 'uncertain' (lowest margin between the two most likely gases under the current model) or
              'mixed' (half uncertain, half random).
    label_weight / old_weight: sample weights of the new labels / of the training batches (needs a model that takes
              sample_weight when either is not 1).
    pseudo: share of each batch's unlabelled rows, the most confident, added as pseudo-labels for later batches.
    Only y_te of the labelled rows is ever read.
    """
    rng = np.random.default_rng(seed)
    Z_tr, Z_te = per_batch_standardise(X_tr, b_tr), per_batch_standardise(X_te, b_te)
    y_te = np.asarray(y_te)
    X_pool, y_pool, w_pool = Z_tr, np.asarray(y_tr), np.full(len(y_tr), old_weight)
    weighted = label_weight != 1.0 or old_weight != 1.0

    def fit(X, y, w):
        return make_model().fit(X, y, sample_weight=w) if weighted else make_model().fit(X, y)

    pred = np.empty(len(Z_te), dtype=y_pool.dtype)
    labelled = np.zeros(len(Z_te), dtype=bool)
    for b in np.unique(b_te):
        rows = np.flatnonzero(b_te == b)
        if strategy == "random":
            pick = rng.choice(rows, budget, replace=False)
        else:
            p = np.sort(fit(X_pool, y_pool, w_pool).predict_proba(Z_te[rows]), axis=1)
            by_margin = rows[np.argsort(p[:, -1] - p[:, -2], kind="stable")]
            if strategy == "uncertain":
                pick = by_margin[:budget]
            elif strategy == "mixed":
                unsure = by_margin[:budget // 2]
                pick = np.concatenate([unsure, rng.choice(np.setdiff1d(rows, unsure), budget - len(unsure), replace=False)])
            else:
                raise ValueError("strategy must be 'random', 'uncertain' or 'mixed'")
        labelled[pick] = True
        X_pool, y_pool = np.vstack([X_pool, Z_te[pick]]), np.concatenate([y_pool, y_te[pick]])
        w_pool = np.concatenate([w_pool, np.full(len(pick), label_weight)])
        model = fit(X_pool, y_pool, w_pool)
        proba = model.predict_proba(Z_te[rows])
        pred[rows] = model.classes_[proba.argmax(axis=1)]
        if pseudo:
            free = ~labelled[rows]
            sure = free & (proba.max(axis=1) >= np.quantile(proba.max(axis=1)[free], 1 - pseudo))
            X_pool, y_pool = np.vstack([X_pool, Z_te[rows][sure]]), np.concatenate([y_pool, pred[rows][sure]])
            w_pool = np.concatenate([w_pool, np.ones(sure.sum())])
    return pred, labelled


# ---------------------------------------------------------------------------------------------- scoring
def evaluate(method, make_model, df, train_batches, test_batches, n_per_gas=0, seed=0, budget=0, focus_gas=None,
             **label_options):
    """Run one method on one split. Returns metric rows per test batch plus a 'pooled' row when there are several.

    For tier 3 (n_per_gas > 0, or budget > 0 with predict_label_budget's options) the labelled recordings are left out
    of the scores. With focus_gas, every row also gets that gas's recall and precision ('focus_recall',
    'focus_precision'; NaN where the batch has none of it).
    """
    train = df[df["batch"].isin(train_batches)]
    test = df[df["batch"].isin(test_batches)]
    X_tr, y_tr, b_tr = train[FEATURE_COLS].to_numpy(float), train["label"].to_numpy(), train["batch"].to_numpy()
    X_te, y_te, b_te = test[FEATURE_COLS].to_numpy(float), test["label"].to_numpy(), test["batch"].to_numpy()

    if budget:
        pred, labelled = predict_label_budget(make_model, X_tr, y_tr, b_tr, X_te, b_te, y_te, budget, seed=seed,
                                              **label_options)
    elif n_per_gas:
        labelled = pick_labelled(y_te, b_te, n_per_gas, seed)
        pred = predict_few_labels(make_model, X_tr, y_tr, b_tr, X_te, b_te, y_te, labelled)
    else:
        labelled = np.zeros(len(y_te), dtype=bool)
        pred = method(make_model, X_tr, y_tr, b_tr, X_te, b_te)

    def scores(m):
        out = compute_metrics(y_te[m], pred[m])
        if focus_gas is not None:
            is_gas, said_gas = y_te[m] == focus_gas, pred[m] == focus_gas
            out["focus_recall"] = (said_gas & is_gas).sum() / is_gas.sum() if is_gas.any() else np.nan
            out["focus_precision"] = (said_gas & is_gas).sum() / said_gas.sum() if said_gas.any() else np.nan
        return out

    scored = ~labelled
    rows = []
    for b in sorted(set(b_te)):
        m = scored & (b_te == b)
        rows.append({"test_batch": str(b), "n_test": int(m.sum()), **scores(m)})
    if len(test_batches) > 1:
        rows.append({"test_batch": "pooled", "n_test": int(scored.sum()), **scores(scored)})
    return pd.DataFrame(rows)
