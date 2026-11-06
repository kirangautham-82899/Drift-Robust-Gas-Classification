"""Step 14 runner: robustness checks of the main findings.

1. Batch-level bootstrap of the Step 10 predictions (honest intervals; see src/robustness.py).
2. Per-batch standardisation from only n recordings of each new batch (n = 10 ... 200), 10 seeds.
3. MLP and CORAL tuned on the same two validation folds as the SVM (Step 8), so the Step 13 validation check is fair.
4. Other training periods: the 7 tuned model families and the per-batch-standardised SVM trained on different early
   batches. Hyperparameters stay those of Step 8. Batch 3 helped choose them, so it is never a test batch.

Finished runs are cached in the CSVs, so the script can be stopped and restarted.
Run from the project root:  python -u scripts/run_step14_robustness.py
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src import adaptation as ad
from src import data_loader as dl
from src import robustness as rb
from src import tuning
from src.tuning import FOLDS

RES = ROOT / "results"
VAL_SPLITS = [(f"val {'-'.join(map(str, tr))} -> {va}", tr, (va,)) for tr, va in FOLDS]
SPLIT_A = ("A", (1, 2, 3), tuple(range(4, 11)))
SAMPLE_SIZES = [10, 25, 50, 100, 200]
PERIODS = [(1, 2), (1, 2, 3), (2, 3), (1, 2, 3, 4), (2, 3, 4), (1, 2, 3, 4, 5)]
FAMILIES = ["kNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes", "AdaBoost", "Gradient Boosting"]


def cached_runs(path, key_cols, jobs, run):
    """jobs: list of dicts holding key_cols; run(job) -> DataFrame of metric rows. Saves after each job."""
    done = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    finished = set(map(tuple, done[key_cols].astype(str).to_numpy())) if len(done) else set()
    for job in jobs:
        if tuple(str(job[k]) for k in key_cols) in finished:
            continue
        m = run(job)
        for i, k in enumerate(key_cols):
            m.insert(i, k, job[k])
        done = pd.concat([done, m], ignore_index=True)
        done.to_csv(path, index=False)
    return done


def pooled_acc(done, **where):
    sel = done
    for k, v in where.items():
        sel = sel[sel[k].astype(str) == str(v)]
    return sel[sel["test_batch"] == "pooled"]["accuracy"].astype(float)


# ------------------------------------------------------------------------------------------- 1. batch bootstrap
def part_batch_bootstrap():
    pred = pd.read_csv(RES / "dashboard_predictions.csv")
    models = [c for c in pred.columns if c not in ("batch", "label")]
    correct = {m: (pred[m] == pred["label"]).to_numpy() for m in models}
    boot = rb.batch_bootstrap(correct, pred["batch"].to_numpy(), n_boot=10000, rng=np.random.default_rng(42))
    row_ci = pd.read_csv(RES / "step10_per_batch_ci.csv")
    row_ci = row_ci[row_ci["group"] == "4-10 (all)"].set_index("model")
    per_batch = {m: pred.assign(ok=correct[m]).groupby("batch")["ok"].mean() for m in models}
    rows = []
    for m in models:
        lo, hi = np.percentile(boot[m]["pooled"], [2.5, 97.5]) * 100
        mlo, mhi = np.percentile(boot[m]["mean"], [2.5, 97.5]) * 100
        dlo, dhi = np.percentile(boot[m]["pooled"] - boot["SVM"]["pooled"], [2.5, 97.5]) * 100
        rows.append({"model": m, "pooled accuracy %": correct[m].mean() * 100,
                     "row CI low": row_ci.loc[m, "acc_low"] * 100, "row CI high": row_ci.loc[m, "acc_high"] * 100,
                     "batch CI low": lo, "batch CI high": hi,
                     "mean over batches %": per_batch[m].mean() * 100, "mean CI low": mlo, "mean CI high": mhi,
                     "vs SVM (points)": (correct[m].mean() - correct["SVM"].mean()) * 100,
                     "vs SVM CI low": dlo, "vs SVM CI high": dhi,
                     "batches won vs SVM": int((per_batch[m] > per_batch["SVM"] + 1e-12).sum()),
                     "batches lost vs SVM": int((per_batch[m] < per_batch["SVM"] - 1e-12).sum())})
    out = pd.DataFrame(rows).sort_values("pooled accuracy %", ascending=False)
    out.loc[out["model"] == "SVM", ["vs SVM CI low", "vs SVM CI high"]] = np.nan
    out.round(2).to_csv(RES / "step14_batch_bootstrap.csv", index=False)
    print("batch bootstrap done", flush=True)
    return out


# ------------------------------------------------------------------------------------------- 2. sample-based std
def part_subsample_std(df, svm):
    jobs = [{"n": n, "seed": s, "split": sp[0]} for n in SAMPLE_SIZES for s in range(10) for sp in VAL_SPLITS + [SPLIT_A]]
    splits = {sp[0]: sp for sp in VAL_SPLITS + [SPLIT_A]}

    def run(job):
        _, tr, te = splits[job["split"]]
        method = lambda make, *a, n=job["n"], s=job["seed"]: rb.predict_subsample_std(make, *a, n=n, seed=s)
        return ad.evaluate(method, svm, df, tr, te)

    done = cached_runs(RES / "step14_subsample_std.csv", ["n", "seed", "split"], jobs, run)
    ref = pd.read_csv(RES / "step13_summary.csv").set_index("method")
    rows = []
    for n in SAMPLE_SIZES:
        sel = done[done["n"].astype(int) == n]
        val = np.mean([pooled_or_single(sel, sp[0]).mean() for sp in VAL_SPLITS]) * 100
        a = sel[(sel["split"] == "A") & (sel["test_batch"] == "pooled")]["accuracy"].astype(float) * 100
        b10 = sel[(sel["split"] == "A") & (sel["test_batch"].astype(str) == "10")]["accuracy"].astype(float) * 100
        rows.append({"recordings used per new batch": n, "validation accuracy %": val, "test accuracy %": a.mean(),
                     "test std (points)": a.std(ddof=0), "test worst seed %": a.min(), "batch 10 accuracy %": b10.mean()})
    full = ref.loc["SVM + per-batch standardisation"]
    rows.append({"recordings used per new batch": "all", "validation accuracy %": float(full["validation accuracy %"]),
                 "test accuracy %": float(full["test accuracy %"]), "test std (points)": 0.0,
                 "test worst seed %": float(full["test accuracy %"]), "batch 10 accuracy %": float(full["batch 10 accuracy %"])})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "step14_subsample_std_summary.csv", index=False)
    print("sample-based standardisation done", flush=True)
    return out


def pooled_or_single(sel, split):
    """Accuracy of a validation split (one test batch, so no pooled row) for every seed."""
    return sel[sel["split"] == split]["accuracy"].astype(float)


# ------------------------------------------------------------------------------------------- 3. tuned MLP / CORAL
MLP_GRID = {"hidden_layer_sizes": [(64,), (128,), (128, 128), (256, 128)], "alpha": [1e-4, 1e-3, 1e-2]}
CORAL_GRID = {"eps": [0.01, 0.1, 1.0, 10.0], "C": [1, 10, 100], "gamma": ["scale", 0.01]}


def grid(g):
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def make_mlp(p, seed):
    return lambda: MLPClassifier(p["hidden_layer_sizes"], alpha=p["alpha"], max_iter=500, early_stopping=True,
                                 random_state=seed)


def coral_method(eps):
    def method(make_model, X_tr, y_tr, b_tr, X_te, b_te):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_tr)
        Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
        for b in np.unique(b_te):
            rows = b_te == b
            Z_te[rows] = ad.coral(Z_tr, Z_te[rows], eps=eps)
        return make_model().fit(Z_tr, y_tr).predict(Z_te)
    return method


def part_tuned_mlp_coral(df):
    configs = [("MLP", json.dumps({**p, "hidden_layer_sizes": list(p["hidden_layer_sizes"])})) for p in grid(MLP_GRID)]
    configs += [("SVM + CORAL", json.dumps(p)) for p in grid(CORAL_GRID)]

    def runner(family, params, seed):
        p = json.loads(params)
        if family == "MLP":
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
            return ad.predict_strict, make_mlp(p, seed)
        return coral_method(p["eps"]), (lambda: SVC(C=p["C"], gamma=p["gamma"], random_state=42))

    # grid search on the validation folds only (seed 0)
    jobs = [{"family": f, "params": p, "seed": 0, "split": sp[0]} for f, p in configs for sp in VAL_SPLITS]
    splits = {sp[0]: sp for sp in VAL_SPLITS + [SPLIT_A]}

    def run(job):
        _, tr, te = splits[job["split"]]
        method, make = runner(job["family"], job["params"], int(job["seed"]))
        return ad.evaluate(method, make, df, tr, te)

    search = cached_runs(RES / "step14_mlp_coral_grid.csv", ["family", "params", "seed", "split"], jobs, run)
    val = search.assign(accuracy=search["accuracy"].astype(float)).groupby(["family", "params"], sort=False)["accuracy"].mean()

    # the best configuration of each family (by validation only), then 5 seeds on validation and protocol A
    best = {f: val[f].idxmax() for f in ["MLP", "SVM + CORAL"]}
    seeds = {"MLP": list(range(5)), "SVM + CORAL": [0]}
    jobs = [{"family": f, "params": best[f], "seed": s, "split": sp[0]}
            for f in best for s in seeds[f] for sp in VAL_SPLITS + [SPLIT_A]]
    final = cached_runs(RES / "step14_mlp_coral_best.csv", ["family", "params", "seed", "split"], jobs, run)

    ref = pd.read_csv(RES / "step13_summary.csv").set_index("method")
    strict_val = float(ref.loc["SVM (strict)", "validation accuracy %"])
    rows = []
    for f, before in [("MLP", "MLP (strict)"), ("SVM + CORAL", "SVM + CORAL")]:
        sel = final[(final["family"] == f) & (final["params"] == best[f])]
        v = np.mean([pooled_or_single(sel, sp[0]).mean() for sp in VAL_SPLITS]) * 100
        a = sel[(sel["split"] == "A") & (sel["test_batch"] == "pooled")]["accuracy"].astype(float) * 100
        rows.append({"method": f"{f}, tuned on validation", "settings": best[f],
                     "validation accuracy % (untuned)": float(ref.loc[before, "validation accuracy %"]),
                     "validation accuracy %": v, "test accuracy % (untuned)": float(ref.loc[before, "test accuracy %"]),
                     "test accuracy %": a.mean(), "seeds": len(a),
                     "strict SVM validation %": strict_val, "wins on validation": bool(v > strict_val)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "step14_mlp_coral_summary.csv", index=False)
    print("tuned MLP / CORAL done", flush=True)
    return out


# ------------------------------------------------------------------------------------------- 4. training periods
def part_training_periods(df, svm):
    best = pd.read_csv(RES / "step8_best_params.csv", keep_default_na=False)
    singles = {}
    for fam in FAMILIES:
        row = best[best["model"] == fam].sort_values("mean_macro_f1", ascending=False, kind="stable").iloc[0]
        name = fam if row["reduction"] == "None" else f"{fam} + {row['reduction']}"
        singles[name] = (fam, row["reduction"], json.loads(row["params"]))

    def test_batches(period):
        return tuple(range(max(max(period) + 1, 4), 11))

    jobs = [{"period": "-".join(map(str, p)), "model": m} for p in PERIODS
            for m in list(singles) + ["SVM + per-batch standardisation"]]

    def run(job):
        period = tuple(int(x) for x in job["period"].split("-"))
        if job["model"] == "SVM + per-batch standardisation":
            return ad.evaluate(ad.predict_per_batch_std, svm, df, period, test_batches(period))
        fam, red, params = singles[job["model"]]
        # tuning.make_tuned already includes the scaler, so predict_strict must not add a second one
        return ad.evaluate(lambda make, Xtr, ytr, btr, Xte, bte: make().fit(Xtr, ytr).predict(Xte),
                           lambda: tuning.make_tuned(fam, red, params), df, period, test_batches(period))

    done = cached_runs(RES / "step14_training_periods.csv", ["period", "model"], jobs, run)
    rows = []
    for (period, model), g in done.groupby(["period", "model"], sort=False):
        per = g[g["test_batch"] != "pooled"].assign(b=lambda d: d["test_batch"].astype(int),
                                                    a=lambda d: d["accuracy"].astype(float) * 100)
        first3, far = per.nsmallest(3, "b")["a"].mean(), per[per["b"] >= 8]["a"].mean()
        rows.append({"training batches": period, "test batches": f"{per['b'].min()}-10", "model": model,
                     "pooled accuracy %": float(g[g["test_batch"] == "pooled"]["accuracy"].iloc[0]) * 100,
                     "mean over batches %": per["a"].mean(), "first 3 test batches %": first3,
                     "batches 8-10 %": far, "drop (points)": far - first3})
    out = pd.DataFrame(rows)
    out["rank (pooled)"] = out.groupby("training batches")["pooled accuracy %"].rank(ascending=False).astype(int)
    out.to_csv(RES / "step14_training_periods_summary.csv", index=False)
    print("training periods done", flush=True)
    return out


# ------------------------------------------------------------------------------------------- figure
def figure(bb, periods):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    d = bb.iloc[::-1]
    y = np.arange(len(d))
    ax = axes[0]
    ax.errorbar(d["pooled accuracy %"], y + 0.15, xerr=[d["pooled accuracy %"] - d["row CI low"],
                d["row CI high"] - d["pooled accuracy %"]], fmt="o", color="#8d99ae", capsize=3, label="rows resampled (Step 10)")
    ax.errorbar(d["pooled accuracy %"], y - 0.15, xerr=[d["pooled accuracy %"] - d["batch CI low"],
                d["batch CI high"] - d["pooled accuracy %"]], fmt="o", color="#e76f51", capsize=3, label="whole batches resampled")
    ax.set_yticks(y); ax.set_yticklabels(d["model"]); ax.set_xlabel("pooled accuracy on batches 4-10 (%), 95% interval")
    ax.set_title("Honest intervals are much wider"); ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    piv = periods.pivot(index="model", columns="training batches", values="pooled accuracy %")
    piv = piv[["-".join(map(str, p)) for p in PERIODS]].sort_values("1-2-3")
    for i, col in enumerate(piv.columns):
        ax.scatter(piv[col], np.arange(len(piv)) + (i - 2.5) * 0.1, s=22, label=f"train {col}")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index); ax.set_xlabel("pooled accuracy on later batches (%)")
    ax.set_title("Different training periods"); ax.legend(fontsize=7, ncol=2); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / "step14_robustness.png", dpi=150)


def main():
    t0 = time.perf_counter()
    df = dl.remove_toluene(dl.load_all())
    best = pd.read_csv(RES / "step8_best_params.csv", keep_default_na=False)
    p = json.loads(best[(best["model"] == "SVM") & (best["reduction"] == "None")].iloc[0]["params"])
    svm = lambda: SVC(**p, random_state=42)

    bb = part_batch_bootstrap()
    part_subsample_std(df, svm)
    part_tuned_mlp_coral(df)
    periods = part_training_periods(df, svm)
    figure(bb, periods)
    print(f"All done in {(time.perf_counter() - t0) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
