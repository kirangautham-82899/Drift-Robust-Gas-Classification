"""Step 8 runner: tune every model/reduction on batches 1-3 only, then evaluate the chosen
configurations on protocol A. Safe to stop and restart: finished work is cached in results/.

Run from the project root:  python -u scripts/run_step8_tuning.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src import data_loader as dl
from src import evaluation as ev
from src import tuning

RESULTS = ROOT / "results"
CV_FILE = RESULTS / "step8_tuning_cv.csv"
TUNED_FILE = RESULTS / "step8_tuned_protocolA.csv"
REDUCTIONS = ["None", "PCA", "LDA"]


def read(path):
    # keep_default_na=False: otherwise pandas turns the text "None" (no reduction) into a missing value
    return pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()


def main():
    df = dl.remove_toluene(dl.load_all())
    train_df = df[df["batch"].isin(tuning.TRAIN_BATCHES)]   # tuning never sees batches 4-10
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)

    # ---- phase 1: cross-validation on batches 1-3 ----
    cv = read(CV_FILE)
    done = set(zip(cv["model"], cv["reduction"])) if len(cv) else set()
    t_start = time.perf_counter()
    for red in REDUCTIONS:
        for model in tuning.ALL_MODELS:
            if (model, red) in done:
                continue
            t0 = time.perf_counter()
            part = tuning.tune(model, red, train_df)
            cv = pd.concat([cv, part], ignore_index=True)
            cv.to_csv(CV_FILE, index=False)
            print(f"[CV ] {model:18s} {red:5s} {len(part):3d} configs  {time.perf_counter() - t0:6.1f}s "
                  f"best CV macro-F1 {part['mean_macro_f1'].max():.3f}", flush=True)

    # ---- phase 2: refit the chosen configuration on batches 1-3, test on protocol A ----
    best = tuning.select_best(cv)
    tuned = read(TUNED_FILE)
    done = set(zip(tuned["model"], tuned["reduction"])) if len(tuned) else set()
    split = ev.get_splits("A")[0]
    for _, row in best.iterrows():
        key = (row["model"], row["reduction"])
        if key in done:
            continue
        t0 = time.perf_counter()
        params = json.loads(row["params"])
        make = lambda: tuning.make_tuned(row["model"], row["reduction"], params)
        metrics, _, fitted = ev.evaluate_split(make, df, split, return_model=True)
        metrics.insert(0, "model", row["model"])
        metrics["reduction"] = row["reduction"]
        metrics["params"] = row["params"]
        tuned = pd.concat([tuned, metrics], ignore_index=True)
        tuned.to_csv(TUNED_FILE, index=False)
        name = f"tuned_{row['reduction'].lower()}_{row['model'].replace(' ', '_').lower()}_protocolA.joblib"
        joblib.dump(fitted, RESULTS / "models" / name, compress=3)
        pooled = metrics[metrics["test_batch"] == "pooled"].iloc[0]
        print(f"[TEST] {row['model']:18s} {row['reduction']:5s} protocol A pooled accuracy "
              f"{pooled['accuracy']:.3f}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    print(f"All done in {(time.perf_counter() - t_start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
