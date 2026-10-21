"""Step 9 runner: fit the stacking ensemble and its comparisons, save everything to results/.

Protocol A (train 1-3, test 4-10) plus protocol C for test batches 4-10 (retrain on all
earlier batches). Safe to stop and restart: finished splits are cached in the CSV.

Run from the project root:  python -u scripts/run_step9_stacking.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from src import data_loader as dl
from src import evaluation as ev
from src import stacking

RESULTS = ROOT / "results"
METRICS_FILE = RESULTS / "step9_stacking_metrics.csv"


def main():
    t_start = time.perf_counter()
    df = dl.remove_toluene(dl.load_all())
    best = pd.read_csv(RESULTS / "step8_best_params.csv", keep_default_na=False)
    configs = stacking.choose_base_configs(best)
    print("Base models:", {k: v["reduction"] for k, v in configs.items()}, flush=True)

    done = pd.read_csv(METRICS_FILE) if METRICS_FILE.exists() else pd.DataFrame()
    finished = set(done["split"] + "|" + done["protocol"]) if len(done) else set()

    splits = [ev.get_splits("A")[0]] + [s for s in ev.get_splits("C") if min(s.test_batches) >= 4]
    for split in splits:
        key = f"{split.name}|{split.protocol}"
        if key in finished:
            print(f"[skip] {key} already done", flush=True)
            continue
        t0 = time.perf_counter()
        metrics, stack = stacking.evaluate_stack_split(df, split, configs, return_model=True)
        done = pd.concat([done, metrics], ignore_index=True)
        done.to_csv(METRICS_FILE, index=False)
        print(f"[{split.protocol}] {split.name:24s} {time.perf_counter() - t0:6.0f}s", flush=True)

        if split.protocol == "A":
            joblib.dump(stack, RESULTS / "models" / "stack_protocolA.joblib", compress=3)
            # what the meta-model trusts: average absolute weight per base model
            coef = np.abs(stack.final_estimator_.coef_)
            per_base = coef.reshape(coef.shape[0], len(configs), -1).mean(axis=(0, 2))
            pd.Series(per_base, index=list(configs), name="mean_abs_weight").to_csv(
                RESULTS / "step9_meta_weights.csv")
            # out-of-fold accuracy of each base model on the training batches
            train = df[df["batch"].isin(split.train_batches)]
            Xtr, ytr = train[dl.FEATURE_COLS].to_numpy(), train["label"].to_numpy()
            folds = stacking.leave_one_batch_out(train["batch"].to_numpy())
            oof = {}
            for name, pipe in stacking.build_base_estimators(configs):
                pred = cross_val_predict(pipe, Xtr, ytr, cv=folds, method="predict")
                oof[name] = {f"batch {b} held out": float((pred[te] == ytr[te]).mean())
                             for b, (_, te) in zip(sorted(train["batch"].unique()), folds)}
            pd.DataFrame(oof).T.to_csv(RESULTS / "step9_oof_accuracy.csv")
            print("[A] saved stack model, meta-model weights and out-of-fold accuracy", flush=True)

    print(f"All done in {(time.perf_counter() - t_start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
