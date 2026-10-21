"""Step 12 runner: repeat the Step 10 models on protocol A with 10 different random seeds.

The seed changes only the algorithms' own randomness (tree building, bootstrap samples, ...).
The train/test split is always the time-aware protocol A. Models are the tuned models chosen
by the Step 8 validation rule, plus the stack and soft vote of Step 9.
Finished runs are cached in the CSV, so the script can be stopped and restarted.

Run from the project root:  python -u scripts/run_step12_repeats.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(ROOT)))

import pandas as pd

from src import data_loader as dl
from src import evaluation as ev
from src import stacking, tuning
from src.repeats import set_all_seeds

OUT = ROOT / "results" / "step12_repeated_runs.csv"
SEEDS = list(range(10))
FAMILIES = ["kNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes", "AdaBoost", "Gradient Boosting"]


def label(family, reduction):
    return family if reduction == "None" else f"{family} + {reduction}"


def main():
    t_start = time.perf_counter()
    df = dl.remove_toluene(dl.load_all())
    best = pd.read_csv(ROOT / "results" / "step8_best_params.csv", keep_default_na=False)
    split = ev.get_splits("A")[0]

    singles = {}
    for family in FAMILIES:
        row = best[best["model"] == family].sort_values("mean_macro_f1", ascending=False, kind="stable").iloc[0]
        singles[label(family, row["reduction"])] = (family, row["reduction"], json.loads(row["params"]))
    stack_configs = stacking.choose_base_configs(best)

    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    finished = set(zip(done["model"], done["seed"])) if len(done) else set()

    for seed in SEEDS:
        t_seed = time.perf_counter()
        for name, (family, reduction, params) in singles.items():
            if (name, seed) in finished:
                continue
            make = lambda f=family, r=reduction, p=params: set_all_seeds(tuning.make_tuned(f, r, p), seed)
            metrics, _ = ev.evaluate_split(make, df, split)
            metrics.insert(0, "model", name)
            metrics["seed"] = seed
            done = pd.concat([done, metrics], ignore_index=True)
        if ("Stack (LR)", seed) not in finished:
            metrics = stacking.evaluate_stack_split(df, split, stack_configs, seed=seed)
            metrics = metrics[metrics["model"].isin(["Stack (LR)", "Soft vote"])].copy()
            metrics["seed"] = seed
            done = pd.concat([done, metrics], ignore_index=True)
        done.to_csv(OUT, index=False)
        pooled = done[(done["seed"] == seed) & (done["test_batch"] == "pooled")].set_index("model")["accuracy"]
        print(f"seed {seed}: {time.perf_counter() - t_seed:5.0f}s  " +
              "  ".join(f"{k}={v:.3f}" for k, v in pooled.items()), flush=True)

    print(f"All done in {(time.perf_counter() - t_start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
