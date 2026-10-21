"""Step 11 runner: the same models under a random 80/20 split and a size-matched random split.

The time-aware numbers (protocol A) are taken from Steps 5 and 7, which used exactly the same
models and settings (library defaults, scaler, no PCA/LDA). Finished runs are cached in the
CSV, so the script can be stopped and restarted.

Run from the project root:  python -u scripts/run_step11_leakage.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import data_loader as dl
from src import leakage, models

OUT = ROOT / "results" / "step11_random_vs_time_aware.csv"
SEEDS = [0, 1, 2]
N_TRAIN_A = 3196        # rows in batches 1-3, the training set of protocol A


def main():
    t_start = time.perf_counter()
    df = dl.remove_toluene(dl.load_all())
    assert int(df["batch"].isin([1, 2, 3]).sum()) == N_TRAIN_A

    factories = {**models.get_baselines("None"), **models.get_boosting("None")}
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    finished = set(zip(done["model"], done["scheme"], done["seed"])) if len(done) else set()

    for seed in SEEDS:
        splits = {"random 80/20": leakage.random_split(df, seed),
                  "random, same sizes as protocol A": leakage.size_matched_split(df, seed, N_TRAIN_A)}
        for model, make in factories.items():
            for scheme, (tr, te) in splits.items():
                if (model, scheme, seed) in finished:
                    continue
                t0 = time.perf_counter()
                row = {"model": model, "scheme": scheme, "seed": seed,
                       **leakage.evaluate_rows(make, df, tr, te)}
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(OUT, index=False)
                print(f"seed {seed} | {model:18s} | {scheme:34s} accuracy {row['accuracy']:.3f}  "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"All done in {(time.perf_counter() - t_start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
