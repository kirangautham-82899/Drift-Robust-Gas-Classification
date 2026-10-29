"""Step 13 runner (bonus B2/B3): can drift adaptation that uses the new batches beat the strict SVM?

Every method is scored on the two forward validation folds of Step 8 (train 1 -> validate 2, train 1-2 -> validate 3)
and on protocol A (train 1-3, test 4-10). A method should only be preferred if it also wins on validation.
Tier 3 methods are repeated with 10 seeds (which recordings get labelled), the MLP with 5 seeds.
Finished runs are cached in the CSV, so the script can be stopped and restarted.

Run from the project root:  python -u scripts/run_step13_adaptation.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(ROOT)))

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src import adaptation as ad
from src import data_loader as dl
from src.tuning import FOLDS

OUT = ROOT / "results" / "step13_adaptation_metrics.csv"
SPLITS = [(f"val {'-'.join(map(str, tr))} -> {va}", tr, (va,)) for tr, va in FOLDS] + [("A", (1, 2, 3), tuple(range(4, 11)))]


def main():
    t_start = time.perf_counter()
    df = dl.remove_toluene(dl.load_all())
    best = pd.read_csv(ROOT / "results" / "step8_best_params.csv", keep_default_na=False)
    svm_params = json.loads(best[(best["model"] == "SVM") & (best["reduction"] == "None")].iloc[0]["params"])
    svm = lambda: SVC(**svm_params, random_state=42)
    svm_proba = lambda: SVC(**svm_params, probability=True, random_state=42)

    # (method name, tier, prediction function, model factory, labels per gas, seeds)
    runs = [("SVM (strict)", 1, ad.predict_strict, svm, 0, [42]),
            ("MLP (strict)", 1, ad.predict_strict, None, 0, list(range(5))),
            ("SVM + CORAL", 2, ad.predict_coral, svm, 0, [42]),
            ("SVM + per-batch standardisation", 2, ad.predict_per_batch_std, svm, 0, [42]),
            ("SVM + per-batch std + self-training", 2, ad.predict_self_training, svm_proba, 0, [42])]
    runs += [(f"SVM + per-batch std + {n} labels per gas", 3, None, svm, n, list(range(10))) for n in (3, 5, 10)]

    done = pd.read_csv(OUT, keep_default_na=False) if OUT.exists() else pd.DataFrame()
    finished = set(zip(done["method"], done["seed"], done["split"])) if len(done) else set()

    for name, tier, method, make, n_per_gas, seeds in runs:
        t_run = time.perf_counter()
        for seed in seeds:
            if make is None:   # the MLP's own randomness is what the seed changes
                make_seeded = lambda s=seed: MLPClassifier((128, 128), max_iter=500, early_stopping=True, random_state=s)
            else:
                make_seeded = make
            for split, tr, te in SPLITS:
                if (name, seed, split) in finished:
                    continue
                m = ad.evaluate(method, make_seeded, df, tr, te, n_per_gas=n_per_gas, seed=seed)
                m.insert(0, "split", split)
                m.insert(0, "seed", seed)
                m.insert(0, "tier", tier)
                m.insert(0, "method", name)
                done = pd.concat([done, m], ignore_index=True)
            done.to_csv(OUT, index=False)
        mine = done[(done["method"] == name) & (done["split"] == "A") & (done["test_batch"] == "pooled")]
        print(f"{name:45s} {time.perf_counter() - t_run:5.0f}s  protocol A pooled accuracy "
              f"{mine['accuracy'].astype(float).mean():.3f}", flush=True)

    print(f"All done in {(time.perf_counter() - t_start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
