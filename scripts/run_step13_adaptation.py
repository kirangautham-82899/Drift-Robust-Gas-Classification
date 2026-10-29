"""Step 13 runner (bonus B2/B3): can drift adaptation that uses the new batches beat the strict SVM?

Every method is scored on the two forward validation folds of Step 8 (train 1 -> validate 2, train 1-2 -> validate 3)
and on protocol A (train 1-3, test 4-10). A method should only be preferred if it also wins on validation.
Tier 3 methods are repeated with 10 seeds (which recordings get labelled), the MLP with 5 seeds.
A second part tries smarter ways to use a fixed budget of 25 labels per new batch (step13_label_strategies.csv), and
a third tries to fix the hardest gas, Acetaldehyde, recording its recall and precision (step13_acetaldehyde.csv).
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src import adaptation as ad
from src import data_loader as dl
from src.tuning import FOLDS

OUT = ROOT / "results" / "step13_adaptation_metrics.csv"
OUT_LABELS = ROOT / "results" / "step13_label_strategies.csv"
OUT_ACE = ROOT / "results" / "step13_acetaldehyde.csv"
ACETALDEHYDE = 4
LABEL_BUDGET = 25   # labels per new batch: the same number as 5 per gas
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
        n_rows = len(done)
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
            if len(done) > n_rows:
                done.to_csv(OUT, index=False)
        mine = done[(done["method"] == name) & (done["split"] == "A") & (done["test_batch"] == "pooled")]
        print(f"{name:45s} {time.perf_counter() - t_run:5.0f}s  protocol A pooled accuracy "
              f"{mine['accuracy'].astype(float).mean():.3f}", flush=True)

    run_label_strategies(df, svm_params)
    run_acetaldehyde(df, svm_params)
    print(f"All done in {(time.perf_counter() - t_start) / 60:.1f} minutes", flush=True)


def run_label_strategies(df, svm_params):
    """Same budget of 25 labels per new batch (chosen without knowing the gas), used in different ways."""
    svm = lambda C=svm_params["C"], g=svm_params["gamma"]: SVC(C=C, gamma=g, probability=True, random_state=42)
    vote = lambda: VotingClassifier([("svm", svm()), ("rf", RandomForestClassifier(300, random_state=42, n_jobs=4)),
                                     ("knn", KNeighborsClassifier(5)), ("lr", LogisticRegression(C=1, max_iter=3000))],
                                    voting="soft")
    random5, one = list(range(5)), [0]   # 'uncertain' is deterministic, everything with random picks gets 5 seeds
    runs = [("random labels, SVM", svm, {"strategy": "random"}, random5),
            ("most uncertain labels, SVM", svm, {"strategy": "uncertain"}, one),
            ("half uncertain + half random, SVM", svm, {"strategy": "mixed"}, random5),
            ("most uncertain labels, SVM + self-training", svm, {"strategy": "uncertain", "pseudo": 0.5}, one),
            ("most uncertain labels, soft vote", vote, {"strategy": "uncertain"}, one),
            ("random labels, SVM + self-training", svm, {"strategy": "random", "pseudo": 0.5}, random5),
            ("random labels, soft vote", vote, {"strategy": "random"}, random5),
            ("random labels, SVM, labels weighted x5", svm, {"strategy": "random", "label_weight": 5.0}, random5),
            ("random labels, SVM, batches 1-3 weighted x0.3", svm, {"strategy": "random", "old_weight": 0.3}, random5)]
    runs += [(f"random labels, SVM C={C} gamma={g}", lambda C=C, g=g: svm(C, g), {"strategy": "random"}, random5)
             for C, g in [(10, 0.01), (100, 0.001), (1000, 0.01), (1000, 0.001)]]

    done = pd.read_csv(OUT_LABELS, keep_default_na=False) if OUT_LABELS.exists() else pd.DataFrame()
    finished = set(zip(done["method"], done["seed"], done["split"])) if len(done) else set()
    for name, make, options, seeds in runs:
        t_run = time.perf_counter()
        n_rows = len(done)
        for seed in seeds:
            for split, tr, te in SPLITS:
                if (name, seed, split) in finished:
                    continue
                m = ad.evaluate(None, make, df, tr, te, budget=LABEL_BUDGET, seed=seed, **options)
                m.insert(0, "split", split)
                m.insert(0, "seed", seed)
                m.insert(0, "method", name)
                done = pd.concat([done, m], ignore_index=True)
            if len(done) > n_rows:
                done.to_csv(OUT_LABELS, index=False)
        mine = done[(done["method"] == name) & (done["split"] == "A") & (done["test_batch"] == "pooled")]
        print(f"{name:45s} {time.perf_counter() - t_run:5.0f}s  protocol A pooled accuracy "
              f"{mine['accuracy'].astype(float).mean():.3f}", flush=True)


def run_acetaldehyde(df, svm_params):
    """Fixes aimed at Acetaldehyde, the gas most often mistaken (for Ethanol). Every row records its recall and
    precision as well as the usual metrics."""
    svm = lambda **kw: SVC(**svm_params, random_state=42, **kw)
    ace_weighted = {g: (3 if g == ACETALDEHYDE else 1) for g in range(1, 6)}
    per_gas = lambda n, n_ace: {g: (n_ace if g == ACETALDEHYDE else n) for g in range(1, 6)}
    random5 = list(range(5))
    # (method, tier, prediction function, model factory, labels per gas, seeds)
    runs = [("SVM (strict)", 1, ad.predict_strict, svm, 0, [42]),
            ("SVM, balanced class weights", 1, ad.predict_strict, lambda: svm(class_weight="balanced"), 0, [42]),
            ("SVM, Acetaldehyde weighted x3", 1, ad.predict_strict, lambda: svm(class_weight=ace_weighted), 0, [42]),
            ("SVM + per-batch standardisation", 2, ad.predict_per_batch_std, svm, 0, [42]),
            ("SVM + per-batch std + Ethanol/Acetaldehyde specialist", 2,
             lambda make, *a: ad.predict_pair_specialist(make, *a, pair=(1, ACETALDEHYDE),
                                                         make_specialist=lambda: SVC(C=10, gamma="scale",
                                                                                     class_weight="balanced",
                                                                                     random_state=42)), svm, 0, [42]),
            ("SVM + per-batch std + 5 labels per gas", 3, None, svm, 5, random5),
            ("SVM + per-batch std + 10 labels per gas", 3, None, svm, 10, random5),
            ("SVM + per-batch std + 5 per gas, 15 Acetaldehyde", 3, None, svm, per_gas(5, 15), random5),
            ("SVM + per-batch std + 10 per gas, 20 Acetaldehyde", 3, None, svm, per_gas(10, 20), random5)]

    done = pd.read_csv(OUT_ACE, keep_default_na=False) if OUT_ACE.exists() else pd.DataFrame()
    finished = set(zip(done["method"], done["seed"], done["split"])) if len(done) else set()
    for name, tier, method, make, n_per_gas, seeds in runs:
        t_run = time.perf_counter()
        n_rows = len(done)
        for seed in seeds:
            for split, tr, te in SPLITS:
                if (name, seed, split) in finished:
                    continue
                m = ad.evaluate(method, make, df, tr, te, n_per_gas=n_per_gas, seed=seed, focus_gas=ACETALDEHYDE)
                m.insert(0, "split", split)
                m.insert(0, "seed", seed)
                m.insert(0, "tier", tier)
                m.insert(0, "method", name)
                done = pd.concat([done, m], ignore_index=True)
            if len(done) > n_rows:
                done.to_csv(OUT_ACE, index=False)
        mine = done[(done["method"] == name) & (done["split"] == "A") & (done["test_batch"] == "pooled")]
        print(f"{name:55s} {time.perf_counter() - t_run:5.0f}s  Acetaldehyde recall "
              f"{mine['focus_recall'].astype(float).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
