"""Every number table in README.md must match the saved result files (Step 13)."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())   # whitespace-normalised


def read(name):
    # keep_default_na=False: otherwise pandas turns the text "None" (no reduction) into a missing value
    return pd.read_csv(ROOT / "results" / name, keep_default_na=False)


def row(text):
    return " ".join(text.split())


def test_headline_table_matches_step12():
    s = read("step12_summary.csv").rename(columns={"Unnamed: 0": "model"}).set_index("model")
    assert len(s) == 9
    for m in s.index:
        expected = (f"| {m} | {s.loc[m, 'mean accuracy %']:.1f}% ± {s.loc[m, 'std (points)']:.2f} "
                    f"| {s.loc[m, 'mean macro-F1 %']:.1f}% |")
        assert row(expected) in README, expected


def test_drift_table_matches_step10():
    nf = read("step10_near_far.csv").set_index("model")
    for m in ["SVM", "Soft vote", "Stack (LR)", "Random Forest + PCA", "kNN + PCA"]:
        x = nf.loc[m]
        expected = f"| {m} | {x['near (4-6) %']:.1f}% | {x['far (8-10) %']:.1f}% | {x['change (points)']:.1f} |"
        assert row(expected) in README, expected
    assert nf["far (8-10) %"].max() <= 67.0          # "No model exceeds 67% on the far batches"


def test_random_vs_time_aware_table_matches_step11():
    t = read("step11_accuracy_table.csv").set_index("model")
    assert len(t) == 7
    for m in t.index:
        x = t.loc[m]
        expected = f"| {m} | {x['time-aware (A)']:.1f}% | {x['random, same sizes']:.1f}% | {x['random 80/20']:.1f}% |"
        assert row(expected) in README, expected
    nn = read("step11_nearest_neighbour.csv").set_index("exam")
    assert nn.loc["random 80/20"].iloc[1] > 98 and nn.loc["time-aware (protocol A)"].iloc[1] == 0   # "98% ... same batch"


def test_protocol_c_table_matches_step9():
    c = read("step9_stacking_metrics.csv")
    c = c[(c["protocol"] == "C") & (c["test_batch"] != "pooled")]
    mean = c.groupby("model")["accuracy"].mean() * 100
    for m in ["SVM", "Stack (LR)", "Soft vote"]:
        assert row(f"| {m} | {mean[m]:.1f}% |") in README


def test_dataset_counts_match_step1():
    counts = read("step1_counts.csv").set_index("batch")
    assert int(counts.loc["Total", "after"]) == 12077 and "12,077" in README
    per_batch = [int(counts.loc[str(b), "after"]) for b in range(1, 11)]
    text = ", ".join(f"{n:,}" for n in per_batch[:-1]) + f" and {per_batch[-1]:,} samples"
    assert text in README, text


def test_tuning_and_validation_claims_match_step8():
    cmp = read("step8_tuned_vs_untuned.csv")
    change = cmp["accuracy change"].astype(float) * 100
    better, same, worse = int((change > 0.5).sum()), int((change.abs() <= 0.5).sum()), int((change < -0.5).sum())
    assert better + same + worse == len(change) == 21
    assert f"{better} of 21 model/reduction combinations improved, {same} stayed about the same and {worse} got worse" in README

    best = read("step8_best_params.csv").set_index(["model", "reduction"])["mean_macro_f1"].astype(float)
    tuned = read("step8_tuned_protocolA.csv")
    test_f1 = tuned[tuned["test_batch"] == "pooled"].set_index(["model", "reduction"])["macro_f1"].astype(float)
    both = pd.concat([best, test_f1], axis=1, keys=["val", "test"]).dropna()
    assert len(both) == 21
    assert f"correlation {both['val'].corr(both['test']):.2f}" in README
    assert f"{(both['val'] - both['test']).mean() * 100:.1f} points too optimistic" in README


def test_adaptation_table_matches_step13():
    s = read("step13_summary.csv").set_index("method")
    assert len(s) == 8
    for m in s.index:
        x = s.loc[m]
        expected = (f"| {m} | {x['tier']} | {float(x['validation accuracy %']):.1f}% | {float(x['test accuracy %']):.1f}% "
                    f"| {float(x['batch 10 accuracy %']):.1f}% |")
        assert row(expected) in README, expected
    best = s.loc["SVM + per-batch standardisation"]
    assert str(best["wins on validation"]) == "True" and f"+{float(best['test change (points)']):.1f}" in README
    assert str(s.loc["MLP (strict)", "wins on validation"]) == str(s.loc["SVM + CORAL", "wins on validation"]) == "False"


def test_label_strategy_claims_match_step13():
    s = read("step13_label_strategies_summary.csv").set_index("method")
    assert len(s) == 13
    ref = s.loc["random labels, SVM"]
    assert (f"{float(ref['validation accuracy %']):.1f}% on validation and {float(ref['test accuracy %']):.1f}% on test"
            in README)
    others = s.drop(index="random labels, SVM")
    assert (others["validation change (points)"].astype(float) <= 0.05).all()       # "never beat that on validation"
    assert f"reach {others['test accuracy %'].astype(float).max():.1f}% on test" in README
    assert (s["fair improvement"].astype(str) == "False").all()


def test_readme_has_required_sections_and_no_viva_sheet():
    for heading in ["## Main results", "## Conclusions", "## Limitations", "## How to reproduce", "## Where each number comes from"]:
        assert heading in README, heading
    assert "viva" not in README.lower()
