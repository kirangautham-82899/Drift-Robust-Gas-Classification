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


def test_acetaldehyde_table_matches_step13():
    s = read("step13_acetaldehyde_summary.csv").set_index("method")
    shown = ["SVM (strict)", "SVM, Acetaldehyde weighted x3", "SVM + per-batch standardisation",
             "SVM + per-batch std + Ethanol/Acetaldehyde specialist", "SVM + per-batch std + 10 labels per gas",
             "SVM + per-batch std + 10 per gas, 20 Acetaldehyde"]
    for m in shown:
        x = {c: float(v) for c, v in s.loc[m].items() if c.endswith("%")}
        expected = (f"| {m} | {x['Acetaldehyde recall %']:.1f}% | {x['Acetaldehyde precision %']:.1f}% "
                    f"| {x['Acetaldehyde recall, batch 10 %']:.1f}% | {x['test accuracy %']:.1f}% |")
        assert row(expected) in README, expected
    strict, pbs = s.loc["SVM (strict)"], s.loc["SVM + per-batch standardisation"]
    assert float(pbs["Acetaldehyde recall %"]) < float(strict["Acetaldehyde recall %"])      # "makes it worse"
    assert float(pbs["test accuracy %"]) > float(strict["test accuracy %"])                  # "overall accuracy rises"
    counts = read("step1_gas_counts_per_batch.csv").set_index("batch").loc[[1, 2, 3]].sum()
    share = counts / counts.sum() * 100
    assert share.idxmin() == "Acetaldehyde" and f"smallest class ({share.min():.1f}%)" in README


def test_per_batch_average_claims_match_step13_and_step9():
    """Per-batch averages next to the pooled numbers: per-batch std vs strict SVM, 25 random labels vs protocol C."""
    m = read("step13_adaptation_metrics.csv")
    a = m[(m["split"] == "A") & (m["test_batch"] != "pooled")]
    mean = a.groupby("method")["accuracy"].apply(lambda s: s.astype(float).mean() * 100)
    pbs, strict = mean["SVM + per-batch standardisation"], mean["SVM (strict)"]
    assert pbs < strict and f"it scores {pbs:.1f}% against {strict:.1f}% for the strict SVM" in README

    lab = read("step13_label_strategies.csv")
    r = lab[(lab["method"] == "random labels, SVM") & (lab["split"] == "A") & (lab["test_batch"] != "pooled")]
    r_mean = r.assign(accuracy=r["accuracy"].astype(float)).groupby("seed")["accuracy"].mean().mean() * 100

    c = read("step9_stacking_metrics.csv")
    c = c[(c["protocol"] == "C") & (c["model"] == "SVM") & c["test_batch"].isin([str(b) for b in range(4, 11)])]
    acc, n = c["accuracy"].astype(float), c["n_test"].astype(float)
    c_pooled, c_mean = (acc * n).sum() / n.sum() * 100, acc.mean() * 100
    s = read("step13_label_strategies_summary.csv").set_index("method")
    r_pooled = float(s.loc["random labels, SVM", "test accuracy %"])
    assert (f"25 random labels per batch reach {r_pooled:.1f}%, against {c_pooled:.1f}% for protocol C on the same "
            f"measure ({r_mean:.1f}% vs {c_mean:.1f}% averaged over batches)") in README
    assert f"92.1% on test ({r_mean:.1f}% averaged over batches)" in README

    counts = read("step1_gas_counts_per_batch.csv").set_index("batch")
    test_sizes = counts.loc[list(range(4, 11))].sum(axis=1)
    assert f"holds {test_sizes[10] / test_sizes.sum() * 100:.0f}% of the test recordings" in README
    share10 = counts.loc[10] / counts.loc[10].sum()
    assert (share10.round(4) == 0.2).all() and "has exactly 20% of each gas" in README


def test_svm_support_vector_claim_matches_saved_model():
    """'only 7 of its 213 support vectors sit at the limit where class weights act'."""
    import joblib
    import numpy as np
    svc = joblib.load(ROOT / "results" / "models" / "tuned_none_svm_protocolA.joblib").named_steps["model"]
    at_bound = int((np.abs(svc.dual_coef_) >= svc.C * (1 - 1e-6)).any(axis=0).sum())
    assert svc.C == 100 and f"only {at_bound} of its {svc.n_support_.sum()} support vectors" in README


def test_batch_bootstrap_table_matches_step14():
    s = read("step14_batch_bootstrap.csv").set_index("model")
    f = lambda v: f"{float(v):.1f}"
    signed = lambda v: f"{float(v):+.1f}"
    for m in ["Soft vote", "SVM", "Stack (LR)", "Random Forest + PCA", "kNN + PCA", "Naive Bayes + PCA", "Decision Tree"]:
        x = s.loc[m]
        diff = "" if m == "SVM" else (f"{signed(x['vs SVM (points)'])} ({f(x['vs SVM CI low'])} to "
                                      f"{signed(x['vs SVM CI high']) if float(x['vs SVM CI high']) > 0 else f(x['vs SVM CI high'])})")
        expected = (f"| {m} | {f(x['pooled accuracy %'])}% | {f(x['row CI low'])}-{f(x['row CI high'])} "
                    f"| {f(x['batch CI low'])}-{f(x['batch CI high'])} | {diff} |")
        assert row(expected) in README, expected
    # "only Naive Bayes and the Decision Tree are clearly worse than the SVM"
    clearly_worse = s.drop(index="SVM")
    clearly_worse = set(clearly_worse[clearly_worse["vs SVM CI high"].astype(float) < 0].index)
    assert clearly_worse == {"Naive Bayes + PCA", "Decision Tree"}
    assert (s.drop(index="SVM")["vs SVM CI low"].astype(float) < 0).all()      # nothing clearly better
    svm = s.loc["SVM"]
    assert f"95% interval of {f(svm['batch CI low'])}-{f(svm['batch CI high'])}%" in README


def test_subsample_std_table_matches_step14():
    s = read("step14_subsample_std_summary.csv")
    for x in s.to_dict("records"):
        f = {k: f"{float(v):.1f}%" for k, v in x.items() if k.endswith("%")}
        expected = (f"| {x['recordings used per new batch']} | {f['validation accuracy %']} | {f['test accuracy %']} "
                    f"| {f['test worst seed %']} | {f['batch 10 accuracy %']} |")
        assert row(expected) in README, expected
    by_n = s.set_index("recordings used per new batch")["test accuracy %"].astype(float)
    strict = float(read("step13_summary.csv").set_index("method").loc["SVM (strict)", "test accuracy %"])
    assert by_n["10"] < strict < by_n["25"]                     # "with 10 it is worse than the strict SVM"


def test_tuned_mlp_coral_table_matches_step14():
    s = read("step14_mlp_coral_summary.csv").set_index("method")
    for m, x in s.iterrows():
        expected = (f"| {m} | {float(x['validation accuracy % (untuned)']):.1f}% | {float(x['validation accuracy %']):.1f}% "
                    f"| {float(x['test accuracy %']):.1f}% |")
        assert row(expected) in README, expected
        assert str(x["wins on validation"]) == "False"           # "neither beats it there"
    strict = read("step13_summary.csv").set_index("method").loc["SVM (strict)"]
    assert row(f"| SVM (strict), for reference | | {float(strict['validation accuracy %']):.1f}% "
               f"| {float(strict['test accuracy %']):.1f}% |") in README


def test_training_period_table_and_claims_match_step14():
    s = read("step14_training_periods_summary.csv")
    strict = s[s["model"] != "SVM + per-batch standardisation"].copy()
    strict["strict rank"] = strict.groupby("training batches")["pooled accuracy %"].rank(ascending=False).astype(int)
    svm = strict[strict["model"] == "SVM"].set_index("training batches")
    pbs = s[s["model"] == "SVM + per-batch standardisation"].set_index("training batches")
    for period in svm.index:
        a, b = svm.loc[period], pbs.loc[period]
        shown = period.split("-")[0] + ("-" + period.split("-")[-1] if "-" in period else "")
        expected = (f"| {shown} | {a['test batches']} | {float(a['pooled accuracy %']):.1f}% | {a['strict rank']} "
                    f"| {float(b['pooled accuracy %']):.1f}% | {float(a['mean over batches %']):.1f}% "
                    f"| {float(b['mean over batches %']):.1f}% |")
        assert row(expected) in README, expected
    near = s["near batches"].astype(str).str.split("-")
    assert near.map(lambda bs: all(int(b) < 8 for b in bs) and 1 <= len(bs) <= 3).all()   # near and far never overlap
    falls = int((s["drop (points)"].astype(float) < 0).sum())
    assert f"{falls} of {len(s)} runs" in README
    assert f"best strict model in {int((svm['strict rank'] == 1).sum())} of {len(svm)} periods" in README
    assert (pbs["pooled accuracy %"].astype(float) > svm["pooled accuracy %"].astype(float)).all()   # "in every period"
    wins_mean = int((pbs["mean over batches %"].astype(float) > svm["mean over batches %"].astype(float)).sum())
    assert f"averaged over batches only in {wins_mean} of {len(pbs)}" in README
    first_test = s["test batches"].astype(str).str.split("-").str[0].astype(int)
    assert (first_test >= 4).all()                                                                  # batch 3 never tested


def test_readme_has_required_sections_and_no_viva_sheet():
    for heading in ["## Main results", "## Conclusions", "## Limitations", "## How to reproduce", "## Where each number comes from"]:
        assert heading in README, heading
    assert "viva" not in README.lower()
