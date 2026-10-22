"""Tests for the dashboard: every page runs, shows numbers that match the saved results, and the helpers behave."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import common  # noqa: E402
import design  # noqa: E402

VIEWS = ["overview", "data_explorer", "drift_viewer", "model_results", "per_batch_drift", "confusion_matrices",
         "random_vs_time_aware", "live_demo", "about"]


def run(view):
    at = AppTest.from_file(str(APP / "views" / f"{view}.py"), default_timeout=180)
    at.run()
    return at


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def page_text(at):
    """All visible text of a page: markdown (our HTML components included), captions, info boxes."""
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
                    + [s.value for s in at.success] + [w.value for w in at.warning])


def read(name):
    return pd.read_csv(ROOT / "results" / name, keep_default_na=False)


# ------------------------------------------------------------------------------------------------ every page runs
@pytest.mark.parametrize("view", VIEWS)
def test_every_page_runs_without_errors(view):
    at = run(view)
    no_errors(at)
    assert "<h1>" in page_text(at), "each page starts with the hero banner"


def test_entry_point_runs_with_all_pages_registered():
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=180)
    at.run()
    no_errors(at)
    assert "Can a model keep recognising gases" in page_text(at)


# ------------------------------------------------------------------------------------------------ numbers come from the results
def test_overview_numbers_come_from_the_result_files():
    text = page_text(run("overview"))
    summary = read("step12_summary.csv").set_index("Unnamed: 0")
    near_far = read("step10_near_far.csv").set_index("model")
    exams = read("step11_accuracy_table.csv").set_index("model")
    assert "12,077 recordings" in text
    assert f'<div class="value">{summary.loc["SVM", "mean accuracy %"]:.1f}%</div>' in text
    assert f'<div class="value">{near_far.loc["SVM", "near (4-6) %"]:.1f}%</div>' in text
    assert f'<div class="value">{near_far.loc["SVM", "far (8-10) %"]:.1f}%</div>' in text
    assert f'<div class="value">{exams.loc["SVM", "random, same sizes"]:.1f}%</div>' in text


def test_data_explorer_runs_the_data_check_and_explains_with_real_numbers():
    text = page_text(run("data_explorer"))
    assert "Data check passed" in text
    assert "The largest batches are batch 7 (3,045 samples) and batch 10 (3,000)" in text
    assert "batch 8, Acetone makes up 52%" in text
    assert "Batch 10 is perfectly balanced (20% each)" in text


def test_data_explorer_filters_work():
    at = run("data_explorer")
    [w for w in at.multiselect if w.label == "Batches"][0].set_value([4])
    [w for w in at.multiselect if w.label == "Gases"][0].set_value(["Ammonia"])
    at.run()
    no_errors(at)
    assert any("12 recordings shown" in c.value for c in at.caption)         # batch 4 has 12 Ammonia samples (Step 1)


def test_drift_viewer_controls_work():
    at = run("drift_viewer")
    at.radio[1].set_value("Gas")               # colour the PCA points by gas
    at.checkbox[0].set_value(True)             # draw the batch-average paths
    at.run()
    no_errors(at)
    at.selectbox[0].set_value("Ammonia")
    at.run()
    no_errors(at)
    assert "Ammonia stays closest to home" in page_text(at)


def test_model_results_leaderboard_matches_the_notebooks():
    lb = common.leaderboard(common.all_results(), "A", "None", "Default")          # Steps 5 and 7
    assert lb.loc["SVM", "accuracy"] == pytest.approx(72.7, abs=0.05)
    assert lb.loc["kNN", "accuracy"] == pytest.approx(69.6, abs=0.05)
    assert lb.loc["Gradient Boosting", "accuracy"] == pytest.approx(55.8, abs=0.05)
    assert common.leaderboard(common.all_results(), "A", "None", "Tuned").loc["SVM", "accuracy"] == pytest.approx(73.2, abs=0.05)    # Step 8
    assert common.leaderboard(common.all_results(), "A", "LDA", "Default").loc["Naive Bayes", "accuracy"] == pytest.approx(71.6, abs=0.05)


def test_model_results_controls_work():
    at = run("model_results")
    at.radio[0].set_value("C")
    at.run()
    no_errors(at)
    at.radio[0].set_value("A")
    at.radio[3].set_value("Tuned")
    at.radio[2].set_value("PCA")
    at.run()
    no_errors(at)
    at.radio[0].set_value("B")                  # tuned is not available for B: falls back with a message
    at.run()
    no_errors(at)
    assert any("only tested on protocol A" in i.value for i in at.info)


def test_per_batch_page_controls_and_the_soft_vote_caution():
    at = run("per_batch_drift")
    assert "beats the SVM on <strong>3</strong> batches, loses on <strong>3</strong> and ties on <strong>1</strong>" in page_text(at)
    at.radio[0].set_value("Macro-F1")
    at.checkbox[0].set_value(False)
    at.run()
    no_errors(at)


def test_confusion_page_controls_work():
    at = run("confusion_matrices")
    at.selectbox[2].set_value("Batch 10")
    at.checkbox[0].set_value(True)
    at.run()
    no_errors(at)


def test_random_vs_time_aware_page_uses_the_step11_numbers():
    at = run("random_vs_time_aware")
    text = page_text(at)
    exams = read("step11_accuracy_table.csv").set_index("model")
    for col in ["time-aware (A)", "random, same sizes", "random 80/20"]:
        assert f'<div class="value">{exams.loc["SVM", col]:.1f}%</div>' in text
    at.selectbox[0].set_value("Gradient Boosting")
    at.run()
    no_errors(at)
    assert f'<div class="value">{exams.loc["Gradient Boosting", "random 80/20"]:.1f}%</div>' in page_text(at)


def test_about_page_states_the_limitations():
    text = page_text(run("about"))
    for phrase in ["Five gases only", "Dennler", "Small, uneven batches", "Gas concentration is missing", "One training period",
                   "Intervals are too narrow", "No perfect accuracy"]:
        assert phrase in text, phrase


# ------------------------------------------------------------------------------------------------ live demo
def test_live_demo_controls_work():
    at = run("live_demo")
    no_errors(at)
    assert "says this is" in page_text(at)
    at.selectbox[0].set_value("Random Forest + PCA")      # model is the first selectbox ...
    at.run()
    no_errors(at)
    at.selectbox[1].set_value(5)                           # ... then batch, then gas
    at.selectbox[2].set_value("Ammonia")
    at.run()
    no_errors(at)
    assert "The recording really is:" in page_text(at)


def test_live_demo_reports_correct_and_wrong_answers_consistently():
    at = run("live_demo")
    text = page_text(at)
    assert ("✓ Correct" in text) != ("✗ Wrong" in text)


def test_demo_predictions_match_the_saved_predictions_for_every_model():
    preds = common.predictions()
    df = common.load_data()
    test = df[df["batch"] >= 4].reset_index(drop=True)
    idx = np.random.default_rng(0).choice(len(test), 300, replace=False)
    X = test.loc[idx, common.dl.FEATURE_COLS].to_numpy()
    for model in common.DEMO_MODELS:
        labels, probs = common.demo_predict(model, X)
        assert probs.shape == (300, 5) and np.allclose(probs.sum(axis=1), 1.0)
        assert (labels == preds.loc[idx, model].to_numpy()).all(), model
        agree = (probs.argmax(axis=1) + 1 == labels).mean()
        assert agree == 1.0 if model != "SVM" else agree > 0.95, (model, agree)     # SVM bars come from a separate calibration


def test_parse_upload_accepts_the_three_layouts_and_rejects_bad_files():
    rng = np.random.default_rng(1)
    plain = pd.DataFrame(rng.normal(size=(4, 128)))
    named = pd.DataFrame(rng.normal(size=(4, 128)), columns=common.dl.FEATURE_COLS)
    with_label = pd.concat([pd.Series([1, 2, 3, 5], name="label"), plain.set_axis(range(1, 129), axis=1)], axis=1)
    for frame, expect_labels in [(plain, False), (named, False), (with_label, True)]:
        X, labels = common.parse_upload(frame)
        assert X.shape == (4, 128) and (labels is not None) == expect_labels
    text_in_feature = plain.copy().astype(object)
    text_in_feature.iloc[0, 0] = "a"
    missing_value = plain.copy()
    missing_value.iloc[0, 0] = np.nan
    bad = {
        "wrong number of columns": pd.DataFrame(rng.normal(size=(3, 50))),
        "text in a feature": text_in_feature,
        "missing value": missing_value,
        "bad gas label": with_label.assign(label=[1, 2, 3, 9]),
        "empty": plain.iloc[0:0],
        "too many rows": pd.DataFrame(np.zeros((common.MAX_UPLOAD_ROWS + 1, 128))),
    }
    for reason, frame in bad.items():
        with pytest.raises(ValueError):
            common.parse_upload(frame)


# ------------------------------------------------------------------------------------------------ saved results behind the pages
def test_batch_counts_match_step1_results():
    counts = common.batch_counts()
    saved = pd.read_csv(ROOT / "results" / "step1_counts.csv", index_col=0)
    for b in range(1, 11):
        assert counts.loc[b, "before"] == saved.loc[str(b), "before"]
        assert counts.loc[b, "Toluene removed"] == saved.loc[str(b), "toluene_removed"]
        assert counts.loc[b, "after"] == saved.loc[str(b), "after"]


def test_gas_distance_table_matches_step3_results():
    table, between = common.centroid_shift()
    saved = pd.read_csv(ROOT / "results" / "step3_centroid_shift.csv", index_col=0)
    assert np.allclose(table.to_numpy(), saved[table.columns].to_numpy(), atol=0.006)   # CSV is rounded to 2 decimals
    assert between == pytest.approx(9.0, abs=0.05)


def test_saved_predictions_file_matches_a_fresh_computation():
    assert common.PREDICTIONS_FILE.exists(), "run scripts/make_dashboard_cache.py"
    saved = pd.read_csv(common.PREDICTIONS_FILE, keep_default_na=False)
    fresh = common.compute_predictions()
    assert list(saved.columns) == list(fresh.columns) and len(saved) == len(fresh) == 8881
    assert (saved.to_numpy() == fresh.to_numpy()).all()


def test_saved_model_predictions_reproduce_the_step10_accuracies():
    preds = common.predictions()
    ci = pd.read_csv(ROOT / "results" / "step10_per_batch_ci.csv", keep_default_na=False)
    pooled = ci[ci["group"] == "4-10 (all)"].set_index("model")["accuracy"].astype(float)
    assert len(preds) == 8881 and set(pooled.index) == set(c for c in preds.columns if c not in ("batch", "label"))
    for model, acc in pooled.items():
        assert (preds[model] == preds["label"]).mean() == pytest.approx(acc, abs=0.0006)    # CSV rounded to 4 decimals


def test_raw_data_cache_matches_a_fresh_parse():
    assert common.RAW_CACHE.exists(), "run scripts/make_dashboard_cache.py"
    assert pd.read_pickle(common.RAW_CACHE).equals(common.dl.load_all())


def test_confusion_helper_counts_every_recording():
    preds = common.predictions()
    cm = common.confusion(preds["label"], preds["SVM"])
    assert cm.sum() == len(preds) and cm.shape == (5, 5)
    assert np.trace(cm) == (preds["label"] == preds["SVM"]).sum()


def test_ammonia_claim_in_the_readme_matches_the_saved_models():
    """README: only 1 of the 10 analysed models drops below 40% found on any batch for Ammonia, 4 to 8 for each other gas."""
    preds = common.predictions()
    models = [c for c in preds.columns if c not in ("batch", "label")]
    below = dict.fromkeys(common.GASES, 0)
    for m in models:
        worst = np.full(5, 100.0)
        for b in range(4, 11):
            d = preds[preds["batch"] == b]
            cm = common.confusion(d["label"], d[m])
            worst = np.minimum(worst, np.diag(cm) / cm.sum(axis=1) * 100)
        for gas, w in zip(common.GASES, worst):
            below[gas] += int(w < 40)
    assert len(models) == 10 and below["Ammonia"] == 1
    assert all(4 <= below[g] <= 8 for g in common.GASES if g != "Ammonia"), below
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert "only 1 of the 10 analysed models drops below 40% found on any batch, against 4 to 8 of 10 for each other gas" in readme


# ------------------------------------------------------------------------------------------------ design system
def test_the_validated_palette_is_unchanged():
    """The categorical colours passed the colour-blind and contrast checks as exactly this set: do not edit by accident."""
    assert design.CATEGORICAL["light"] == ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    assert design.CATEGORICAL["dark"] == ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def test_every_entity_keeps_one_colour_and_gases_have_distinct_shapes():
    assert len({design.gas_color(g) for g in design.GASES}) == 5
    assert len({design.GAS_SYMBOL[g] for g in design.GASES}) == 5
    assert design.model_color("SVM (untuned)") == design.model_color("SVM")
    assert len({design.model_color(m) for m in design.MODEL_SLOT}) == 8


def test_chart_style_is_applied():
    import plotly.graph_objects as go
    fig = design.style(go.Figure(go.Scatter(x=[1, 2], y=[1, 2])), height=300)
    assert fig.layout.height == 300 and fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.yaxis.showgrid and fig.layout.xaxis.showgrid is False
