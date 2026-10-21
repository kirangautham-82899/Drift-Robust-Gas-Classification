"""Smoke tests for the dashboard: every page runs without errors and shows numbers that match the saved results."""
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


def run(view):
    at = AppTest.from_file(str(APP / "views" / view), default_timeout=180)
    at.run()
    return at


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_overview_runs_and_shows_the_dataset_size():
    at = run("overview.py")
    no_errors(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Samples"] == "12,077" and metrics["Gases"] == "5" and metrics["Batches"] == "10"
    assert metrics["Features"] == "128"


def test_overview_headline_numbers_come_from_the_result_files():
    at = run("overview.py")
    summary = pd.read_csv(ROOT / "results" / "step12_summary.csv", keep_default_na=False).set_index("Unnamed: 0")
    near_far = pd.read_csv(ROOT / "results" / "step10_near_far.csv", keep_default_na=False).set_index("model")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Best single model (SVM)"] == f"{summary.loc['SVM', 'mean accuracy %']:.1f}%"
    assert metrics["SVM on near batches 4-6"] == f"{near_far.loc['SVM', 'near (4-6) %']:.1f}%"
    assert metrics["SVM on far batches 8-10"] == f"{near_far.loc['SVM', 'far (8-10) %']:.1f}%"


def test_data_explorer_runs_and_the_data_check_passes():
    at = run("data_explorer.py")
    no_errors(at)
    assert any("Data check passed" in s.value for s in at.success)


def test_data_explorer_explanations_use_the_real_numbers():
    at = run("data_explorer.py")
    text = " ".join(i.value for i in at.info)
    assert "largest are batch 7 (3,045 samples) and batch 10 (3,000)" in text
    assert "batch 8, Acetone makes up 52%" in text
    assert "Batch 10 is perfectly balanced (20% each)" in text


def test_data_explorer_filters_work():
    at = run("data_explorer.py")
    batch_select = [w for w in at.multiselect if w.label == "Batches"][0]
    gas_select = [w for w in at.multiselect if w.label == "Gases"][0]
    batch_select.set_value([4]); gas_select.set_value(["Ammonia"])
    at.run()
    no_errors(at)
    assert any("12 recordings shown" in c.value for c in at.caption)     # batch 4 has 12 Ammonia samples (Step 1)


def test_drift_viewer_runs_and_controls_work():
    at = run("drift_viewer.py")
    no_errors(at)
    at.radio[1].set_value("Gas")             # colour the PCA points by gas
    at.checkbox[0].set_value(True)           # draw the batch-average paths
    at.run()
    no_errors(at)
    at.selectbox[0].set_value("Ammonia")
    at.run()
    no_errors(at)


def test_entry_point_runs():
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=180)
    at.run()
    no_errors(at)


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
