import numpy as np
import pytest

from src import data_loader as dl


@pytest.fixture(scope="module")
def raw():
    return dl.load_all()


@pytest.fixture(scope="module")
def clean(raw):
    return dl.remove_toluene(raw)


def test_raw_passes_all_checks(raw):
    results = dl.check_dataset(raw, dl.EXPECTED_BEFORE, allowed_labels=range(1, 7))
    failed = {k: v for k, (ok, v) in results.items() if not ok}
    assert not failed, failed


def test_clean_passes_all_checks(clean):
    results = dl.check_dataset(clean, dl.EXPECTED_AFTER, allowed_labels=range(1, 6))
    failed = {k: v for k, (ok, v) in results.items() if not ok}
    assert not failed, failed


def test_totals(raw, clean):
    assert len(raw) == 13910
    assert len(clean) == 12077


def test_toluene_gone_and_others_untouched(raw, clean):
    assert (clean["label"] == 6).sum() == 0
    assert (raw["label"] == 6).sum() == 1833
    kept = raw[raw["label"] != 6].reset_index(drop=True)
    assert clean.equals(kept)


def test_remove_toluene_does_not_modify_input(raw):
    n = len(raw)
    dl.remove_toluene(raw)
    assert len(raw) == n


def test_first_row_matches_raw_file(raw):
    # first line of batch1.dat: label 1, 1:15596.162100, 2:1.868245 ...
    row = raw.iloc[0]
    assert row["batch"] == 1 and row["label"] == 1
    assert np.isclose(row["f1"], 15596.1621)
    assert np.isclose(row["f2"], 1.868245)
