import numpy as np
import pandas as pd

from src import data_loader as dl
from src import eda


def test_feature_info_layout():
    info = eda.feature_info()
    assert len(info) == 128
    assert info["sensor"].nunique() == 16
    assert info.groupby("sensor").size().eq(8).all()
    assert info.loc[0, "type"] == "dR" and info.loc[7, "type"] == "decay_a0.1"
    assert info.loc[8, "sensor"] == 2 and info.loc[8, "type"] == "dR"


def test_class_percent_rows_sum_to_100():
    clean = dl.remove_toluene(dl.load_all())
    pct = eda.class_percent_per_batch(clean)
    assert np.allclose(pct.sum(axis=1), 100)
    assert list(pct.index) == list(range(1, 11))
    assert pct.shape[1] == 5


def test_type_summary_has_all_types():
    clean = dl.remove_toluene(dl.load_all())
    summary = eda.type_summary(clean)
    assert list(summary.index) == eda.FEATURE_TYPES
    assert (summary["max"] >= summary["min"]).all()
