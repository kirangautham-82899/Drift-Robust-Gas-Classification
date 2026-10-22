"""Save what the dashboard would otherwise compute on start-up, so its pages open instantly:

- a pre-parsed copy of the raw batch files (results/dashboard_raw.pkl.xz), and
- every saved model's predictions for the test batches 4-10 (results/dashboard_predictions.csv).

Run from the project root:  python scripts/make_dashboard_cache.py
The dashboard falls back to computing both itself if a file is missing, and tests check that both files match a fresh
computation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import common   # noqa: E402

raw = common.dl.load_all()
raw.to_pickle(common.RAW_CACHE, compression="xz")
print(f"saved {len(raw):,} raw rows to {common.RAW_CACHE}")

out = common.compute_predictions()
out.to_csv(common.PREDICTIONS_FILE, index=False)
print(f"saved {len(out):,} rows x {out.shape[1]} columns to {common.PREDICTIONS_FILE}")
