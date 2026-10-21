"""Save every saved model's predictions for the test batches (4-10) so the dashboard pages open instantly.

Run from the project root:  python scripts/make_dashboard_cache.py
The dashboard falls back to computing the predictions itself if the file is missing, and a test checks that the file
matches a fresh computation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import common   # noqa: E402

out = common.compute_predictions()
out.to_csv(common.PREDICTIONS_FILE, index=False)
print(f"saved {len(out):,} rows x {out.shape[1]} columns to {common.PREDICTIONS_FILE}")
