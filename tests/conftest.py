import os
import sys
from pathlib import Path

# joblib cannot count physical cores on some Windows machines and warns; give it the logical count instead
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
