"""
app/pages/model_performance.py
------------------------------
Alias entry for Page 4: Model Performance.
"""
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import app.pages.04_model_performance  # noqa: F401
