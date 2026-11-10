"""
app/pages/dashboard.py
----------------------
Alias entry for Page 1: Dashboard.
"""
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Execute the main dashboard implementation
import app.pages.01_dashboard  # noqa: F401

