"""
app/pages/graph_explorer.py
---------------------------
Alias entry for Page 2: Knowledge Graph Explorer.
"""
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import app.pages.02_graph_explorer  # noqa: F401

