"""
app/main.py
-----------
MediGraph AI — Application Entry Point.
Delegates to app/app.py.
"""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Run the complete application entry point
import app.app  # noqa: F401
