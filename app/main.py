"""
app/main.py
-----------
MediGraph AI — Streamlit application entry point.
Member 4 responsibility.

Run with:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on the Python path when running from any directory
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediGraph AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_STATE_DEFAULTS: dict = {
    "model":        None,
    "node_df":      None,
    "edge_index":   None,
    "edge_type":    None,
    "node_mapping": None,
    "rel_mapping":  None,
    "predictions":  None,
}

for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Landing page content
# ---------------------------------------------------------------------------
st.title("🧬 MediGraph AI")
st.subheader("Intelligent Medical Knowledge Graph and GNN Platform")

st.warning(
    "⚠️ **Research & Educational Prototype Only.**  "
    "This system is NOT a clinical diagnosis or treatment recommendation tool.  "
    "Model outputs must NOT be interpreted as medical advice.",
    icon="⚠️",
)

st.markdown(
    """
    Use the sidebar to navigate between pages:

    | Page | Description |
    |------|-------------|
    | **Dashboard** | Graph statistics and system overview |
    | **Entity Search** | Search for biomedical entities and their neighbours |
    | **Graph Visualisation** | Interactive knowledge graph explorer |
    | **GNN Prediction** | Query the R-GCN for link predictions |
    | **Model Performance** | Training curves and evaluation metrics |
    """
)

st.info(
    "📂 **Status:** Data and model not yet loaded.  "
    "Complete preprocessing (Member 1) and training (Member 3) first.",
    icon="ℹ️",
)

