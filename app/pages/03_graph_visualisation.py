"""
app/pages/03_graph_visualisation.py
------------------------------------
Interactive knowledge graph visualisation.
Member 4 responsibility.

STUB: implement after nodes.csv and edges.csv are available.

Visualisation rules (see docs/API_CONTRACT.md §9.3):
- Existing edges → blue
- Predicted edges → orange dashed
- Always show legend
- Label scores as "Model Score", NOT "Probability"
"""

import streamlit as st

st.set_page_config(page_title="Graph Visualisation — MediGraph AI", page_icon="🕸️", layout="wide")

st.title("🕸️ Knowledge Graph Visualisation")
st.caption("Interactive explorer of the biomedical knowledge graph")

st.info("🚧 **Coming soon.** Implement after Member 1 delivers nodes.csv and edges.csv.", icon="🚧")

st.markdown(
    """
    **Legend (for reference):**
    - 🔵 Blue edges — existing biomedical relationships from PrimeKG
    - 🟠 Orange dashed edges — R-GCN model predictions
    """
)

