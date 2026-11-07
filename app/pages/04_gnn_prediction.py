"""
app/pages/04_gnn_prediction.py
-------------------------------
GNN link prediction query interface.
Member 4 responsibility.

STUB: implement after Member 2 (model) and Member 3 (training/inference) are complete.
"""

import streamlit as st

st.set_page_config(page_title="GNN Prediction — MediGraph AI", page_icon="🤖", layout="wide")

st.title("🤖 GNN Link Prediction")
st.caption("Query the R-GCN model to discover predicted biomedical relationships")

st.warning(
    "⚠️ Model scores are research outputs and must NOT be interpreted "
    "as clinical probabilities or medical advice.",
)

st.info("🚧 **Coming soon.** Implement after model training is complete.", icon="🚧")

