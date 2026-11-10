"""
app/app.py
----------
MediGraph AI — Main Application Entry Point.

Intelligent Medical Knowledge Graph and Graph Neural Network Platform.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.utils.config import load_config
from app.components.data_loader import load_project_data
from app.components.metric_cards import (
    render_metric_card,
    render_kpi_row,
    render_section_header,
    render_disclaimer_banner,
)

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediGraph AI — Medical Knowledge Graph & GNN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Modern Medical Technology Interface
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Navigation Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"] a:hover {
        background-color: #1E293B !important;
        border-radius: 6px;
    }
    
    /* Main Headings */
    h1, h2, h3 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Card Containers */
    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0D9488;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 8px 18px;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #0F766E;
    }
    
    /* Clean Divider */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Content
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 MediGraph AI")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.82rem; margin-top: -8px;'>"
        "Intelligent Medical Knowledge Graph & GNN Platform"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Application Pages**")
    st.markdown(
        """
        - [📊 01 Dashboard](dashboard)
        - [🔍 02 Graph Explorer](graph_explorer)
        - [🧬 03 GNN Prediction](predictions)
        - [📈 04 Model Performance](model_performance)
        """
    )
    st.divider()

    st.markdown(
        """
        <div style="background: #1E293B; border-radius: 6px; padding: 12px; font-size: 0.78rem; line-height: 1.4; color: #CBD5E1;">
            <strong>⚠️ Research Prototype:</strong><br>
            Designed exclusively for biomedical knowledge graph analysis and GNN link prediction.
            Not evaluated for clinical diagnostic or treatment utility.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main Landing Header
# ---------------------------------------------------------------------------
st.title("🧬 MediGraph AI")
st.markdown(
    "<h3 style='color: #0D9488; font-weight: 600; font-size: 1.35rem; margin-top: -12px; margin-bottom: 20px;'>"
    "Intelligent Medical Knowledge Graph and Graph Neural Network Platform"
    "</h3>",
    unsafe_allow_html=True,
)

render_disclaimer_banner()

# ---------------------------------------------------------------------------
# Load & Verify System Assets
# ---------------------------------------------------------------------------
cfg = load_config()
data_loaded = load_project_data(cfg)

ckpt_path = Path("artifacts/checkpoints/best_model.pt")
metrics_path = Path("artifacts/metrics/metrics.json")
has_model = ckpt_path.exists()
has_metrics = metrics_path.exists()

# ---------------------------------------------------------------------------
# System Status Banner
# ---------------------------------------------------------------------------
render_section_header("System & Infrastructure Status", "Verification of biomedical data layers, model checkpoints, and evaluation metrics", "🩺")

status_cols = st.columns(3)
with status_cols[0]:
    if data_loaded:
        node_df = st.session_state.node_df
        edge_index = st.session_state.edge_index
        render_metric_card("Knowledge Graph Data", f"{len(node_df):,} Nodes / {edge_index.shape[1]:,} Edges", "PrimeKG processed partition", icon="✅", badge="Ready", color_accent="teal")
    else:
        render_metric_card("Knowledge Graph Data", "Data Not Found", "Execute src/data/preprocessing.py", icon="⚠️", badge="Action Required", color_accent="coral")

with status_cols[1]:
    if has_model:
        render_metric_card("R-GCN Neural Model", "Checkpoint Loaded", "artifacts/checkpoints/best_model.pt", icon="🧠", badge="Trained", color_accent="navy")
    else:
        render_metric_card("R-GCN Neural Model", "Checkpoint Missing", "Execute src/training/train.py", icon="⚠️", badge="Untrained", color_accent="amber")

with status_cols[2]:
    if has_metrics:
        render_metric_card("Empirical Evaluation", "Metrics Available", "MRR, Hits@K & DistMult ablation", icon="📊", badge="Verified", color_accent="purple")
    else:
        render_metric_card("Empirical Evaluation", "Metrics Missing", "Execute src/training/evaluate.py", icon="⚠️", badge="Pending", color_accent="slate")

st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Platform Navigation Cards
# ---------------------------------------------------------------------------
render_section_header("Explore Platform Modules", "Navigate to specialized interactive analysis pages", "🧭")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        """
        <div class="med-card">
            <h4 style="color: #0F172A; margin-top: 0;">📊 Executive Dashboard</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Inspect core entity distributions (diseases, drugs, genes, phenotypes),
                relationship frequency breakdown, degree centralities, and graph-theoretic connectivity.
            </p>
            <p style="font-size: 0.85rem; color: #0D9488; font-weight: 600;">
                ➔ Navigate via the sidebar to: <strong>01 Dashboard</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="med-card">
            <h4 style="color: #0F172A; margin-top: 0;">🔍 Knowledge Graph Explorer</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Search for any disease, compound, or gene. Interactively visualize 1-hop and 2-hop
                local relational neighborhoods with entity-colored nodes and typed directed edges.
            </p>
            <p style="font-size: 0.85rem; color: #0D9488; font-weight: 600;">
                ➔ Navigate via the sidebar to: <strong>02 Graph Explorer</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        """
        <div class="med-card">
            <h4 style="color: #0F172A; margin-top: 0;">🧬 GNN Link Prediction Studio</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Query the trained 2-layer Relational GCN (R-GCN) encoder and DistMult decoder
                to rank novel candidate drug targets, disease associations, and symptom links.
            </p>
            <p style="font-size: 0.85rem; color: #0D9488; font-weight: 600;">
                ➔ Navigate via the sidebar to: <strong>03 Predictions</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="med-card">
            <h4 style="color: #0F172A; margin-top: 0;">📈 Model Performance & Ablation</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Review loss convergence curves, filtered MRR, Hits@1/3/10, ROC-AUC,
                and empirical ablation comparing R-GCN against the DistMult baseline.
            </p>
            <p style="font-size: 0.85rem; color: #0D9488; font-weight: 600;">
                ➔ Navigate via the sidebar to: <strong>04 Model Performance</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Platform Architectural Overview
# ---------------------------------------------------------------------------
render_section_header("Platform Architecture & GNN Pipeline", "End-to-end representation learning workflow", "🏛️")

st.markdown(
    """
    ```
    ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
    │  PrimeKG Curated Graph  │ ───►  │   2-Layer R-GCN Encoder │ ───►  │    DistMult Decoder     │
    │  • 39 Sample Nodes      │       │   • Basis Decomposition │       │    • Bilinear Scoring   │
    │  • 36 Directed Edges    │       │   • Message Passing     │       │    • Candidate Ranking  │
    │  • 7 Relation Types     │       │   • 128-dim Embeddings  │       │    • Novel Link Hypoth. │
    └─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
    ```
    """
)
