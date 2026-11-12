"""
app/app.py
----------
MediGraph AI — Main Application Entry Point & Navigation Router.

Intelligent Medical Knowledge Graph and Graph Neural Network Platform.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path and remove script directory to avoid package shadowing
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_APP_DIR = _PROJECT_ROOT / "app"
sys.path = [p for p in sys.path if Path(p).resolve() not in (_APP_DIR.resolve(), (_APP_DIR / "pages").resolve())]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure 'app' is recognized as a package even when Streamlit executes app.py directly
if "app" in sys.modules:
    mod = sys.modules["app"]
    if not hasattr(mod, "__path__"):
        mod.__path__ = [str(_APP_DIR)]

import streamlit as st

# ---------------------------------------------------------------------------
# Streamlit Page Configuration (Global)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediGraph AI — Biomedical Knowledge Graph & GNN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Modern Healthcare & AI Research Aesthetic
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
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 8px;
    }
    [data-testid="stSidebarNav"] span {
        font-size: 0.90rem !important;
        font-weight: 500 !important;
    }
    
    /* Main Headings */
    h1, h2, h3 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Medical Card Containers */
    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Branding & Provenance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                🧬
            </div>
            <div>
                <h3 style="color: #FFFFFF; margin: 0; font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em;">MediGraph AI</h3>
                <span style="color: #14B8A6; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">GNN Platform v0.1</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.80rem; margin-top: 4px; line-height: 1.4;'>"
        "Harvard PrimeKG Precision Medicine Knowledge Graph & Relational Graph Convolutional Network"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px; font-size: 0.76rem; line-height: 1.45; color: #CBD5E1; margin-bottom: 16px;">
            <strong style="color: #F59E0B;">⚠️ Research Prototype:</strong><br>
            Developed exclusively for biomedical knowledge graph analysis and GNN link prediction.
            Not evaluated for clinical diagnostic or treatment utility.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Modern Streamlit Page Navigation with Unique URL Paths
# ---------------------------------------------------------------------------
dashboard_page = st.Page(
    "pages/01_dashboard.py",
    title="Dashboard",
    icon="📊",
    url_path="dashboard",
    default=True,
)
explorer_page = st.Page(
    "pages/02_graph_explorer.py",
    title="Knowledge Graph Explorer",
    icon="🔍",
    url_path="graph-explorer",
)
predictions_page = st.Page(
    "pages/03_predictions.py",
    title="AI Predictions",
    icon="🧬",
    url_path="predictions",
)
performance_page = st.Page(
    "pages/04_model_performance.py",
    title="Model Performance",
    icon="📈",
    url_path="model-performance",
)

pg = st.navigation(
    {
        "Platform Modules": [
            dashboard_page,
            explorer_page,
            predictions_page,
            performance_page,
        ]
    }
)

# Execute the selected page
pg.run()

# ---------------------------------------------------------------------------
# Global Application Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 56px; padding: 24px 0 16px 0; border-top: 1px solid #E2E8F0; text-align: center; color: #64748B; font-size: 0.82rem;">
        <p style="margin: 0; font-weight: 600; color: #334155;">MediGraph AI — Biomedical Knowledge Graph and GNN Research Prototype</p>
        <p style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">
            Harvard PrimeKG Data Partition • 2-Layer R-GCN Encoder • DistMult Bilinear Decoder • PyTorch Geometric
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
