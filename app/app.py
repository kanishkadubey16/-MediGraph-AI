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
    page_title="MediGraph AI — Biomedical Knowledge Graph & GNN",
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
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
        border-radius: 8px;
        padding: 8px 12px;
        transition: all 0.2s ease;
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
        background-color: #1E293B !important;
    }
    
    /* Main Headings */
    h1, h2, h3 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Hero Banner Container */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0D9488 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 20px;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.5;
    }
    .hero-stats-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 16px;
    }
    .hero-stat-chip {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 9999px;
        padding: 6px 16px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Card Containers */
    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .med-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.08);
    }
    
    /* Pill Tags */
    .entity-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px;
    }
    
    /* Architecture Flow Box */
    .arch-step-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Content
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">
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
        "PrimeKG Precision Knowledge Graph & Relational Graph Convolutional Network"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>Navigation Modules</p>", unsafe_allow_html=True)
    
    st.page_link("pages/01_dashboard.py", label="Executive Dashboard", icon="📊")
    st.page_link("pages/02_graph_explorer.py", label="Knowledge Graph Explorer", icon="🔍")
    st.page_link("pages/03_predictions.py", label="GNN Prediction Studio", icon="🧬")
    st.page_link("pages/04_model_performance.py", label="Model Performance & Eval", icon="📈")
    
    st.divider()

    st.markdown(
        """
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px; font-size: 0.78rem; line-height: 1.45; color: #CBD5E1;">
            <strong style="color: #F59E0B;">⚠️ Research Prototype:</strong><br>
            Developed exclusively for biomedical knowledge graph analysis and GNN link prediction.
            Not evaluated for clinical diagnostic or treatment utility.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load & Verify System Assets
# ---------------------------------------------------------------------------
cfg = load_config()
data_loaded = load_project_data(cfg)

ckpt_path = Path("artifacts/checkpoints/best_model.pt")
metrics_path = Path("artifacts/metrics/metrics.json")
has_model = ckpt_path.exists()
has_metrics = metrics_path.exists()

node_count = len(st.session_state.node_df) if data_loaded else 0
edge_count = st.session_state.edge_index.shape[1] if data_loaded else 0
rel_count = len(st.session_state.rel_mapping) if data_loaded else 0

# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <h1 class="hero-title">Intelligent Medical Knowledge Graph & Relational GNN</h1>
        <p class="hero-subtitle">
            Explore multi-scale biomedical relationships, navigate precision medicine subgraphs,
            and predict potentially novel drug-disease indications and genomic interactions using
            Relational Graph Convolutional Networks (R-GCN) with DistMult scoring.
        </p>
        <div class="hero-stats-row">
            <div class="hero-stat-chip">🌐 {node_count:,} PrimeKG Nodes</div>
            <div class="hero-stat-chip">🔗 {edge_count:,} Directed Edges</div>
            <div class="hero-stat-chip">🏷️ {rel_count} Relation Types</div>
            <div class="hero-stat-chip">🧠 2-Layer R-GCN (64-dim)</div>
            <div class="hero-stat-chip">🎯 Filtered DistMult Scoring</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_disclaimer_banner()

# ---------------------------------------------------------------------------
# System Status Indicators
# ---------------------------------------------------------------------------
render_section_header("System & Infrastructure Status", "Real-time health verification of data layers, neural weights, and empirical benchmarks", "🩺")

status_cols = st.columns(3)
with status_cols[0]:
    if data_loaded:
        render_metric_card("Knowledge Graph Data", f"{node_count:,} Nodes", f"{edge_count:,} verified biomedical edges", icon="🌐", badge="Ready", color_accent="teal")
    else:
        render_metric_card("Knowledge Graph Data", "Data Missing", "Run src/data/preprocessing.py", icon="⚠️", badge="Required", color_accent="coral")

with status_cols[1]:
    if has_model:
        render_metric_card("R-GCN Neural Model", "Trained & Validated", "Epoch 10 • best_model.pt", icon="🧠", badge="Active", color_accent="navy")
    else:
        render_metric_card("R-GCN Neural Model", "Checkpoint Missing", "Run src/training/train.py", icon="⚠️", badge="Untrained", color_accent="amber")

with status_cols[2]:
    if has_metrics:
        render_metric_card("Empirical Evaluation", "Benchmarks Ready", "Filtered MRR & Hits@1/3/10", icon="📊", badge="Verified", color_accent="purple")
    else:
        render_metric_card("Empirical Evaluation", "Metrics Missing", "Run src/training/evaluate.py", icon="⚠️", badge="Pending", color_accent="slate")

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Platform Navigation Cards
# ---------------------------------------------------------------------------
render_section_header("Explore Platform Modules", "Interactive tools for graph analysis, multi-hop exploration, and neural link inference", "🧭")

nav_row1_col1, nav_row1_col2 = st.columns(2)

with nav_row1_col1:
    st.markdown(
        """
        <div class="med-card" style="border-top: 4px solid #0D9488;">
            <div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-size: 1.8rem;">📊</span>
                    <span style="background: #F0FDFA; color: #0F766E; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; text-transform: uppercase;">Overview</span>
                </div>
                <h3 style="color: #0F172A; font-size: 1.25rem; margin: 0 0 8px 0; font-weight: 700;">Executive Dashboard</h3>
                <p style="color: #64748B; font-size: 0.88rem; line-height: 1.5; margin-bottom: 16px;">
                    Comprehensive topological analysis of the PrimeKG partition: entity distributions,
                    relationship breakdowns, degree distribution, density, and hub entity rankings.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/01_dashboard.py", label="Open Executive Dashboard →", icon="📊")

with nav_row1_col2:
    st.markdown(
        """
        <div class="med-card" style="border-top: 4px solid #3B82F6;">
            <div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-size: 1.8rem;">🔍</span>
                    <span style="background: #EFF6FF; color: #1E40AF; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; text-transform: uppercase;">Interactive</span>
                </div>
                <h3 style="color: #0F172A; font-size: 1.25rem; margin: 0 0 8px 0; font-weight: 700;">Knowledge Graph Explorer</h3>
                <p style="color: #64748B; font-size: 0.88rem; line-height: 1.5; margin-bottom: 16px;">
                    Search any disease, drug, gene, or phenotype. Interactively visualize 1-hop and 2-hop
                    neighborhoods with Plotly 2D or PyVis physics-enabled dynamic force layout.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/02_graph_explorer.py", label="Launch Graph Explorer →", icon="🔍")

nav_row2_col1, nav_row2_col2 = st.columns(2)

with nav_row2_col1:
    st.markdown(
        """
        <div class="med-card" style="border-top: 4px solid #8B5CF6;">
            <div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-size: 1.8rem;">🧬</span>
                    <span style="background: #F5F3FF; color: #5B21B6; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; text-transform: uppercase;">Inference</span>
                </div>
                <h3 style="color: #0F172A; font-size: 1.25rem; margin: 0 0 8px 0; font-weight: 700;">GNN Prediction Studio</h3>
                <p style="color: #64748B; font-size: 0.88rem; line-height: 1.5; margin-bottom: 16px;">
                    Query the trained R-GCN encoder and DistMult decoder to rank missing biomedical links,
                    discover novel drug indications, and compare predictions with known facts.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/03_predictions.py", label="Launch Prediction Studio →", icon="🧬")

with nav_row2_col2:
    st.markdown(
        """
        <div class="med-card" style="border-top: 4px solid #F59E0B;">
            <div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-size: 1.8rem;">📈</span>
                    <span style="background: #FFFBEB; color: #92400E; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; text-transform: uppercase;">Ablation</span>
                </div>
                <h3 style="color: #0F172A; font-size: 1.25rem; margin: 0 0 8px 0; font-weight: 700;">Model Performance & Ablation</h3>
                <p style="color: #64748B; font-size: 0.88rem; line-height: 1.5; margin-bottom: 16px;">
                    Review training convergence curves, test set MRR & Hits@K metrics,
                    and empirical ablation proving message-passing benefits over DistMult alone.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/04_model_performance.py", label="Inspect Model Benchmarks →", icon="📈")

st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Platform Architectural Pipeline
# ---------------------------------------------------------------------------
render_section_header("Deep Learning Pipeline Architecture", "End-to-end representation learning and link inference workflow", "🏛️")

flow_cols = st.columns(4)
with flow_cols[0]:
    st.markdown(
        """
        <div class="arch-step-box">
            <div style="color: #0D9488; font-size: 1.6rem; margin-bottom: 6px;">📥</div>
            <strong style="color: #0F172A; font-size: 0.88rem;">1. PrimeKG Ingestion</strong>
            <p style="color: #64748B; font-size: 0.76rem; margin-top: 4px;">Curated multi-relational medical graph partition (39 nodes, 36 edges, 7 relation types).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with flow_cols[1]:
    st.markdown(
        """
        <div class="arch-step-box">
            <div style="color: #3B82F6; font-size: 1.6rem; margin-bottom: 6px;">🧠</div>
            <strong style="color: #0F172A; font-size: 0.88rem;">2. R-GCN Encoder</strong>
            <p style="color: #64748B; font-size: 0.76rem; margin-top: 4px;">2-layer relational graph convolution with basis decomposition and neighborhood aggregation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with flow_cols[2]:
    st.markdown(
        """
        <div class="arch-step-box">
            <div style="color: #8B5CF6; font-size: 1.6rem; margin-bottom: 6px;">🎯</div>
            <strong style="color: #0F172A; font-size: 0.88rem;">3. DistMult Decoder</strong>
            <p style="color: #64748B; font-size: 0.76rem; margin-top: 4px;">Bilinear candidate scoring: <code>score(h,r,t) = z_h ⊙ W_r ⊙ z_t</code> across candidate tails.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with flow_cols[3]:
    st.markdown(
        """
        <div class="arch-step-box">
            <div style="color: #10B981; font-size: 1.6rem; margin-bottom: 6px;">💡</div>
            <strong style="color: #0F172A; font-size: 0.88rem;">4. Hypothesis Ranking</strong>
            <p style="color: #64748B; font-size: 0.76rem; margin-top: 4px;">Ranked candidate predictions with ground truth verification and confidence metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
