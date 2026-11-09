"""
app/pages/01_dashboard.py
-------------------------
Dashboard page — graph statistics and system overview.
Member 4 responsibility — MediGraph AI.

Displays:
  • Data and model load status via st.metric cards
  • Graph statistics (node/edge counts, type breakdowns)
  • Node-type distribution bar chart (Plotly)
  • Relation-type distribution bar chart (Plotly, top-20)
  • Model status and last-checkpoint metrics
  • A "Load Data" section with a file-system browser button

Design decisions
----------------
* Blues (#4A90D9 family) for existing-data metrics; orange (#E8831A) for
  model-output metrics — per API_CONTRACT.md §9.3.
* All paths sourced from config.yaml via load_config().
* Never prints bare; uses get_logger(__name__).
* Disclaimer is always visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger
from app.components.data_loader import get_data_status, load_project_data

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard — MediGraph AI",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    cfg = load_config()
except Exception as exc:  # noqa: BLE001
    st.error(f"❌ [MediGraph] Cannot load config.yaml: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Constants / colour palette
# ---------------------------------------------------------------------------
_BLUE = "#4A90D9"
_ORANGE = "#E8831A"
_DISCLAIMER = (
    "⚠️ **Research & Educational Prototype Only.** "
    "This system is NOT a clinical diagnosis or treatment recommendation tool. "
    "Model outputs must NOT be interpreted as medical advice."
)
_NODE_TYPE_COLOURS: dict[str, str] = {
    "disease":             "#E05C5C",
    "drug":                "#4A90D9",
    "gene":                "#50C878",
    "protein":             "#9B59B6",
    "phenotype":           "#F39C12",
    "pathway":             "#1ABC9C",
    "anatomy":             "#E67E22",
    "biological_process":  "#2980B9",
    "molecular_function":  "#8E44AD",
    "cellular_component":  "#27AE60",
    "exposure":            "#C0392B",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Dashboard")
st.caption("Graph statistics and system overview — MediGraph AI")

st.warning(_DISCLAIMER, icon="⚠️")
st.divider()

# ---------------------------------------------------------------------------
# Data status
# ---------------------------------------------------------------------------
status = get_data_status()

st.subheader("🔌 System Status")
col_data, col_model, col_nodes, col_edges = st.columns(4)

with col_data:
    st.metric(
        label="Graph Data",
        value="✅ Loaded" if status["data_loaded"] else "❌ Not Loaded",
        delta=None,
    )
with col_model:
    st.metric(
        label="GNN Model",
        value="✅ Loaded" if status["model_loaded"] else "❌ Not Loaded",
        delta=None,
    )
with col_nodes:
    st.metric(
        label="Total Nodes",
        value=f"{status['num_nodes']:,}" if status["data_loaded"] else "—",
    )
with col_edges:
    st.metric(
        label="Total Edges",
        value=f"{status['num_edges']:,}" if status["data_loaded"] else "—",
    )

st.divider()

# ---------------------------------------------------------------------------
# Graph statistics (only when data is loaded)
# ---------------------------------------------------------------------------
if status["data_loaded"]:
    node_df = st.session_state.node_df

    # ---- Derived counts -------------------------------------------------
    num_node_types = node_df["entity_type"].nunique()
    rel_mapping: dict = st.session_state.rel_mapping or {}
    num_relation_types = len(rel_mapping)

    edge_index = st.session_state.edge_index
    num_edges = int(edge_index.shape[1])
    num_nodes = len(node_df)

    st.subheader("📈 Graph Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{num_nodes:,}")
    c2.metric("Edges", f"{num_edges:,}")
    c3.metric("Node Types", str(num_node_types))
    c4.metric("Relation Types", str(num_relation_types))

    st.divider()

    # ---- Node-type distribution bar chart ------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Node Type Distribution")
        type_counts = (
            node_df["entity_type"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "entity_type", "count": "count"})
        )
        # Pandas ≥2.0 value_counts column names differ; normalise
        if "entity_type" not in type_counts.columns:
            type_counts.columns = ["entity_type", "count"]

        type_colours = [
            _NODE_TYPE_COLOURS.get(t, _BLUE)
            for t in type_counts["entity_type"]
        ]
        fig_node_types = px.bar(
            type_counts,
            x="entity_type",
            y="count",
            labels={"entity_type": "Entity Type", "count": "Count"},
            color="entity_type",
            color_discrete_map=_NODE_TYPE_COLOURS,
            title="Nodes by Entity Type",
        )
        fig_node_types.update_layout(
            showlegend=False,
            xaxis_tickangle=-35,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
        )
        st.plotly_chart(fig_node_types, use_container_width=True)

    # ---- Relation-type distribution bar chart (top 20) -----------------
    with col_right:
        st.markdown("#### Relation Type Distribution (Top 20)")
        import pandas as pd  # local import to avoid issues with page-reload order

        edge_df_rel: pd.DataFrame | None = None
        # Try to reconstruct relation labels from edge_type tensor + rel_mapping
        rel_mapping_inv = {v: k for k, v in rel_mapping.items()}
        edge_type_tensor = st.session_state.edge_type
        if edge_type_tensor is not None:
            import numpy as np

            rel_ids = edge_type_tensor.numpy()
            rel_names = [rel_mapping_inv.get(int(r), str(r)) for r in rel_ids]
            rel_series = pd.Series(rel_names, name="relation")
            rel_counts = (
                rel_series.value_counts()
                .head(20)
                .reset_index()
                .rename(columns={"index": "relation", "count": "count"})
            )
            if "relation" not in rel_counts.columns:
                rel_counts.columns = ["relation", "count"]

            fig_rels = px.bar(
                rel_counts,
                x="count",
                y="relation",
                orientation="h",
                labels={"relation": "Relation Type", "count": "Edge Count"},
                color_discrete_sequence=[_BLUE],
                title="Top 20 Relation Types by Edge Count",
            )
            fig_rels.update_layout(
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA",
            )
            st.plotly_chart(fig_rels, use_container_width=True)
        else:
            st.info("Edge type data unavailable.")

    st.divider()
else:
    st.info(
        "📂 Graph data is not loaded. Use the **Load Data** section below to load "
        "processed graph files.",
        icon="ℹ️",
    )
    st.divider()

# ---------------------------------------------------------------------------
# Model status (only when model is loaded)
# ---------------------------------------------------------------------------
st.subheader("🤖 Model Status")
if status["model_loaded"]:
    meta: dict = st.session_state.get("_checkpoint_meta", {})
    epoch = meta.get("epoch", "—")
    val_mrr = meta.get("val_mrr")
    val_hits10 = meta.get("val_hits_at_10")
    ckpt_path = meta.get("checkpoint_path", "—")

    st.success(f"Model loaded from: `{ckpt_path}`", icon="✅")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Last Checkpoint Epoch", str(epoch))
    mc2.metric(
        "Val MRR",
        f"{val_mrr:.4f}" if val_mrr is not None else "—",
        help="Mean Reciprocal Rank on validation set. NOT a clinical metric.",
    )
    mc3.metric(
        "Val Hits@10",
        f"{val_hits10:.4f}" if val_hits10 is not None else "—",
        help="Hits@10 on validation set. Raw model output — not a probability.",
    )
    st.caption(
        f"⚠️ Metrics shown are raw model outputs on the validation split. "
        "They must NOT be interpreted as clinical accuracy or probabilities."
    )
else:
    st.info(
        "Model not loaded. Use the **Load Model** section to load a checkpoint.",
        icon="ℹ️",
    )

st.divider()

# ---------------------------------------------------------------------------
# Load Data section
# ---------------------------------------------------------------------------
st.subheader("📂 Load Data")

with st.expander("Load Graph Data", expanded=not status["data_loaded"]):
    nodes_path_display = cfg["paths"].get("nodes_csv", "data/processed/nodes.csv")
    edges_path_display = cfg["paths"].get("edges_csv", "data/processed/edges.csv")

    st.markdown(
        f"Will load from paths defined in `config.yaml`:\n"
        f"- `{nodes_path_display}`\n"
        f"- `{edges_path_display}`\n"
        f"- `{cfg['paths'].get('node_mapping_json', 'data/processed/node_mapping.json')}`\n"
        f"- `{cfg['paths'].get('relation_mapping_json', 'data/processed/relation_mapping.json')}`"
    )
    if st.button("🔄 Load Graph Data", type="primary"):
        with st.spinner("Loading graph data …"):
            success = load_project_data(cfg)
        if success:
            st.success("✅ Graph data loaded successfully!", icon="✅")
            log.info("[MediGraph] Dashboard: data loaded via button click.")
            st.rerun()
        else:
            st.error(
                "[MediGraph] Data loading failed. See warnings above.",
                icon="❌",
            )

with st.expander("Load Model Checkpoint", expanded=not status["model_loaded"]):
    checkpoints_dir = Path(cfg["paths"].get("checkpoints", "artifacts/checkpoints"))
    ckpt_files = sorted(checkpoints_dir.glob("*.pt")) if checkpoints_dir.exists() else []

    if not ckpt_files:
        st.info(
            "No checkpoint files found in "
            f"`{checkpoints_dir}`.  "
            "Train the model first with `python src/training/train.py`.",
            icon="ℹ️",
        )
    else:
        ckpt_options = [str(f) for f in ckpt_files]
        selected_ckpt = st.selectbox(
            "Select checkpoint",
            ckpt_options,
            index=len(ckpt_options) - 1,
            help="Choose the .pt checkpoint file to load.",
        )
        if st.button("🔄 Load Model", type="primary"):
            from app.components.data_loader import load_model_checkpoint

            with st.spinner("Loading model checkpoint …"):
                success = load_model_checkpoint(selected_ckpt, cfg)
            if success:
                st.success("✅ Model loaded successfully!", icon="✅")
                st.rerun()

st.divider()
st.caption(
    "MediGraph AI v{} · Research prototype · Not for clinical use.".format(
        cfg.get("project", {}).get("version", "0.1.0")
    )
)
