"""
app/pages/03_predictions.py
---------------------------
Page 3: Relational Graph Convolutional Network (R-GCN) Link Prediction Studio.

Features:
- Select query source entity and target relationship type.
- Optional candidate target entity type filter (e.g. only predict gene/protein associations).
- Top-K candidate slider and minimum score threshold slider.
- Executes real GNN inference via `src.inference.predict.predict_links`.
- Visual differentiation:
  * "✓ Known Edge" (validated relationship present in dataset)
  * "✨ Novel Hypothesis" (GNN-predicted candidate link)
- Subgraph visualization showing query node, existing neighbors, and predicted edges.
- Tabular ranked candidates with raw model affinity scores and CSV download.
- Full medical safety disclaimers with transparent explanation of DistMult bilinear logits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx
import pandas as pd
import streamlit as st
import torch

from src.utils.config import load_config
from src.models.decoder import MediGraphModel
from src.inference.predict import predict_links, DISCLAIMER_TEXT
from app.components.data_loader import load_project_data
from app.components.metric_cards import (
    render_metric_card,
    render_section_header,
    render_disclaimer_banner,
)
from app.components.graph_visualization import (
    build_plotly_subgraph,
    ENTITY_COLORS,
)

st.set_page_config(
    page_title="GNN Prediction — MediGraph AI",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialise & Load Data
# ---------------------------------------------------------------------------
cfg = load_config()

st.title("🧬 GNN Biomedical Link Prediction Studio")
st.markdown(
    "<p style='color: #64748B; font-size: 1.05rem; margin-top: -10px;'>"
    "Infer potentially undiscovered medical relationships and drug targets using 2-layer R-GCN + DistMult."
    "</p>",
    unsafe_allow_html=True,
)

render_disclaimer_banner()

data_loaded = load_project_data(cfg)
if not data_loaded:
    st.error("Please run the preprocessing pipeline first: `python src/data/preprocessing.py`")
    st.stop()

node_df: pd.DataFrame = st.session_state.node_df
edge_index = st.session_state.edge_index
edge_type = st.session_state.edge_type
node_mapping: dict[str, int] = st.session_state.node_mapping
rel_mapping: dict[str, int] = st.session_state.rel_mapping
rel_mapping_inv = {v: k for k, v in rel_mapping.items()}

# ---------------------------------------------------------------------------
# Load Model Checkpoint with Caching
# ---------------------------------------------------------------------------
@st.cache_resource
def get_trained_model(ckpt_path_str: str) -> tuple[Optional[MediGraphModel], Optional[str]]:
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        return None, f"Checkpoint not found at '{ckpt_path_str}'. Run `python src/training/train.py`."
    try:
        model, meta = MediGraphModel.load_checkpoint(str(ckpt_path), map_location="cpu")
        model.eval()
        return model, None
    except Exception as exc:
        return None, f"Failed to load checkpoint: {exc}"

ckpt_file = cfg.get("paths", {}).get("checkpoints", "artifacts/checkpoints") + "/best_model.pt"
model, load_err = get_trained_model(ckpt_file)

if model is None:
    st.error(
        f"❌ **Trained Model Checkpoint Unavailable**\n\n"
        f"Explanation: {load_err}\n\n"
        "To generate predictions, train the R-GCN model by executing:\n"
        "```bash\npython src/training/train.py --epochs 50 --device cpu\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# 1. Prediction Query Configuration
# ---------------------------------------------------------------------------
render_section_header("Inference Query Setup", "Select query head entity, relation type, and candidate filter", "⚙️")

col_q1, col_q2, col_q3 = st.columns([2, 1.5, 1.5])

with col_q1:
    # Filter source entities by type
    all_entity_types = ["All Types"] + sorted(node_df["entity_type"].unique().tolist())
    head_type_filter = st.selectbox("Head Entity Category:", all_entity_types, index=0)

    if head_type_filter == "All Types":
        candidate_heads = node_df.sort_values("entity_name")
    else:
        candidate_heads = node_df[node_df["entity_type"] == head_type_filter].sort_values("entity_name")

    head_names = candidate_heads["entity_name"].tolist()
    default_head_idx = head_names.index("Alzheimer disease") if "Alzheimer disease" in head_names else 0

    query_head_entity = st.selectbox(
        "Source Entity (Head):",
        head_names,
        index=default_head_idx,
        help="Select the entity to evaluate for outgoing or incoming links.",
    )

with col_q2:
    relation_options = sorted(list(rel_mapping.keys()))
    default_rel_idx = relation_options.index("disease_protein") if "disease_protein" in relation_options else 0
    query_relation = st.selectbox(
        "Target Relationship Type:",
        relation_options,
        index=default_rel_idx,
        help="The semantic biomedical relationship to evaluate.",
    )

with col_q3:
    # Candidate target type filter
    all_target_types = ["All Target Types"] + sorted(node_df["entity_type"].unique().tolist())
    target_type_filter = st.selectbox(
        "Candidate Target Type Filter:",
        all_target_types,
        index=0,
        help="Optionally restrict candidate ranking to a specific biomedical entity class.",
    )

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 1])

with col_ctrl1:
    top_k = st.slider("Top-K Candidates to Rank:", min_value=5, max_value=35, value=12, step=1)

with col_ctrl2:
    score_threshold = st.slider(
        "Score Threshold (Logit Filter):",
        min_value=-5.0,
        max_value=15.0,
        value=0.0,
        step=0.5,
        help="Only display candidates with model logit score >= threshold.",
    )

with col_ctrl3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run GNN Inference", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# 2. Execute Prediction
# ---------------------------------------------------------------------------
if run_btn or "last_predictions" in st.session_state:
    with st.spinner(f"Computing DistMult candidate affinity scores for ('{query_head_entity}', '{query_relation}', ?)..."):
        try:
            preds = predict_links(
                query_node=query_head_entity,
                relation_type=query_relation,
                model=model,
                node_df=node_df,
                node_mapping=node_mapping,
                rel_mapping=rel_mapping,
                edge_index=edge_index,
                edge_type=edge_type,
                top_k=top_k * 2,  # Fetch more to allow type filtering
                score_threshold=score_threshold,
            )
            # Apply target type filter if set
            if target_type_filter != "All Target Types":
                preds = [p for p in preds if p["target_type"].lower() == target_type_filter.lower()]
            preds = preds[:top_k]
            st.session_state["last_predictions"] = preds
            st.session_state["last_query"] = (query_head_entity, query_relation)
        except Exception as exc:
            st.error(f"Inference execution failed: {exc}")
            st.stop()

    preds = st.session_state.get("last_predictions", [])
    query_info = st.session_state.get("last_query", (query_head_entity, query_relation))

    if not preds:
        st.warning(f"No candidate entities met the score threshold ({score_threshold}) for the selected filter.")
        st.stop()

    # ---------------------------------------------------------------------------
    # 3. Prediction KPI Summary Cards
    # ---------------------------------------------------------------------------
    known_count = sum(1 for p in preds if p["is_known_edge"])
    novel_count = len(preds) - known_count
    max_score = max(p["score"] for p in preds) if preds else 0.0

    render_section_header("Inference Results & Candidate Breakdown", f"Evaluated query: ({query_info[0]}, {query_info[1]}, ?)", "🎯")

    pred_kpis = [
        {
            "title": "Top Candidates Evaluated",
            "value": len(preds),
            "description": f"Top-{top_k} ranked answers",
            "icon": "📊",
            "color_accent": "navy",
        },
        {
            "title": "Known Dataset Edges",
            "value": known_count,
            "description": "Ground-truth true positives",
            "icon": "✓",
            "badge": "Ground Truth",
            "color_accent": "teal",
        },
        {
            "title": "Novel Hypotheses",
            "value": novel_count,
            "description": "Model-predicted novel links",
            "icon": "✨",
            "badge": "Hypothesis",
            "color_accent": "amber",
        },
        {
            "title": "Peak Affinity Score",
            "value": f"{max_score:.2f}",
            "description": "Highest DistMult logit",
            "icon": "📈",
            "color_accent": "purple",
        },
    ]
    st.columns(4)
    for col, m in zip(st.columns(4), pred_kpis):
        with col:
            render_metric_card(
                title=m["title"],
                value=m["value"],
                description=m["description"],
                icon=m["icon"],
                badge=m.get("badge"),
                color_accent=m["color_accent"],
            )

    # ---------------------------------------------------------------------------
    # 4. Interactive Prediction Subgraph
    # ---------------------------------------------------------------------------
    render_section_header("Hypothesis Subnetwork Visualization", "Solid lines indicate existing edges; dashed amber lines show novel predicted links", "🕸️")

    # Build local graph for visualization: Center Node + Neighbors + Top Predicted Edges
    center_id = node_mapping[query_info[0]]
    G_pred = nx.DiGraph()
    G_pred.add_node(
        center_id,
        entity_name=query_info[0],
        entity_type=str(node_df[node_df["node_id"] == center_id]["entity_type"].iloc[0]),
        source_id=str(node_df[node_df["node_id"] == center_id]["source_id"].iloc[0]),
    )

    predicted_edge_tuples = []
    for p in preds:
        tgt_id = p["target_id"]
        G_pred.add_node(
            tgt_id,
            entity_name=p["target_name"],
            entity_type=p["target_type"],
            source_id=p.get("target_source_id", "N/A"),
        )
        if p["is_known_edge"]:
            G_pred.add_edge(center_id, tgt_id, relation=query_info[1])
        else:
            predicted_edge_tuples.append((center_id, tgt_id, query_info[1]))

    fig_pred = build_plotly_subgraph(
        G=G_pred,
        center_node_id=center_id,
        title=f"Predicted Relational Subgraph: ('{query_info[0]}', '{query_info[1]}', ?)",
        predicted_edges=predicted_edge_tuples,
        height=540,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---------------------------------------------------------------------------
    # 5. Candidate Ranking Table
    # ---------------------------------------------------------------------------
    render_section_header("Ranked Candidate Table", "Rank-ordered candidates according to DistMult bilinear scoring", "📋")

    table_data = []
    for p in preds:
        status_badge = "✓ Known Ground Truth" if p["is_known_edge"] else "✨ Novel Predicted Hypothesis"
        table_data.append({
            "Rank": p["rank"],
            "Target Entity": p["target_name"],
            "Target Type": p["target_type"].capitalize(),
            "Raw Affinity Score (Logit)": round(float(p["score"]), 4),
            "Status": status_badge,
            "Source ID": p.get("target_source_id", "N/A"),
        })

    df_preds_table = pd.DataFrame(table_data)

    st.dataframe(
        df_preds_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Target Entity": st.column_config.TextColumn(width="medium"),
            "Target Type": st.column_config.TextColumn(width="small"),
            "Raw Affinity Score (Logit)": st.column_config.NumberColumn(format="%.4f"),
            "Status": st.column_config.TextColumn(width="medium"),
        },
    )

    st.info(
        f"ℹ️ **Score Interpretation:** {DISCLAIMER_TEXT} "
        "Scores are computed via the DistMult bilinear decoder `<e_head, r_rel, e_tail>`. "
        "Higher values denote stronger graph embedding association.",
        icon="ℹ️",
    )

    # Download CSV button
    csv_bytes = df_preds_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_bytes,
        file_name=f"predictions_{query_info[0].replace(' ', '_')}_{query_info[1]}.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Configure query parameters above and click **'Run GNN Inference'** to predict candidate relationships.")
