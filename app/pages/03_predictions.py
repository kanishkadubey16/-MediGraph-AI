"""
app/pages/03_predictions.py
---------------------------
Page 3: GNN Biomedical Link Prediction Studio.

Features:
- Live inference against trained 2-layer R-GCN + DistMult model checkpoint.
- Query specification: Head entity, typed relation, and optional target type filter.
- Quick query preset buttons (Alzheimer disease, Albuterol, Breast cancer, Asthma).
- Real-time scoring using DistMult bilinear decoder: score(h, r, t) = z_h ⊙ W_r ⊙ z_t.
- Visual comparison of existing dataset ground truth vs novel predicted hypotheses.
- Interactive hypothesis subgraph with solid lines for existing edges and dashed amber lines for novel links.
- Tabular ranked candidates with raw model affinity scores and CSV download.
- Full medical safety disclaimers with transparent explanation of DistMult bilinear logits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap sys.path and ensure package resolution
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
sys.path = [p for p in sys.path if Path(p).resolve() not in (_APP_DIR.resolve(), (_APP_DIR / "pages").resolve())]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    sys.modules["app"].__path__ = [str(_APP_DIR)]

import networkx as nx
import numpy as np
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
    render_breadcrumb,
)
from app.components.graph_visualization import (
    build_plotly_subgraph,
    ENTITY_COLORS,
)

try:
    st.set_page_config(
        page_title="AI Predictions — MediGraph AI",
        page_icon="🧬",
        layout="wide",
    )
except Exception:
    pass

# ---------------------------------------------------------------------------
# Initialise & Load Data
# ---------------------------------------------------------------------------
cfg = load_config()

render_breadcrumb("Machine Learning", "AI Predictions")

st.title("🧬 MediGraph AI — AI Predictions")
st.markdown(
    "<p style='color: #64748B; font-size: 1.05rem; margin-top: -10px;'>"
    "Infer potentially undiscovered medical relationships and candidate drug targets using 2-layer R-GCN + DistMult (ByTek style)."
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
# Quick Query Presets
# ---------------------------------------------------------------------------
st.markdown("##### 💡 Quick Exploration Presets")
preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

if "preset_head" not in st.session_state:
    st.session_state["preset_head"] = "Alzheimer disease"
if "preset_rel" not in st.session_state:
    st.session_state["preset_rel"] = "disease_protein"

with preset_col1:
    if st.button("🧠 Alzheimer ➔ disease_protein", use_container_width=True):
        st.session_state["preset_head"] = "Alzheimer disease"
        st.session_state["preset_rel"] = "disease_protein"
        st.session_state.pop("last_predictions", None)
with preset_col2:
    if st.button("💊 Albuterol ➔ drug_protein", use_container_width=True):
        st.session_state["preset_head"] = "Albuterol"
        st.session_state["preset_rel"] = "drug_protein"
        st.session_state.pop("last_predictions", None)
with preset_col3:
    if st.button("🎗️ Breast Cancer ➔ indication", use_container_width=True):
        st.session_state["preset_head"] = "breast cancer"
        st.session_state["preset_rel"] = "indication"
        st.session_state.pop("last_predictions", None)
with preset_col4:
    if st.button("🫁 Asthma ➔ phenotype", use_container_width=True):
        st.session_state["preset_head"] = "asthma"
        st.session_state["preset_rel"] = "disease_phenotype_positive"
        st.session_state.pop("last_predictions", None)

# ---------------------------------------------------------------------------
# 1. Prediction Query Configuration
# ---------------------------------------------------------------------------
render_section_header("Inference Query Setup", "Select query head entity, relation type, and candidate filter", "⚙️")

col_q1, col_q2, col_q3 = st.columns([2, 1.5, 1.5])

with col_q1:
    all_entity_types = ["All Types"] + sorted(node_df["entity_type"].unique().tolist())
    head_type_filter = st.selectbox("Head Entity Category Filter:", all_entity_types, index=0)

    if head_type_filter == "All Types":
        candidate_heads = node_df.sort_values("entity_name")
    else:
        candidate_heads = node_df[node_df["entity_type"] == head_type_filter].sort_values("entity_name")

    head_names = candidate_heads["entity_name"].tolist()
    pref_head = st.session_state.get("preset_head", "Alzheimer disease")
    default_head_idx = head_names.index(pref_head) if pref_head in head_names else 0

    query_head_entity = st.selectbox(
        "Source Entity (Head):",
        head_names,
        index=default_head_idx,
        help="Select the entity to evaluate for outgoing or incoming links.",
    )

with col_q2:
    relation_options = sorted(list(rel_mapping.keys()))
    pref_rel = st.session_state.get("preset_rel", "disease_protein")
    default_rel_idx = relation_options.index(pref_rel) if pref_rel in relation_options else 0
    query_relation = st.selectbox(
        "Target Relationship Type:",
        relation_options,
        index=default_rel_idx,
        help="The semantic biomedical relationship to evaluate.",
    )

with col_q3:
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
        value=-2.0,
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
                top_k=top_k * 2,
                score_threshold=score_threshold,
            )
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
            "description": "Ground-truth true positive facts",
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
    pred_cols = st.columns(4)
    for col, m in zip(pred_cols, pred_kpis):
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
    # 4. Top Candidate Spotlight Banner
    # ---------------------------------------------------------------------------
    top_candidate = preds[0]
    top_is_known = top_candidate["is_known_edge"]
    top_badge_color = "#0D9488" if top_is_known else "#F59E0B"
    top_badge_text = "Verified Ground Truth Fact" if top_is_known else "Novel GNN Hypothesis"
    top_prob = 1.0 / (1.0 + np.exp(-float(top_candidate["score"])))

    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 5px solid {top_badge_color};
            border-radius: 12px;
            padding: 16px 22px;
            margin: 16px 0 20px 0;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        ">
            <div>
                <span style="background: {top_badge_color}1A; color: {top_badge_color}; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; text-transform: uppercase;">
                    Top-Ranked Target #1
                </span>
                <div style="color: #0F172A; font-size: 1.45rem; font-weight: 800; margin-top: 4px;">
                    {query_info[0]} <span style="color: #64748B; font-weight: 400; font-size: 1.1rem;">➔ [{query_info[1]}] ➔</span> <strong style="color: #0F172A;">{top_candidate['target_name']}</strong>
                </div>
                <div style="color: #64748B; font-size: 0.82rem; margin-top: 2px;">
                    Target Type: <strong>{top_candidate['target_type'].capitalize()}</strong> • Status: <strong>{top_badge_text}</strong>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="color: #64748B; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">Affinity Logit / Sigmoid</div>
                <div style="color: #0F172A; font-size: 1.5rem; font-weight: 800;">{float(top_candidate['score']):.3f} <span style="font-size: 0.95rem; color: #0D9488;">({top_prob * 100:.1f}%)</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # 5. Interactive Prediction Subgraph
    # ---------------------------------------------------------------------------
    render_section_header("Hypothesis Subnetwork Visualization", "Solid slate lines indicate existing verified edges; dashed amber lines show novel predicted links", "🕸️")

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
        tgt_id = p.get("target_node_id", p.get("target_id"))
        tgt_source_id = p.get("target_source_id") or p.get("source_id")
        if not tgt_source_id and tgt_id in node_df["node_id"].values:
            tgt_source_id = str(node_df[node_df["node_id"] == tgt_id]["source_id"].iloc[0])

        G_pred.add_node(
            tgt_id,
            entity_name=p["target_name"],
            entity_type=p["target_type"],
            source_id=tgt_source_id or "N/A",
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
        height=560,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---------------------------------------------------------------------------
    # 6. Candidate Ranking Table
    # ---------------------------------------------------------------------------
    render_section_header("Ranked Candidate Table", "Rank-ordered candidates according to DistMult bilinear scoring", "📋")

    table_data = []
    for p in preds:
        score_val = float(p["score"])
        prob_val = 1.0 / (1.0 + np.exp(-score_val))
        status_badge = "✓ Ground Truth" if p["is_known_edge"] else "✨ Novel Hypothesis"
        tgt_id = p.get("target_node_id", p.get("target_id"))
        sid = p.get("target_source_id") or p.get("source_id")
        if not sid and tgt_id in node_df["node_id"].values:
            sid = str(node_df[node_df["node_id"] == tgt_id]["source_id"].iloc[0])
        table_data.append({
            "Rank": p["rank"],
            "Target Entity": p["target_name"],
            "Target Type": p["target_type"].capitalize(),
            "Raw Affinity (Logit)": round(score_val, 4),
            "Sigmoid Probability": f"{prob_val * 100:.1f}%",
            "Status": status_badge,
            "Source ID": sid or "N/A",
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
            "Raw Affinity (Logit)": st.column_config.NumberColumn(format="%.4f"),
            "Sigmoid Probability": st.column_config.TextColumn(width="small"),
            "Status": st.column_config.TextColumn(width="medium"),
        },
    )

    st.info(
        f"ℹ️ **Score Interpretation:** {DISCLAIMER_TEXT} "
        "Scores are computed via the DistMult bilinear decoder `<e_head, r_rel, e_tail>`. "
        "Higher values denote stronger graph embedding association.",
        icon="ℹ️",
    )

    csv_bytes = df_preds_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_bytes,
        file_name=f"predictions_{query_info[0].replace(' ', '_')}_{query_info[1]}.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Select a preset above or configure custom parameters, then click **'Run GNN Inference'**.")
