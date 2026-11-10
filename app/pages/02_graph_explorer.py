"""
app/pages/02_graph_explorer.py
------------------------------
Page 2: Knowledge Graph Explorer.

Features:
- Entity search with autocomplete and filter by entity type (disease, drug, gene, etc.).
- Entity detail card (name, type, ontology source ID, in/out degrees).
- Interactive 1-hop and 2-hop local neighborhood visualizer using Plotly and PyVis.
- Nodes colored by biomedical type with custom hover attributes.
- Edge relationship labels with direction indicators.
- Filter neighborhood by relation type or entity category.
- Tabular relationship explorer with CSV export.
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

from src.utils.config import load_config
from app.components.data_loader import load_project_data
from app.components.metric_cards import (
    render_metric_card,
    render_section_header,
    render_disclaimer_banner,
)
from app.components.graph_visualization import (
    build_plotly_subgraph,
    render_pyvis_network,
    ENTITY_COLORS,
)

st.set_page_config(
    page_title="Graph Explorer — MediGraph AI",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialise & Load Data
# ---------------------------------------------------------------------------
cfg = load_config()

st.title("🔍 Biomedical Knowledge Graph Explorer")
st.markdown(
    "<p style='color: #64748B; font-size: 1.05rem; margin-top: -10px;'>"
    "Search any biomedical entity and inspect its multi-hop validated relational neighborhood."
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
rel_mapping: dict[str, int] = st.session_state.rel_mapping
rel_mapping_inv = {v: k for k, v in rel_mapping.items()}

# Build global NetworkX graph for neighborhood extraction
@st.cache_data
def build_full_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in node_df.iterrows():
        nid = int(row["node_id"])
        G.add_node(
            nid,
            entity_name=str(row["entity_name"]),
            entity_type=str(row["entity_type"]),
            source_id=str(row.get("source_id", "N/A")),
        )
    srcs = edge_index[0].tolist()
    tgts = edge_index[1].tolist()
    rels = edge_type.tolist()
    for s, t, r in zip(srcs, tgts, rels):
        rel_str = rel_mapping_inv.get(int(r), f"rel_{r}")
        G.add_edge(int(s), int(t), relation=rel_str, relation_id=int(r))
    return G

G_full = build_full_graph()

# ---------------------------------------------------------------------------
# 1. Search & Selection Panel
# ---------------------------------------------------------------------------
render_section_header("Entity Search & Neighborhood Parameters", "Select an entity to explore its localized network", "🔎")

col_search1, col_search2, col_search3 = st.columns([2, 1.5, 1.5])

with col_search1:
    # Optional entity type filter
    all_types = ["All Types"] + sorted(node_df["entity_type"].unique().tolist())
    selected_type_filter = st.selectbox("Filter Entity Type:", all_types, index=0)

    if selected_type_filter == "All Types":
        candidate_entities = node_df.sort_values("entity_name")
    else:
        candidate_entities = node_df[node_df["entity_type"] == selected_type_filter].sort_values("entity_name")

    entity_names = candidate_entities["entity_name"].tolist()
    default_idx = 0
    # Try to default to "Alzheimer disease" if present
    if "Alzheimer disease" in entity_names:
        default_idx = entity_names.index("Alzheimer disease")

    selected_entity_name = st.selectbox(
        "Search or Select Entity:",
        entity_names,
        index=default_idx,
        help="Type or select any disease, drug, gene, phenotype, or pathway in the knowledge graph.",
    )

with col_search2:
    hop_depth = st.radio("Neighborhood Depth:", [1, 2], index=0, horizontal=True, help="1-hop: immediate neighbors; 2-hop: neighbors of neighbors.")
    renderer_mode = st.radio("Visualizer Engine:", ["Plotly (2D)", "PyVis (Physics)"], index=0, horizontal=True)

with col_search3:
    max_neighbors = st.slider("Max Neighbors to Display:", min_value=5, max_value=60, value=25, step=5)

# Resolve selected node ID and metadata
node_record = node_df[node_df["entity_name"] == selected_entity_name].iloc[0]
center_node_id = int(node_record["node_id"])
center_entity_type = str(node_record["entity_type"])
center_source_id = str(node_record.get("source_id", "N/A"))

# ---------------------------------------------------------------------------
# 2. Selected Entity Profile Cards
# ---------------------------------------------------------------------------
in_deg = G_full.in_degree(center_node_id)
out_deg = G_full.out_degree(center_node_id)
total_deg = in_deg + out_deg

profile_cols = st.columns(4)
with profile_cols[0]:
    render_metric_card("Selected Entity", selected_entity_name, f"Node ID: {center_node_id}", icon="🎯", color_accent="teal")
with profile_cols[1]:
    render_metric_card("Entity Type", center_entity_type.capitalize(), f"Source: {center_source_id}", icon="🏷️", color_accent="navy")
with profile_cols[2]:
    render_metric_card("Incoming Edges", in_deg, "Target of relationships", icon="📥", color_accent="coral")
with profile_cols[3]:
    render_metric_card("Outgoing Edges", out_deg, "Source of relationships", icon="📤", color_accent="amber")

# ---------------------------------------------------------------------------
# 3. Extract Subgraph
# ---------------------------------------------------------------------------
if hop_depth == 1:
    neighbors = set(G_full.successors(center_node_id)).union(set(G_full.predecessors(center_node_id)))
    sub_nodes = list(neighbors)[:max_neighbors] + [center_node_id]
    G_sub = G_full.subgraph(sub_nodes).copy()
else:
    # 2-hop ego graph
    G_sub = nx.ego_graph(G_full, center_node_id, radius=2)
    if G_sub.number_of_nodes() > max_neighbors + 1:
        # Prune to max neighbors while keeping center node
        sorted_nodes = sorted(
            [n for n in G_sub.nodes() if n != center_node_id],
            key=lambda n: G_sub.degree(n),
            reverse=True,
        )[:max_neighbors]
        G_sub = G_full.subgraph(sorted_nodes + [center_node_id]).copy()

# ---------------------------------------------------------------------------
# 4. Interactive Graph Visualization
# ---------------------------------------------------------------------------
render_section_header(
    f"Subnetwork for '{selected_entity_name}' ({G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges)",
    "Color-coded by biomedical entity category; solid links represent verified dataset ground truth",
    "🕸️",
)

if renderer_mode == "Plotly (2D)":
    fig = build_plotly_subgraph(
        G=G_sub,
        center_node_id=center_node_id,
        title=f"Neighborhood Graph: {selected_entity_name} (depth={hop_depth})",
        height=580,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    render_pyvis_network(G_sub, center_node_id=center_node_id, height_px=580)

# ---------------------------------------------------------------------------
# 5. Connected Relationships Table
# ---------------------------------------------------------------------------
render_section_header("Direct Relational Table", "Tabular view of all incoming and outgoing connections", "📋")

edge_rows = []
for u, v, data in G_sub.edges(data=True):
    u_name = G_sub.nodes[u].get("entity_name", str(u))
    u_type = str(G_sub.nodes[u].get("entity_type", "")).capitalize()
    v_name = G_sub.nodes[v].get("entity_name", str(v))
    v_type = str(G_sub.nodes[v].get("entity_type", "")).capitalize()
    rel = data.get("relation", "")

    direction = "Outgoing (Center → Target)" if u == center_node_id else "Incoming (Source → Center)" if v == center_node_id else "Neighborhood Edge"

    edge_rows.append({
        "Source Entity": u_name,
        "Source Type": u_type,
        "Relationship": rel,
        "Target Entity": v_name,
        "Target Type": v_type,
        "Direction": direction,
    })

if edge_rows:
    df_sub_edges = pd.DataFrame(edge_rows)
    st.dataframe(
        df_sub_edges,
        use_container_width=True,
        hide_index=True,
    )

    # Download CSV
    csv_data = df_sub_edges.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Neighborhood Connections (CSV)",
        data=csv_data,
        file_name=f"neighborhood_{selected_entity_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )
else:
    st.info("No connections found for this entity within the selected filter.")
