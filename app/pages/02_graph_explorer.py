"""
app/pages/02_graph_explorer.py
------------------------------
Page 2: Knowledge Graph Explorer.

Features:
- Entity search with autocomplete and filter by entity type (disease, drug, gene, etc.).
- Entity detail spotlight card (name, type, ontology source ID, in/out degrees).
- Interactive 1-hop and 2-hop local neighborhood visualizer using Plotly and PyVis.
- Nodes colored by biomedical type with custom hover attributes.
- Edge relationship labels with direction indicators.
- Filter neighborhood by relation type or entity category.
- Tabular relationship explorer with CSV export.
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
import pandas as pd
import streamlit as st

from src.utils.config import load_config
from app.components.data_loader import load_project_data
from app.components.metric_cards import (
    render_metric_card,
    render_section_header,
    render_disclaimer_banner,
    render_breadcrumb,
)
from app.components.graph_visualization import (
    build_plotly_subgraph,
    render_pyvis_network,
    ENTITY_COLORS,
)

try:
    st.set_page_config(
        page_title="Knowledge Graph Explorer — MediGraph AI",
        page_icon="🔍",
        layout="wide",
    )
except Exception:
    pass

# ---------------------------------------------------------------------------
# Initialise & Load Data
# ---------------------------------------------------------------------------
cfg = load_config()

render_breadcrumb("Analytics", "Knowledge Graph Explorer")

st.title("🔍 MediGraph AI — Knowledge Graph Explorer")
st.markdown(
    "<p style='color: #64748B; font-size: 1.05rem; margin-top: -10px;'>"
    "Search any biomedical entity and interactively explore its multi-hop validated relational neighborhood (BioBox style)."
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
    all_types = ["All Types"] + sorted(node_df["entity_type"].unique().tolist())
    selected_type_filter = st.selectbox("Filter Entity Category:", all_types, index=0)

    if selected_type_filter == "All Types":
        candidate_entities = node_df.sort_values("entity_name")
    else:
        candidate_entities = node_df[node_df["entity_type"] == selected_type_filter].sort_values("entity_name")

    entity_names = candidate_entities["entity_name"].tolist()
    default_idx = 0
    if "Alzheimer disease" in entity_names:
        default_idx = entity_names.index("Alzheimer disease")

    selected_entity_name = st.selectbox(
        "Search or Select Entity:",
        entity_names,
        index=default_idx,
        help="Type or select any disease, drug, gene, phenotype, or pathway in the knowledge graph.",
    )

with col_search2:
    hop_depth = st.radio("Relational Hop Depth:", [1, 2], index=0, horizontal=True, help="1-hop: immediate neighbors; 2-hop: multi-hop path connections.")
    renderer_mode = st.radio("Visualizer Engine:", ["Plotly (2D)", "PyVis (Physics)"], index=0, horizontal=True)

with col_search3:
    max_neighbors = st.slider("Max Neighbors to Display:", min_value=5, max_value=60, value=25, step=5)

# Resolve selected node ID and metadata
node_record = node_df[node_df["entity_name"] == selected_entity_name].iloc[0]
center_node_id = int(node_record["node_id"])
center_type = str(node_record["entity_type"])
center_source_id = str(node_record.get("source_id", "N/A"))

# ---------------------------------------------------------------------------
# 2. Entity Spotlight Header Card
# ---------------------------------------------------------------------------
type_color = ENTITY_COLORS.get(center_type.lower(), "#0D9488")
in_deg = G_full.in_degree(center_node_id)
out_deg = G_full.out_degree(center_node_id)
total_deg = in_deg + out_deg

st.markdown(
    f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid {type_color};
        border-radius: 12px;
        padding: 18px 24px;
        margin: 16px 0 20px 0;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    ">
        <div>
            <div style="color: #64748B; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                Center Query Focus
            </div>
            <div style="color: #0F172A; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em;">
                {selected_entity_name}
            </div>
            <div style="margin-top: 6px; display: flex; gap: 8px; align-items: center;">
                <span style="background: {type_color}1A; color: {type_color}; border: 1px solid {type_color}44; font-size: 0.75rem; font-weight: 700; padding: 2px 10px; border-radius: 9999px; text-transform: uppercase;">
                    {center_type}
                </span>
                <span style="color: #64748B; font-size: 0.82rem;">
                    Ontology ID: <strong>{center_source_id}</strong>
                </span>
            </div>
        </div>
        <div style="display: flex; gap: 20px; align-items: center;">
            <div style="text-align: center;">
                <div style="color: #64748B; font-size: 0.70rem; font-weight: 700; text-transform: uppercase;">Total Degree</div>
                <div style="color: #0F172A; font-size: 1.4rem; font-weight: 800;">{total_deg}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #64748B; font-size: 0.70rem; font-weight: 700; text-transform: uppercase;">Incoming</div>
                <div style="color: #3B82F6; font-size: 1.4rem; font-weight: 800;">{in_deg}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #64748B; font-size: 0.70rem; font-weight: 700; text-transform: uppercase;">Outgoing</div>
                <div style="color: #0D9488; font-size: 1.4rem; font-weight: 800;">{out_deg}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 3. Extract Subgraph Neighborhood
# ---------------------------------------------------------------------------
# Extract nodes within k hops
visited_nodes = {center_node_id}
current_frontier = {center_node_id}

for _ in range(hop_depth):
    next_frontier = set()
    for n in current_frontier:
        # Undirected traversal for neighborhood context
        succs = set(G_full.successors(n))
        preds = set(G_full.predecessors(n))
        next_frontier.update(succs | preds)
    current_frontier = next_frontier - visited_nodes
    visited_nodes.update(next_frontier)
    if len(visited_nodes) >= max_neighbors:
        break

subgraph_nodes = list(visited_nodes)[:max_neighbors]
if center_node_id not in subgraph_nodes:
    subgraph_nodes.append(center_node_id)

G_sub = G_full.subgraph(subgraph_nodes).copy()

# ---------------------------------------------------------------------------
# 4. Render Graph Visualization
# ---------------------------------------------------------------------------
render_section_header(
    f"Relational Subgraph: {selected_entity_name}",
    f"{G_sub.number_of_nodes()} nodes and {G_sub.number_of_edges()} verified relationships ({hop_depth}-hop depth)",
    "🕸️",
)

# Entity color legend
legend_html = " ".join([
    f'<span style="background:{c}22; color:{c}; border:1px solid {c}55; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; margin-right:6px; text-transform:capitalize;">● {t}</span>'
    for t, c in ENTITY_COLORS.items() if t not in ("other", "protein")
])
st.markdown(f"<div style='margin-bottom: 12px;'><strong>Entity Legend:</strong> {legend_html}</div>", unsafe_allow_html=True)

if renderer_mode == "Plotly (2D)":
    fig = build_plotly_subgraph(
        G=G_sub,
        center_node_id=center_node_id,
        title=f"Local Neighborhood ({hop_depth}-Hop) around '{selected_entity_name}'",
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    render_pyvis_network(
        G=G_sub,
        center_node_id=center_node_id,
        height="620px",
    )

# ---------------------------------------------------------------------------
# 5. Tabular Relationship Explorer
# ---------------------------------------------------------------------------
render_section_header("Validated Neighborhood Relationships Table", "Detailed edge records for the selected localized graph", "📋")

edge_rows = []
for u, v, data in G_sub.edges(data=True):
    u_name = G_sub.nodes[u].get("entity_name", str(u))
    u_type = G_sub.nodes[u].get("entity_type", "unknown")
    v_name = G_sub.nodes[v].get("entity_name", str(v))
    v_type = G_sub.nodes[v].get("entity_type", "unknown")
    rel = data.get("relation", "connected_to")
    edge_rows.append({
        "Source Entity": u_name,
        "Source Type": u_type.capitalize(),
        "Relationship": rel,
        "Target Entity": v_name,
        "Target Type": v_type.capitalize(),
    })

df_sub_edges = pd.DataFrame(edge_rows)

if not df_sub_edges.empty:
    col_tab1, col_tab2 = st.columns([4, 1])
    with col_tab1:
        st.dataframe(df_sub_edges, use_container_width=True, hide_index=True)
    with col_tab2:
        csv_bytes = df_sub_edges.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Edges (CSV)",
            data=csv_bytes,
            file_name=f"subgraph_{selected_entity_name.replace(' ', '_')}.csv",
            mime="text/csv",
        )
else:
    st.info("No relationships found for this entity within the selected parameters.")
