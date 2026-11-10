"""
app/pages/01_dashboard.py
-------------------------
Page 1: MediGraph AI Executive Dashboard.

Displays:
- Summary KPI Cards (Total Nodes, Edges, Diseases, Drugs, Genes/Proteins, Phenotypes, Relation Categories)
- Entity Type Distribution (Plotly Donut Chart)
- Relationship Type Frequency Breakdown (Plotly Horizontal Bar Chart)
- Graph Topological Properties (Density, Average Degree, Connected Components)
- Real-time dataset health & schema integrity indicators

Data is dynamically loaded from processed datasets (data/processed/).
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import load_config
from app.components.data_loader import load_project_data
from app.components.metric_cards import (
    render_metric_card,
    render_kpi_row,
    render_section_header,
    render_disclaimer_banner,
)
from app.components.graph_visualization import ENTITY_COLORS

st.set_page_config(
    page_title="Dashboard — MediGraph AI",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialise & Load Data
# ---------------------------------------------------------------------------
cfg = load_config()

st.title("📊 Biomedical Knowledge Graph Dashboard")
st.markdown(
    "<p style='color: #64748B; font-size: 1.05rem; margin-top: -10px;'>"
    "Real-time topological metrics and entity distribution of the PrimeKG medical graph partition."
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

# Calculate exact actual counts
total_nodes = len(node_df)
total_edges = edge_index.shape[1]
total_rel_types = len(rel_mapping)

type_counts = node_df["entity_type"].value_counts().to_dict()
total_diseases = type_counts.get("disease", 0)
total_drugs = type_counts.get("drug", 0)
total_genes = type_counts.get("gene", 0) + type_counts.get("protein", 0)
total_phenotypes = type_counts.get("phenotype", 0)

# ---------------------------------------------------------------------------
# 1. Primary KPI Cards
# ---------------------------------------------------------------------------
render_section_header("Knowledge Graph Core Inventory", "Actual entity counts from processed PrimeKG partition", "🧬")

row1_metrics = [
    {
        "title": "Total Entities (Nodes)",
        "value": f"{total_nodes:,}",
        "description": "Unique medical entities",
        "icon": "🌐",
        "badge": "Active",
        "color_accent": "navy",
    },
    {
        "title": "Total Relationships (Edges)",
        "value": f"{total_edges:,}",
        "description": "Validated biomedical facts",
        "icon": "🔗",
        "badge": "Directed",
        "color_accent": "teal",
    },
    {
        "title": "Relation Categories",
        "value": f"{total_rel_types}",
        "description": "Typed semantic relations",
        "icon": "🏷️",
        "badge": "R-GCN Relations",
        "color_accent": "purple",
    },
]
render_kpi_row(row1_metrics)

row2_metrics = [
    {
        "title": "Diseases",
        "value": f"{total_diseases:,}",
        "description": "MONDO / Orphanet ontologies",
        "icon": "🦠",
        "color_accent": "coral",
    },
    {
        "title": "Drugs & Compounds",
        "value": f"{total_drugs:,}",
        "description": "DrugBank approved agents",
        "icon": "💊",
        "color_accent": "navy",
    },
    {
        "title": "Genes & Proteins",
        "value": f"{total_genes:,}",
        "description": "NCBI Gene / UniProt targets",
        "icon": "🧬",
        "color_accent": "teal",
    },
    {
        "title": "Symptoms & Phenotypes",
        "value": f"{total_phenotypes:,}",
        "description": "HPO clinical observations",
        "icon": "🩺",
        "color_accent": "amber",
    },
]
render_kpi_row(row2_metrics)

st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Charts: Distributions
# ---------------------------------------------------------------------------
render_section_header("Entity & Relationship Distributions", "Empirical breakdown of graph categories", "📈")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Entity Type Distribution")
    df_entity_types = node_df["entity_type"].value_counts().reset_index()
    df_entity_types.columns = ["Entity Type", "Count"]

    color_seq = [
        ENTITY_COLORS.get(t.lower(), ENTITY_COLORS["other"])
        for t in df_entity_types["Entity Type"]
    ]

    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=df_entity_types["Entity Type"].str.capitalize(),
                values=df_entity_types["Count"],
                hole=0.55,
                marker=dict(colors=color_seq, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=20, b=30, l=10, r=10),
        height=340,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_right:
    st.markdown("#### Relationship Type Breakdown")
    # Read relation counts from edges.csv
    edges_path = cfg["paths"]["edges_csv"]
    df_edges = pd.read_csv(edges_path)
    df_rel_counts = df_edges["relation"].value_counts().reset_index()
    df_rel_counts.columns = ["Relationship", "Frequency"]
    df_rel_counts = df_rel_counts.sort_values("Frequency", ascending=True)

    fig_bar = go.Figure(
        data=[
            go.Bar(
                x=df_rel_counts["Frequency"],
                y=df_rel_counts["Relationship"],
                orientation="h",
                marker=dict(color="#0D9488", line=dict(color="#115E59", width=1)),
                text=df_rel_counts["Frequency"],
                textposition="auto",
            )
        ]
    )
    fig_bar.update_layout(
        xaxis_title="Edge Count",
        yaxis_title="",
        margin=dict(t=20, b=40, l=10, r=10),
        height=340,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Graph Topological Statistics
# ---------------------------------------------------------------------------
render_section_header("Network Topology & Connectivity Metrics", "Fundamental structural properties of the directed graph", "🕸️")

# Build NetworkX DiGraph for actual graph-theoretic metrics
G = nx.DiGraph()
for _, row in node_df.iterrows():
    G.add_node(int(row["node_id"]), entity_name=row["entity_name"], entity_type=row["entity_type"])
for _, row in df_edges.iterrows():
    G.add_edge(int(row["source"]), int(row["target"]), relation=row["relation"])

density = nx.density(G)
avg_in_degree = sum(dict(G.in_degree()).values()) / max(1, total_nodes)
avg_out_degree = sum(dict(G.out_degree()).values()) / max(1, total_nodes)
isolated_nodes = len(list(nx.isolates(G)))
weakly_connected = nx.number_weakly_connected_components(G)

topo_cols = st.columns(4)
with topo_cols[0]:
    render_metric_card("Graph Density", f"{density:.4f}", "Fraction of possible edges", icon="📐", color_accent="slate")
with topo_cols[1]:
    render_metric_card("Avg Node In-Degree", f"{avg_in_degree:.2f}", "Incoming connections / node", icon="📥", color_accent="navy")
with topo_cols[2]:
    render_metric_card("Avg Node Out-Degree", f"{avg_out_degree:.2f}", "Outgoing connections / node", icon="📤", color_accent="teal")
with topo_cols[3]:
    render_metric_card("Weakly Connected Components", f"{weakly_connected}", "Disconnected graph clusters", icon="🧩", color_accent="amber")

# Top Hub Entities Table
st.markdown("#### Top Connected Hub Entities")
node_degrees = []
for n, d in G.degree():
    data = G.nodes[n]
    node_degrees.append({
        "Node ID": n,
        "Entity Name": data.get("entity_name", str(n)),
        "Entity Type": str(data.get("entity_type", "")).capitalize(),
        "In-Degree": G.in_degree(n),
        "Out-Degree": G.out_degree(n),
        "Total Degree": d,
    })

df_hubs = pd.DataFrame(node_degrees).sort_values("Total Degree", ascending=False).head(8)
st.dataframe(
    df_hubs,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Node ID": st.column_config.NumberColumn(width="small"),
        "Entity Name": st.column_config.TextColumn(width="medium"),
        "Entity Type": st.column_config.TextColumn(width="small"),
        "Total Degree": st.column_config.ProgressColumn(
            "Connection Degree",
            format="%d",
            min_value=0,
            max_value=int(df_hubs["Total Degree"].max()),
        ),
    },
)
