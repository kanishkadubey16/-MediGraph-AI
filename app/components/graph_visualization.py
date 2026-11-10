"""
app/components/graph_visualization.py
-------------------------------------
Reusable, interactive graph visualizers for MediGraph AI using Plotly and PyVis.

Key Features:
- Visual differentiation of node types (disease, drug, gene, phenotype, pathway).
- Visual distinction between existing dataset ground truth (solid teal/slate edges)
  and model-predicted hypotheses (dashed coral/gold edges).
- Interactive 2D NetworkX + Plotly layout with zoom, pan, hover tooltips, and legend.
- Interactive PyVis physics-enabled network explorer embeddable in Streamlit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Standard entity color palette
ENTITY_COLORS = {
    "disease":             "#EF4444",   # Coral Red
    "drug":                "#3B82F6",   # Vibrant Blue
    "gene":                "#10B981",   # Emerald Green
    "protein":             "#10B981",   # Emerald Green
    "phenotype":           "#F59E0B",   # Amber / Orange
    "pathway":             "#8B5CF6",   # Purple
    "biological_process":  "#06B6D4",   # Cyan
    "anatomy":             "#64748B",   # Slate
    "other":               "#94A3B8",   # Neutral Grey
}


def build_plotly_subgraph(
    G: nx.DiGraph,
    center_node_id: Optional[int] = None,
    title: str = "Knowledge Graph Neighborhood",
    predicted_edges: Optional[List[Tuple[int, int, str]]] = None,
    height: int = 580,
) -> go.Figure:
    """
    Generate an interactive Plotly figure from a NetworkX DiGraph.

    Parameters
    ----------
    G : nx.DiGraph
        Graph with node attributes [entity_name, entity_type] and edge attributes [relation].
    center_node_id : Optional[int]
        Center/query entity ID to visually highlight.
    title : str
        Figure title.
    predicted_edges : Optional[List[Tuple[int, int, str]]]
        List of (src, tgt, relation) edges predicted by the model (rendered with dashed lines).
    height : int
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure object.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Empty graph — no nodes to display",
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#F8FAFC",
        )
        return fig

    # Compute 2D node layout
    try:
        pos = nx.spring_layout(G, seed=42, k=0.6 / np.sqrt(max(1, G.number_of_nodes())), iterations=50)
    except Exception:
        pos = nx.circular_layout(G)

    # 1. Existing edges trace
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    edge_hover_x: List[float] = []
    edge_hover_y: List[float] = []
    edge_hover_text: List[str] = []

    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # Midpoint for hover label
        edge_hover_x.append((x0 + x1) / 2.0)
        edge_hover_y.append((y0 + y1) / 2.0)
        u_name = G.nodes[u].get("entity_name", str(u))
        v_name = G.nodes[v].get("entity_name", str(v))
        rel = data.get("relation", "connected_to")
        edge_hover_text.append(f"{u_name} &rarr; <b>[{rel}]</b> &rarr; {v_name}")

    existing_edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#94A3B8"),
        hoverinfo="none",
        mode="lines",
        name="Existing Relationship",
    )

    existing_edge_hover = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers",
        marker=dict(size=4, color="#94A3B8", opacity=0.01),
        hoverinfo="text",
        text=edge_hover_text,
        showlegend=False,
    )

    # 2. Predicted novel edges trace (if any)
    pred_edge_x: List[Optional[float]] = []
    pred_edge_y: List[Optional[float]] = []
    pred_hover_x: List[float] = []
    pred_hover_y: List[float] = []
    pred_hover_text: List[str] = []

    if predicted_edges:
        for u, v, rel in predicted_edges:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            pred_edge_x.extend([x0, x1, None])
            pred_edge_y.extend([y0, y1, None])
            pred_hover_x.append((x0 + x1) / 2.0)
            pred_hover_y.append((y0 + y1) / 2.0)
            u_name = G.nodes[u].get("entity_name", str(u))
            v_name = G.nodes[v].get("entity_name", str(v))
            pred_hover_text.append(f"✨ <i>Predicted:</i> {u_name} &rarr; <b>[{rel}]</b> &rarr; {v_name}")

    predicted_edge_trace = go.Scatter(
        x=pred_edge_x,
        y=pred_edge_y,
        line=dict(width=2.5, color="#F59E0B", dash="dash"),
        hoverinfo="none",
        mode="lines",
        name="Model Hypothesis (Novel)",
    )

    predicted_edge_hover = go.Scatter(
        x=pred_hover_x,
        y=pred_hover_y,
        mode="markers",
        marker=dict(size=6, color="#F59E0B", opacity=0.01),
        hoverinfo="text",
        text=pred_hover_text,
        showlegend=False,
    )

    # 3. Nodes grouped by entity type for distinct colors and legend entries
    node_traces: List[go.Scatter] = []
    entity_groups: Dict[str, Dict[str, list]] = {}

    for node in G.nodes():
        if node not in pos:
            continue
        data = G.nodes[node]
        etype = str(data.get("entity_type", "other")).lower()
        ename = str(data.get("entity_name", str(node)))
        src_id = str(data.get("source_id", "N/A"))
        deg = G.degree(node)

        if etype not in entity_groups:
            entity_groups[etype] = {"x": [], "y": [], "text": [], "size": [], "ids": []}

        entity_groups[etype]["x"].append(pos[node][0])
        entity_groups[etype]["y"].append(pos[node][1])
        entity_groups[etype]["ids"].append(node)

        is_center = (center_node_id is not None and node == center_node_id)
        size = 28 if is_center else max(14, min(24, 12 + deg * 2))
        entity_groups[etype]["size"].append(size)

        center_badge = " ⭐ [QUERY CENTER]" if is_center else ""
        hover_txt = (
            f"<b>{ename}</b>{center_badge}<br>"
            f"Type: <span style='text-transform: capitalize;'>{etype}</span><br>"
            f"Source ID: {src_id}<br>"
            f"Connections: {deg}"
        )
        entity_groups[etype]["text"].append(hover_txt)

    for etype, gdata in entity_groups.items():
        color = ENTITY_COLORS.get(etype, ENTITY_COLORS["other"])
        node_traces.append(
            go.Scatter(
                x=gdata["x"],
                y=gdata["y"],
                mode="markers+text",
                name=etype.capitalize(),
                text=[G.nodes[n].get("entity_name", str(n)) for n in gdata["ids"]],
                textposition="top center",
                textfont=dict(size=10, color="#1E293B", family="system-ui, sans-serif"),
                hovertext=gdata["text"],
                hoverinfo="text",
                marker=dict(
                    size=gdata["size"],
                    color=color,
                    line=dict(width=2, color="#FFFFFF"),
                    opacity=0.92,
                ),
            )
        )

    # Assemble Plotly Figure
    traces = [existing_edge_trace, existing_edge_hover]
    if predicted_edges:
        traces.extend([predicted_edge_trace, predicted_edge_hover])
    traces.extend(node_traces)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#0F172A", family="sans-serif")),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=height,
    )
    return fig


def render_pyvis_network(
    G: nx.DiGraph,
    center_node_id: Optional[int] = None,
    height_px: int = 560,
) -> None:
    """
    Render an interactive physics-enabled PyVis network in Streamlit.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("[MediGraph] PyVis is not installed. Falling back to Plotly renderer.")
        return

    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#1E293B")
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    for node, data in G.nodes(data=True):
        ename = str(data.get("entity_name", str(node)))
        etype = str(data.get("entity_type", "other")).lower()
        color = ENTITY_COLORS.get(etype, ENTITY_COLORS["other"])
        src_id = str(data.get("source_id", "N/A"))

        is_center = (center_node_id is not None and node == center_node_id)
        size = 32 if is_center else 18
        title = f"{ename} ({etype})<br>Source ID: {src_id}"

        net.add_node(
            node,
            label=ename,
            title=title,
            color=color,
            size=size,
            shape="star" if is_center else "dot",
        )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        net.add_edge(u, v, title=rel, label=rel, color="#94A3B8", arrows="to")

    html_content = net.generate_html()
    components.html(html_content, height=height_px, scrolling=False)

