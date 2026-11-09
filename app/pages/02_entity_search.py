"""
app/pages/02_entity_search.py
-----------------------------
Entity search page — node lookup and 1-hop neighbourhood viewer.
Member 4 responsibility — MediGraph AI.

Features
--------
1. Text input for entity search (case-insensitive substring match).
2. Up to 10 matching results shown; user selects one.
3. Entity detail card: node_id, entity_type, source_id.
4. 1-hop neighbourhood retrieved via get_node_neighbourhood() (§8.2).
5. Neighbourhood displayed as a Plotly table.
6. Small interactive PyVis network graph embedded via st.components.v1.html.
7. Graceful "Data not loaded" guard with instructions.

Design decisions
----------------
* Blues (#4A90D9) for existing/known edges; orange (#E8831A) for predicted.
* Never labels scores as "Probability" — uses "Model Score".
* Always shows the disclaimer.
* All paths sourced from config.yaml via load_config().
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger
from app.components.data_loader import get_data_status

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Entity Search — MediGraph AI",
    page_icon="🔍",
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
# Constants
# ---------------------------------------------------------------------------
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
_DEFAULT_NODE_COLOUR = "#AAAAAA"
_EDGE_COLOUR = "#4A90D9"

# ---------------------------------------------------------------------------
# Header + disclaimer
# ---------------------------------------------------------------------------
st.title("🔍 Entity Search")
st.caption("Search for biomedical entities and explore their 1-hop neighbourhood")
st.warning(_DISCLAIMER, icon="⚠️")
st.divider()

# ---------------------------------------------------------------------------
# Data guard
# ---------------------------------------------------------------------------
status = get_data_status()

if not status["data_loaded"]:
    st.info(
        "📂 **Graph data is not loaded.**  "
        "Go to the **Dashboard** page and click **Load Graph Data** first.",
        icon="ℹ️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Retrieve session state
# ---------------------------------------------------------------------------
node_df: pd.DataFrame = st.session_state.node_df
edge_index = st.session_state.edge_index
edge_type = st.session_state.edge_type
rel_mapping: dict[str, int] = st.session_state.rel_mapping or {}
rel_mapping_inv: dict[int, str] = {v: k for k, v in rel_mapping.items()}

# ---------------------------------------------------------------------------
# Search input
# ---------------------------------------------------------------------------
st.subheader("🔎 Search Entity")
query = st.text_input(
    "Entity name",
    placeholder="e.g. Alzheimer disease, metformin, APOE …",
    help="Type a partial or full entity name. The search is case-insensitive.",
)

if not query:
    st.info("Enter an entity name above to search the knowledge graph.", icon="ℹ️")
    st.stop()

# Case-insensitive substring match — up to 10 results
mask = node_df["entity_name"].str.contains(query, case=False, na=False)
matches: pd.DataFrame = node_df[mask].head(10)

if matches.empty:
    st.warning(
        f"[MediGraph] No entities found matching **'{query}'**.  "
        "Try a shorter or different term.",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Entity selection
# ---------------------------------------------------------------------------
entity_options: list[str] = matches["entity_name"].tolist()
selected_name: str = st.selectbox(
    f"Found {len(matches)} match(es). Select an entity:",
    entity_options,
)

selected_row = node_df[node_df["entity_name"] == selected_name].iloc[0]
selected_node_id: int = int(selected_row["node_id"])

st.divider()

# ---------------------------------------------------------------------------
# Entity detail card
# ---------------------------------------------------------------------------
st.subheader(f"📋 Entity Details: {selected_name}")
detail_col1, detail_col2, detail_col3 = st.columns(3)
detail_col1.metric("Node ID", str(selected_node_id))
detail_col2.metric("Entity Type", selected_row["entity_type"])
detail_col3.metric("Source ID", selected_row["source_id"])

st.divider()

# ---------------------------------------------------------------------------
# 1-hop neighbourhood
# ---------------------------------------------------------------------------
st.subheader("🕸️ 1-Hop Neighbourhood")

try:
    from src.inference.predict import get_node_neighborhood

    neighborhood: dict[str, Any] = get_node_neighborhood(
        node_id=selected_node_id,
        edge_index=edge_index,
        edge_type=edge_type,
        node_df=node_df,
        rel_mapping_inv=rel_mapping_inv,
        max_hops=1,
    )
    neighbors: list[dict] = neighborhood.get("neighbors", [])
except ImportError:
    # Inference module not yet available; fall back to manual computation
    log.warning(
        "[MediGraph] src.inference.predict not importable; "
        "computing neighbourhood manually."
    )
    neighbors = _compute_neighborhood_fallback(
        selected_node_id, edge_index, edge_type, node_df, rel_mapping_inv
    )
    neighborhood = {
        "center": {
            "node_id": selected_node_id,
            "entity_name": selected_name,
            "entity_type": selected_row["entity_type"],
        },
        "neighbors": neighbors,
    }
except Exception as exc:  # noqa: BLE001
    st.error(f"❌ [MediGraph] Error computing neighbourhood: {exc}", icon="❌")
    log.error("[MediGraph] get_node_neighborhood error: %s", exc, exc_info=True)
    st.stop()


def _compute_neighborhood_fallback(
    node_id: int,
    edge_index,
    edge_type,
    node_df: pd.DataFrame,
    rel_mapping_inv: dict[int, str],
) -> list[dict]:
    """Compute 1-hop neighbourhood directly from tensors when inference module is unavailable."""
    import torch

    ei = edge_index  # (2, E)
    et = edge_type   # (E,)

    rows: list[dict] = []
    id_to_row = node_df.set_index("node_id")

    # Outgoing edges: ei[0] == node_id
    out_mask = (ei[0] == node_id).nonzero(as_tuple=True)[0]
    for idx in out_mask.tolist():
        tgt = int(ei[1, idx].item())
        rel = rel_mapping_inv.get(int(et[idx].item()), str(et[idx].item()))
        if tgt in id_to_row.index:
            row = id_to_row.loc[tgt]
            rows.append({
                "node_id": tgt,
                "entity_name": row["entity_name"],
                "entity_type": row["entity_type"],
                "relation": rel,
                "direction": "outgoing",
            })

    # Incoming edges: ei[1] == node_id
    in_mask = (ei[1] == node_id).nonzero(as_tuple=True)[0]
    for idx in in_mask.tolist():
        src = int(ei[0, idx].item())
        rel = rel_mapping_inv.get(int(et[idx].item()), str(et[idx].item()))
        if src in id_to_row.index:
            row = id_to_row.loc[src]
            rows.append({
                "node_id": src,
                "entity_name": row["entity_name"],
                "entity_type": row["entity_type"],
                "relation": rel,
                "direction": "incoming",
            })

    return rows


if not neighbors:
    st.info(
        f"No neighbours found for **{selected_name}** in the loaded graph.",
        icon="ℹ️",
    )
else:
    st.markdown(
        f"**{len(neighbors)}** neighbour(s) found for **{selected_name}**."
    )

    # ---- Plotly table of neighbours ------------------------------------
    st.markdown("#### Neighbour Table")
    nbr_df = pd.DataFrame(neighbors)[
        ["entity_name", "entity_type", "relation", "direction"]
    ].rename(
        columns={
            "entity_name": "Neighbour Name",
            "entity_type": "Entity Type",
            "relation": "Relation",
            "direction": "Direction",
        }
    )

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(nbr_df.columns),
                    fill_color="#1E3A5F",
                    font=dict(color="white", size=13),
                    align="left",
                ),
                cells=dict(
                    values=[nbr_df[c].tolist() for c in nbr_df.columns],
                    fill_color="#0E1117",
                    font=dict(color="#FAFAFA", size=12),
                    align="left",
                ),
            )
        ]
    )
    fig_table.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_table, use_container_width=True)

    st.divider()

    # ---- PyVis network graph -------------------------------------------
    st.markdown("#### Neighbourhood Graph")
    _build_pyvis_neighborhood(
        center=neighborhood["center"],
        neighbors=neighbors,
    )


def _build_pyvis_neighborhood(
    center: dict[str, Any],
    neighbors: list[dict[str, Any]],
) -> None:
    """Render a 1-hop neighbourhood as an interactive PyVis graph embedded in Streamlit.

    Parameters
    ----------
    center :
        Dict with keys ``node_id``, ``entity_name``, ``entity_type``.
    neighbors :
        List of neighbour dicts (see §8.2 schema).
    """
    try:
        from pyvis.network import Network
    except ImportError:
        st.info(
            "📦 PyVis is not installed. Install it with `pip install pyvis` "
            "to see the interactive graph.",
            icon="ℹ️",
        )
        return

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0E1117",
        font_color="#FAFAFA",
        notebook=False,
        directed=True,
    )
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=100)

    # Center node
    center_colour = _NODE_TYPE_COLOURS.get(
        center.get("entity_type", ""), _DEFAULT_NODE_COLOUR
    )
    net.add_node(
        center["node_id"],
        label=center["entity_name"],
        color=center_colour,
        size=25,
        title=(
            f"<b>{center['entity_name']}</b><br>"
            f"Type: {center.get('entity_type', '—')}<br>"
            f"Node ID: {center['node_id']}"
        ),
        borderWidth=3,
        borderWidthSelected=5,
    )

    seen_nodes: set[int] = {center["node_id"]}
    for nbr in neighbors:
        nid = nbr["node_id"]
        nbr_colour = _NODE_TYPE_COLOURS.get(nbr.get("entity_type", ""), _DEFAULT_NODE_COLOUR)
        if nid not in seen_nodes:
            net.add_node(
                nid,
                label=nbr["entity_name"],
                color=nbr_colour,
                size=15,
                title=(
                    f"<b>{nbr['entity_name']}</b><br>"
                    f"Type: {nbr.get('entity_type', '—')}<br>"
                    f"Node ID: {nid}"
                ),
            )
            seen_nodes.add(nid)

        if nbr["direction"] == "outgoing":
            src, tgt = center["node_id"], nid
        else:
            src, tgt = nid, center["node_id"]

        net.add_edge(
            src,
            tgt,
            label=nbr["relation"],
            color=_EDGE_COLOUR,
            title=nbr["relation"],
            arrows="to",
        )

    # Write to temp file and embed
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        net.save_graph(tmp.name)
        html_content = Path(tmp.name).read_text(encoding="utf-8")

    components.html(html_content, height=520, scrolling=False)

    # Legend
    st.markdown(
        "**Legend:** "
        + "  ".join(
            f"<span style='color:{c};'>⬤</span> {t}"
            for t, c in _NODE_TYPE_COLOURS.items()
            if any(n.get("entity_type") == t for n in neighbors)
            or center.get("entity_type") == t
        ),
        unsafe_allow_html=True,
    )


st.divider()
st.caption(
    "MediGraph AI · Research prototype · Not for clinical use."
)
