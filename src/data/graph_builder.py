"""
src/data/graph_builder.py
-------------------------
Graph construction and conversion module for MediGraph AI.

Responsibilities:
  - Convert nodes.csv and edges.csv into NetworkX DiGraph
  - Convert into PyTorch Geometric homogeneous Data format (edge_index, edge_type)
  - Generate comprehensive graph statistics (degrees, distributions, connectivity)
  - Generate an interactive HTML visualization of a graph sample using PyVis / NetworkX
  - Validate graph integrity before passing to R-GCN

Usage:
    from src.data.graph_builder import GraphBuilder
    builder = GraphBuilder()
    G = builder.build_networkx(node_df, edge_df)
    stats = builder.get_graph_stats(node_df, edge_df)
    edge_index, edge_type = builder.build_pyg_tensors(edge_df, rel_mapping)

Author: Member 1 — Data Engineering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd

# Optional torch import so data engineering doesn't crash if PyTorch isn't installed yet
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Entity type color palette (for visualizations)
# ---------------------------------------------------------------------------
ENTITY_COLORS: dict[str, str] = {
    "disease":             "#E05A47",   # Coral red
    "drug":                "#3B82F6",   # Blue
    "gene":                "#10B981",   # Green (includes proteins)
    "protein":             "#10B981",   # Green
    "phenotype":           "#F59E0B",   # Amber / yellow
    "pathway":             "#8B5CF6",   # Purple
    "biological_process":  "#06B6D4",   # Cyan
    "molecular_function":  "#EC4899",   # Pink
    "cellular_component":  "#84CC16",   # Lime
    "anatomy":             "#64748B",   # Slate / grey
    "exposure":            "#D97706",   # Bronze
}

DEFAULT_NODE_COLOR = "#94A3B8"  # Light slate for unmapped types


class GraphBuilder:
    """
    Constructs NetworkX graphs, PyTorch Geometric tensors, and statistics
    from processed MediGraph AI dataframes.

    Parameters
    ----------
    node_df : pd.DataFrame, optional
        Pre-loaded nodes dataframe.
    edge_df : pd.DataFrame, optional
        Pre-loaded edges dataframe.
    relation_mapping : dict, optional
        Pre-loaded relation mapping dict.
    """

    def __init__(
        self,
        node_df: Optional[pd.DataFrame] = None,
        edge_df: Optional[pd.DataFrame] = None,
        relation_mapping: Optional[dict[str, int]] = None,
    ) -> None:
        self.node_df = node_df
        self.edge_df = edge_df
        self.relation_mapping = relation_mapping

    # -----------------------------------------------------------------------
    # 1. NetworkX Graph Construction
    # -----------------------------------------------------------------------

    def build_networkx(
        self,
        node_df: Optional[pd.DataFrame] = None,
        edge_df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> nx.DiGraph:
        """
        Build a NetworkX DiGraph from node_df and edge_df.

        Each node in the graph has attributes:
          - `entity_name`: str
          - `entity_type`: str
          - `source_id`: str
          - `color`: hex color string for visualization

        Each edge has attributes:
          - `relation`: str

        Parameters
        ----------
        node_df : pd.DataFrame, optional
            Defaults to self.node_df.
        edge_df : pd.DataFrame, optional
            Defaults to self.edge_df.
        sample_size : int, optional
            If provided, sample this many edges for a quick exploration graph.
        seed : int
            Random seed for sampling.

        Returns
        -------
        nx.DiGraph
            Directed graph preserving original PrimeKG relationship direction.
        """
        node_df = node_df if node_df is not None else self.node_df
        edge_df = edge_df if edge_df is not None else self.edge_df

        if node_df is None or edge_df is None:
            raise ValueError(
                "[MediGraph] Both node_df and edge_df must be provided to build_networkx."
            )

        log.info("[MediGraph] Building NetworkX DiGraph...")
        G = nx.DiGraph()

        # Add all nodes with attributes
        for _, row in node_df.iterrows():
            nid = int(row["node_id"])
            etype = str(row["entity_type"])
            G.add_node(
                nid,
                entity_name=str(row["entity_name"]),
                entity_type=etype,
                source_id=str(row["source_id"]),
                color=ENTITY_COLORS.get(etype, DEFAULT_NODE_COLOR),
            )

        # Sample edges if requested
        edges_to_add = edge_df
        if sample_size is not None and len(edge_df) > sample_size:
            log.info(
                "[MediGraph] Sampling %d edges for NetworkX graph (seed=%d).",
                sample_size, seed,
            )
            edges_to_add = edge_df.sample(n=sample_size, random_state=seed)

        # Add directed edges with relation attribute
        for _, row in edges_to_add.iterrows():
            src = int(row["source"])
            tgt = int(row["target"])
            rel = str(row["relation"])
            G.add_edge(src, tgt, relation=rel)

        log.info(
            "[MediGraph] NetworkX DiGraph built: %d nodes, %d edges.",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    # -----------------------------------------------------------------------
    # 2. PyG Tensor Construction
    # -----------------------------------------------------------------------

    def build_pyg_tensors(
        self,
        edge_df: Optional[pd.DataFrame] = None,
        relation_mapping: Optional[dict[str, int]] = None,
    ) -> tuple[Any, Any]:
        """
        Convert edges dataframe to PyTorch Geometric edge tensors.

        Parameters
        ----------
        edge_df : pd.DataFrame, optional
            Defaults to self.edge_df.
        relation_mapping : dict, optional
            Defaults to self.relation_mapping.

        Returns
        -------
        edge_index : torch.LongTensor, shape (2, num_edges)
            Row 0: source node IDs. Row 1: target node IDs.
        edge_type : torch.LongTensor, shape (num_edges,)
            Integer relation IDs mapped via relation_mapping.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        KeyError
            If any relation string is missing from relation_mapping.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "[MediGraph] PyTorch is required to build PyG tensors. "
                "Install PyTorch with: pip install torch"
            )

        edge_df = edge_df if edge_df is not None else self.edge_df
        rel_map = relation_mapping if relation_mapping is not None else self.relation_mapping

        if edge_df is None or rel_map is None:
            raise ValueError(
                "[MediGraph] Both edge_df and relation_mapping are required for build_pyg_tensors."
            )

        # Map relation string → int
        missing = set(edge_df["relation"].unique()) - set(rel_map.keys())
        if missing:
            raise KeyError(
                f"[MediGraph] Relations in edge_df not found in relation_mapping: {missing}"
            )

        rel_ids = edge_df["relation"].map(rel_map).values.astype(np.int64)
        src_ids = edge_df["source"].values.astype(np.int64)
        tgt_ids = edge_df["target"].values.astype(np.int64)

        edge_index = torch.tensor(
            np.stack([src_ids, tgt_ids], axis=0),
            dtype=torch.long,
        )
        edge_type = torch.tensor(rel_ids, dtype=torch.long)

        log.info(
            "[MediGraph] PyG tensors created: edge_index shape %s, edge_type shape %s.",
            tuple(edge_index.shape),
            tuple(edge_type.shape),
        )
        return edge_index, edge_type

    def build_homogeneous(
        self,
        node_df: Optional[pd.DataFrame] = None,
        edge_df: Optional[pd.DataFrame] = None,
        rel_mapping: Optional[dict[str, int]] = None,
    ) -> tuple[Any, Any]:
        """
        Returns:
            edge_index: LongTensor shape (2, E)
            edge_type:  LongTensor shape (E,)
        Values are integer node IDs from node_df.node_id.
        All arguments fall back to the values supplied at construction time.
        """
        edge_df = edge_df if edge_df is not None else self.edge_df
        rel_mapping = rel_mapping if rel_mapping is not None else self.relation_mapping
        return self.build_pyg_tensors(edge_df=edge_df, relation_mapping=rel_mapping)

    def to_pyg_data(
        self,
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        rel_mapping: dict[str, int],
    ) -> Any:
        """
        Convert into PyTorch Geometric Data object.
        """
        edge_index, edge_type = self.build_homogeneous(node_df, edge_df, rel_mapping)
        try:
            from torch_geometric.data import Data
            return Data(
                edge_index=edge_index,
                edge_type=edge_type,
                num_nodes=len(node_df),
            )
        except ImportError:
            return {
                "edge_index": edge_index,
                "edge_type": edge_type,
                "num_nodes": len(node_df),
            }


    # -----------------------------------------------------------------------
    # 3. Comprehensive Graph Statistics
    # -----------------------------------------------------------------------

    def get_graph_stats(
        self,
        node_df: Optional[pd.DataFrame] = None,
        edge_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """
        Compute comprehensive statistics for the knowledge graph.

        Computes:
          - Basic counts: num_nodes, num_edges, num_node_types, num_relation_types
          - Degree statistics: in-degree, out-degree, total-degree (min, max, mean, median)
          - Distributions: entity type counts, relation type counts
          - Density and connectivity metrics
          - Top-10 hub nodes by total degree

        Parameters
        ----------
        node_df : pd.DataFrame, optional
        edge_df : pd.DataFrame, optional

        Returns
        -------
        dict
            Structured statistics dictionary.
        """
        node_df = node_df if node_df is not None else self.node_df
        edge_df = edge_df if edge_df is not None else self.edge_df

        if node_df is None or edge_df is None:
            raise ValueError(
                "[MediGraph] Both node_df and edge_df must be provided to get_graph_stats."
            )

        log.info("[MediGraph] Computing graph statistics...")

        num_nodes = len(node_df)
        num_edges = len(edge_df)
        num_entity_types = int(node_df["entity_type"].nunique())
        num_relation_types = int(edge_df["relation"].nunique())

        # Entity type distribution
        entity_dist = node_df["entity_type"].value_counts().to_dict()

        # Relation type distribution
        relation_dist = edge_df["relation"].value_counts().to_dict()

        # Degree calculations via pandas (fast, vectorised)
        out_degrees = edge_df["source"].value_counts().reindex(node_df["node_id"], fill_value=0)
        in_degrees  = edge_df["target"].value_counts().reindex(node_df["node_id"], fill_value=0)
        total_degrees = out_degrees + in_degrees

        def _dist_stats(s: pd.Series) -> dict[str, float]:
            return {
                "min":    float(s.min()),
                "max":    float(s.max()),
                "mean":   round(float(s.mean()), 2),
                "median": float(s.median()),
                "std":    round(float(s.std()), 2),
            }

        # Graph density: E / (V * (V - 1)) for directed graph
        possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / possible_edges if possible_edges > 0 else 0.0

        # Isolated nodes (degree == 0)
        isolated_nodes = int((total_degrees == 0).sum())

        # Top 10 hub nodes
        top10_ids = total_degrees.nlargest(10).index.tolist()
        top10_nodes = []
        node_lookup = node_df.set_index("node_id")
        for nid in top10_ids:
            if nid in node_lookup.index:
                row = node_lookup.loc[nid]
                top10_nodes.append({
                    "node_id":      int(nid),
                    "entity_name":  str(row["entity_name"]),
                    "entity_type":  str(row["entity_type"]),
                    "total_degree": int(total_degrees[nid]),
                    "out_degree":   int(out_degrees[nid]),
                    "in_degree":    int(in_degrees[nid]),
                })

        # Connected components (NetworkX weakly & strongly connected)
        G_temp = nx.DiGraph()
        G_temp.add_nodes_from(node_df["node_id"].tolist())
        for _, row in edge_df.iterrows():
            G_temp.add_edge(int(row["source"]), int(row["target"]))
        num_connected_components = int(nx.number_weakly_connected_components(G_temp))
        num_strongly_connected = int(nx.number_strongly_connected_components(G_temp))

        # Dataset provenance & sample determination
        dataset_source = "PrimeKG (Precision Medicine Knowledge Graph)"
        is_sample = bool(num_nodes < 100_000)

        # Degree distribution breakdown (degree -> count of nodes)
        deg_dist = {int(k): int(v) for k, v in total_degrees.value_counts().sort_index().items()}

        stats: dict[str, Any] = {
            "num_nodes":              num_nodes,
            "num_edges":              num_edges,
            "num_entity_types":       num_entity_types,
            "num_relation_types":     num_relation_types,
            "num_connected_components": num_connected_components,
            "num_strongly_connected": num_strongly_connected,
            "dataset_source":          dataset_source,
            "is_sample":               is_sample,
            "density":                round(density, 6),
            "isolated_nodes":         isolated_nodes,
            "isolated_pct":           round(100 * isolated_nodes / num_nodes, 2) if num_nodes > 0 else 0.0,
            "out_degree_stats":       _dist_stats(out_degrees),
            "in_degree_stats":        _dist_stats(in_degrees),
            "total_degree_stats":     _dist_stats(total_degrees),
            "degree_distribution":     deg_dist,
            "entity_type_distribution": entity_dist,
            "relation_type_distribution": relation_dist,
            "top_10_hubs":            top10_nodes,
            "most_connected_entities": top10_nodes,
        }

        log.info(
            "[MediGraph] Stats complete: %d nodes, %d edges, avg degree: %.1f.",
            num_nodes, num_edges, stats["total_degree_stats"]["mean"],
        )
        return stats

    # -----------------------------------------------------------------------
    # 4. Save statistics report
    # -----------------------------------------------------------------------

    def save_stats_report(
        self,
        stats: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Save graph statistics dict as formatted JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        log.info("[MediGraph] Graph statistics saved to '%s'", path)

    # -----------------------------------------------------------------------
    # 5. Visual Sample of the Graph (PyVis / HTML)
    # -----------------------------------------------------------------------

    def create_visual_sample(
        self,
        output_html_path: str | Path,
        node_df: Optional[pd.DataFrame] = None,
        edge_df: Optional[pd.DataFrame] = None,
        max_edges: int = 150,
        focus_entity: Optional[str] = None,
        seed: int = 42,
    ) -> Path:
        """
        Generate an interactive HTML visualization of a graph sample using PyVis.

        If PyVis is not installed, falls back to generating a static NetworkX
        edge-list summary HTML file with color-coded nodes.

        Parameters
        ----------
        output_html_path : str or Path
            Where to save the HTML file.
        node_df, edge_df : pd.DataFrame, optional
        max_edges : int
            Max edges in the sample (keeps visualization responsive).
        focus_entity : str, optional
            If provided, center the sample on the neighbourhood of this entity name.
        seed : int
            Random seed.

        Returns
        -------
        Path
            Path to the generated HTML file.
        """
        node_df = node_df if node_df is not None else self.node_df
        edge_df = edge_df if edge_df is not None else self.edge_df

        if node_df is None or edge_df is None:
            raise ValueError(
                "[MediGraph] Both node_df and edge_df must be provided to create_visual_sample."
            )

        output_path = Path(output_html_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select edges to visualize
        if focus_entity is not None:
            # Find node_id for focus entity
            match = node_df[node_df["entity_name"].str.lower() == focus_entity.lower()]
            if match.empty:
                log.warning(
                    "[MediGraph] Focus entity '%s' not found. Falling back to random sample.",
                    focus_entity,
                )
                sample_edges = edge_df.sample(n=min(max_edges, len(edge_df)), random_state=seed)
            else:
                focal_id = int(match.iloc[0]["node_id"])
                # 1-hop neighbours
                hop1_edges = edge_df[
                    (edge_df["source"] == focal_id) | (edge_df["target"] == focal_id)
                ]
                sample_edges = hop1_edges.head(max_edges)
        else:
            sample_edges = edge_df.sample(n=min(max_edges, len(edge_df)), random_state=seed)

        # Nodes referenced in sample
        sample_node_ids = set(sample_edges["source"].tolist()) | set(sample_edges["target"].tolist())
        sample_nodes = node_df[node_df["node_id"].isin(sample_node_ids)]

        node_lookup = {
            int(r["node_id"]): {
                "name": str(r["entity_name"]),
                "type": str(r["entity_type"]),
                "source_id": str(r["source_id"]),
            }
            for _, r in sample_nodes.iterrows()
        }

        # Attempt PyVis
        try:
            from pyvis.network import Network  # type: ignore[import]

            net = Network(
                height="650px",
                width="100%",
                directed=True,
                notebook=False,
                bgcolor="#1e1e2e",
                font_color="#ffffff",
            )
            net.barnes_hut(
                gravity=-3000,
                central_gravity=0.3,
                spring_length=120,
                spring_strength=0.05,
                damping=0.09,
            )

            # Add nodes
            for nid, info in node_lookup.items():
                color = ENTITY_COLORS.get(info["type"], DEFAULT_NODE_COLOR)
                title_html = (
                    f"Name: {info['name']}<br>"
                    f"Type: {info['type']}<br>"
                    f"Source ID: {info['source_id']}<br>"
                    f"Node ID: {nid}"
                )
                net.add_node(
                    n_id=nid,
                    label=info["name"][:20] + "..." if len(info["name"]) > 20 else info["name"],
                    title=title_html,
                    color=color,
                    shape="dot",
                    size=16,
                )

            # Add edges
            for _, r in sample_edges.iterrows():
                src = int(r["source"])
                tgt = int(r["target"])
                rel = str(r["relation"])
                net.add_edge(
                    src, tgt,
                    title=f"Relation: {rel}",
                    label=rel,
                    color="#4A90D9",
                    arrows="to",
                )

            net.write_html(str(output_path))
            log.info("[MediGraph] PyVis interactive graph saved → '%s'", output_path)
            return output_path

        except ImportError:
            log.warning(
                "[MediGraph] PyVis not installed. Generating static HTML graph summary..."
            )
            # Fallback: clean standalone HTML summary table
            html_content = _generate_html_fallback(sample_nodes, sample_edges, node_lookup)
            output_path.write_text(html_content, encoding="utf-8")
            log.info("[MediGraph] HTML graph summary saved → '%s'", output_path)
            return output_path


def _generate_html_fallback(
    sample_nodes: pd.DataFrame,
    sample_edges: pd.DataFrame,
    node_lookup: dict,
) -> str:
    """Generate a clean HTML summary table when PyVis is not available."""
    node_rows = ""
    for _, r in sample_nodes.iterrows():
        etype = r["entity_type"]
        color = ENTITY_COLORS.get(etype, DEFAULT_NODE_COLOR)
        node_rows += (
            f"<tr><td>{r['node_id']}</td>"
            f"<td><strong>{r['entity_name']}</strong></td>"
            f"<td><span style='background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;'>{etype}</span></td>"
            f"<td><code>{r['source_id']}</code></td></tr>\n"
        )

    edge_rows = ""
    for _, r in sample_edges.iterrows():
        src_name = node_lookup.get(int(r["source"]), {}).get("name", str(r["source"]))
        tgt_name = node_lookup.get(int(r["target"]), {}).get("name", str(r["target"]))
        edge_rows += (
            f"<tr><td>{src_name}</td>"
            f"<td><span style='color:#3B82F6;font-weight:bold;'>→ [{r['relation']}] →</span></td>"
            f"<td>{tgt_name}</td></tr>\n"
        )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MediGraph AI — Graph Sample</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #0f172a; color: #e2e8f0; margin: 20px; }}
    h1, h2 {{ color: #38bdf8; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 14px; }}
    th, td {{ border: 1px solid #334155; padding: 8px 12px; text-align: left; }}
    th {{ background: #1e293b; color: #94a3b8; }}
    tr:nth-child(even) {{ background: #1e293b; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; margin: 2px; }}
  </style>
</head>
<body>
  <h1>🧬 MediGraph AI — Knowledge Graph Sample</h1>
  <p>Sample containing <strong>{len(sample_nodes)} nodes</strong> and <strong>{len(sample_edges)} edges</strong>.</p>
  <p><em>To view interactive PyVis graph, run: <code>pip install pyvis</code></em></p>

  <h2>Nodes ({len(sample_nodes)})</h2>
  <table>
    <tr><th>Node ID</th><th>Entity Name</th><th>Entity Type</th><th>Source ID</th></tr>
    {node_rows}
  </table>

  <h2>Edges ({len(sample_edges)})</h2>
  <table>
    <tr><th>Source Entity</th><th>Relationship</th><th>Target Entity</th></tr>
    {edge_rows}
  </table>
</body>
</html>"""
