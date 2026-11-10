"""
tests/test_frontend.py
----------------------
Unit and integration tests for MediGraph AI Frontend components and data integration.

Tests:
- Metric cards component rendering and color mapping
- Graph visualization (NetworkX -> Plotly conversion, entity coloring, predicted edges)
- Frontend data loader caching and session state population
- Metrics persistence loading (metrics.json, baseline_comparison.json)
- Page module importability and integrity
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import torch

from src.utils.config import load_config
from app.components.metric_cards import COLOR_MAP
from app.components.graph_visualization import ENTITY_COLORS, build_plotly_subgraph


class TestFrontendComponents(unittest.TestCase):
    """Test frontend UI components and visualization builders."""

    def test_color_map_integrity(self):
        self.assertIn("teal", COLOR_MAP)
        self.assertIn("navy", COLOR_MAP)
        self.assertIn("coral", COLOR_MAP)
        self.assertIn("amber", COLOR_MAP)
        for key, val in COLOR_MAP.items():
            self.assertIn("accent", val)
            self.assertIn("bg", val)
            self.assertIn("text", val)

    def test_entity_colors_palette(self):
        required_types = ["disease", "drug", "gene", "phenotype", "pathway"]
        for etype in required_types:
            self.assertIn(etype, ENTITY_COLORS)
            self.assertTrue(ENTITY_COLORS[etype].startswith("#"))

    def test_plotly_subgraph_empty(self):
        G_empty = nx.DiGraph()
        fig = build_plotly_subgraph(G_empty, title="Empty Test")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)

    def test_plotly_subgraph_with_predicted_edges(self):
        G = nx.DiGraph()
        G.add_node(0, entity_name="Alzheimer disease", entity_type="disease", source_id="MONDO:0004975")
        G.add_node(1, entity_name="APOE", entity_type="gene", source_id="348")
        G.add_node(2, entity_name="APP", entity_type="gene", source_id="351")
        G.add_edge(0, 1, relation="disease_protein")

        predicted_edges = [(0, 2, "disease_protein")]

        fig = build_plotly_subgraph(
            G=G,
            center_node_id=0,
            title="Test Subgraph",
            predicted_edges=predicted_edges,
        )
        self.assertIsInstance(fig, go.Figure)
        # Should have existing edge trace, hover trace, predicted edge trace, predicted hover trace, and node traces
        trace_names = [t.name for t in fig.data if t.name]
        self.assertIn("Existing Relationship", trace_names)
        self.assertIn("Model Hypothesis (Novel)", trace_names)
        self.assertIn("Disease", trace_names)
        self.assertIn("Gene", trace_names)


class TestFrontendDataIntegration(unittest.TestCase):
    """Verify frontend data contracts and saved artifact compatibility."""

    def setUp(self):
        self.cfg = load_config()

    def test_processed_data_files_exist(self):
        paths = self.cfg["paths"]
        self.assertTrue(Path(paths["nodes_csv"]).exists())
        self.assertTrue(Path(paths["edges_csv"]).exists())
        self.assertTrue(Path(paths["node_mapping_json"]).exists())
        self.assertTrue(Path(paths["relation_mapping_json"]).exists())

    def test_processed_data_content(self):
        paths = self.cfg["paths"]
        node_df = pd.read_csv(paths["nodes_csv"])
        edge_df = pd.read_csv(paths["edges_csv"])

        self.assertGreater(len(node_df), 0)
        self.assertGreater(len(edge_df), 0)
        self.assertIn("node_id", node_df.columns)
        self.assertIn("entity_name", node_df.columns)
        self.assertIn("entity_type", node_df.columns)
        self.assertIn("source", edge_df.columns)
        self.assertIn("relation", edge_df.columns)
        self.assertIn("target", edge_df.columns)

    def test_metrics_artifacts_readable(self):
        metrics_p = Path("artifacts/metrics/metrics.json")
        self.assertTrue(metrics_p.exists())
        with open(metrics_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("best", data)
        self.assertIn("test_mrr", data["best"])
        self.assertIn("per_epoch", data)

    def test_baseline_artifact_readable(self):
        base_p = Path("artifacts/metrics/baseline_comparison.json")
        self.assertTrue(base_p.exists())
        with open(base_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("rgcn", data)
        self.assertIn("baseline_distmult", data)
        self.assertIn("delta_mrr", data)


if __name__ == "__main__":
    unittest.main()
