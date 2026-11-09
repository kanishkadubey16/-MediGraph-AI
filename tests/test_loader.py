"""
tests/test_loader.py
--------------------
Unit tests for MediGraphLoader and GraphBuilder against processed files.

Author: Member 1 — Data Engineering
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import MediGraphLoader
from src.data.graph_builder import GraphBuilder
from src.utils.config import load_config


class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.loader = MediGraphLoader()

    def test_load_all_processed(self) -> None:
        """Verify load_all loads nodes.csv, edges.csv, and mappings successfully."""
        node_df, edge_df, node_mapping, rel_mapping = self.loader.load_all(self.cfg)

        self.assertGreater(len(node_df), 0)
        self.assertGreater(len(edge_df), 0)
        self.assertGreater(len(node_mapping), 0)
        self.assertGreater(len(rel_mapping), 0)

        # Check required columns
        for col in ["node_id", "entity_name", "entity_type", "source_id"]:
            self.assertIn(col, node_df.columns)
        for col in ["source", "relation", "target"]:
            self.assertIn(col, edge_df.columns)

    def test_graph_builder_pyg_tensors(self) -> None:
        """Verify GraphBuilder creates PyG tensors from loaded files."""
        node_df, edge_df, _, rel_mapping = self.loader.load_all(self.cfg)
        builder = GraphBuilder(node_df, edge_df, rel_mapping)

        edge_index, edge_type = builder.build_pyg_tensors()
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], len(edge_df))
        self.assertEqual(edge_type.shape[0], len(edge_df))

        # Check homogeneous builder interface
        edge_idx2, edge_type2 = builder.build_homogeneous(node_df, edge_df, rel_mapping)
        self.assertEqual(edge_idx2.shape, edge_index.shape)
        self.assertEqual(edge_type2.shape, edge_type.shape)

    def test_networkx_graph_conversion(self) -> None:
        """Verify GraphBuilder creates a valid NetworkX DiGraph."""
        node_df, edge_df, _, rel_mapping = self.loader.load_all(self.cfg)
        builder = GraphBuilder(node_df, edge_df, rel_mapping)
        G = builder.build_networkx()

        self.assertEqual(G.number_of_nodes(), len(node_df))
        self.assertEqual(G.number_of_edges(), len(edge_df))
        self.assertTrue(G.is_directed())


if __name__ == "__main__":
    unittest.main()

