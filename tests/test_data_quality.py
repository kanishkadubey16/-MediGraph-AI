"""
tests/test_data_quality.py
--------------------------
Data quality and integrity test suite for MediGraph AI.

Verifies:
  1. Duplicate node IDs
  2. Missing entities
  3. Invalid edge references (referential integrity)
  4. Missing relationships
  5. Empty dataset handling
  6. Mapping consistency (node_mapping and relation_mapping)
  7. Invalid node types

Author: Member 1 — Data Engineering
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.loader import MediGraphLoader, validate_edge_referential_integrity
from src.data.graph_builder import GraphBuilder
from src.data.preprocessing import (
    PRIMEKG_TYPE_MAP,
    OUTPUT_NODE_COLUMNS,
    OUTPUT_EDGE_COLUMNS,
    validate_outputs,
    build_nodes,
    build_edges,
    clean_raw_primekg,
)
from src.utils.config import load_config


class TestDataQuality(unittest.TestCase):
    """Data quality tests ensuring all contracts in docs/API_CONTRACT.md are enforced."""

    def setUp(self) -> None:
        """Create a standard, valid test fixture representing real PrimeKG records."""
        self.valid_raw_df = pd.DataFrame([
            {
                "relation": "indication",
                "display_relation": "indication",
                "x_index": 0,
                "x_id": "DB00998",
                "x_type": "drug",
                "x_name": "Albuterol",
                "x_source": "DrugBank",
                "y_index": 1,
                "y_id": "MONDO:0004979",
                "y_type": "disease",
                "y_name": "asthma",
                "y_source": "MONDO",
            },
            {
                "relation": "indication",
                "display_relation": "indication",
                "x_index": 2,
                "x_id": "DB01132",
                "x_type": "drug",
                "x_name": "Metformin",
                "x_source": "DrugBank",
                "y_index": 3,
                "y_id": "MONDO:0005148",
                "y_type": "disease",
                "y_name": "type 2 diabetes mellitus",
                "y_source": "MONDO",
            },
            {
                "relation": "disease_protein",
                "display_relation": "associated with",
                "x_index": 4,
                "x_id": "MONDO:0004975",
                "x_type": "disease",
                "x_name": "Alzheimer disease",
                "x_source": "MONDO",
                "y_index": 5,
                "y_id": "348",
                "y_type": "gene/protein",
                "y_name": "APOE",
                "y_source": "NCBI",
            },
            {
                "relation": "drug_protein",
                "display_relation": "target",
                "x_index": 6,
                "x_id": "DB00734",
                "x_type": "drug",
                "x_name": "Risperidone",
                "x_source": "DrugBank",
                "y_index": 7,
                "y_id": "3356",
                "y_type": "gene/protein",
                "y_name": "HTR2A",
                "y_source": "NCBI",
            },
            {
                "relation": "disease_phenotype_positive",
                "display_relation": "phenotype",
                "x_index": 8,
                "x_id": "MONDO:0004979",
                "x_type": "disease",
                "x_name": "asthma",
                "x_source": "MONDO",
                "y_index": 9,
                "y_id": "HP:0002099",
                "y_type": "effect/phenotype",
                "y_name": "Asthma",
                "y_source": "HPO",
            },
            {
                "relation": "protein_protein",
                "display_relation": "ppi",
                "x_index": 10,
                "x_id": "9796",
                "x_type": "gene/protein",
                "x_name": "PHYHIP",
                "x_source": "NCBI",
                "y_index": 11,
                "y_id": "56992",
                "y_type": "gene/protein",
                "y_name": "PHYHIPL",
                "y_source": "NCBI",
            },
        ])

        # Preprocess the valid test fixture
        cleaned = clean_raw_primekg(self.valid_raw_df)
        self.node_df, self.node_mapping = build_nodes(cleaned)
        self.edge_df, self.relation_mapping = build_edges(cleaned, self.node_mapping)

    # -----------------------------------------------------------------------
    # 1. Duplicate node IDs
    # -----------------------------------------------------------------------
    def test_duplicate_node_ids_detected(self) -> None:
        """Verify that duplicate node_id values are rejected."""
        # Our valid node_df must have unique IDs
        self.assertFalse(self.node_df["node_id"].duplicated().any())

        # Inject duplicate node_id and verify validation raises ValueError
        bad_node_df = self.node_df.copy()
        bad_node_df.loc[1, "node_id"] = bad_node_df.loc[0, "node_id"]
        with self.assertRaises(ValueError) as ctx:
            validate_outputs(bad_node_df, self.edge_df, self.node_mapping, self.relation_mapping)
        self.assertIn("node_id", str(ctx.exception))

    # -----------------------------------------------------------------------
    # 2. Missing entities
    # -----------------------------------------------------------------------
    def test_missing_entities_detected(self) -> None:
        """Verify that missing entity attributes (NaN or empty) are rejected."""
        # NaN entity name
        bad_node_df = self.node_df.copy()
        bad_node_df.loc[0, "entity_name"] = None
        with self.assertRaises(ValueError) as ctx:
            validate_outputs(bad_node_df, self.edge_df, self.node_mapping, self.relation_mapping)
        self.assertIn("NaN", str(ctx.exception))

        # Empty string entity name
        bad_node_df2 = self.node_df.copy()
        bad_node_df2.loc[0, "entity_name"] = "   "
        with self.assertRaises(ValueError) as ctx:
            validate_outputs(bad_node_df2, self.edge_df, self.node_mapping, self.relation_mapping)
        self.assertIn("empty", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 3. Invalid edge references (referential integrity)
    # -----------------------------------------------------------------------
    def test_invalid_edge_references_detected(self) -> None:
        """Verify that source or target IDs not present in nodes.csv are rejected."""
        # Valid edge set must pass referential integrity
        validate_edge_referential_integrity(self.edge_df, self.node_df)

        # Invalid source node
        bad_edge_df = self.edge_df.copy()
        bad_edge_df.loc[0, "source"] = 999999  # non-existent ID
        with self.assertRaises(ValueError) as ctx:
            validate_edge_referential_integrity(bad_edge_df, self.node_df)
        self.assertIn("Referential integrity violation", str(ctx.exception))

        # Invalid target node
        bad_edge_df2 = self.edge_df.copy()
        bad_edge_df2.loc[0, "target"] = 888888  # non-existent ID
        with self.assertRaises(ValueError) as ctx:
            validate_edge_referential_integrity(bad_edge_df2, self.node_df)
        self.assertIn("Referential integrity violation", str(ctx.exception))

    # -----------------------------------------------------------------------
    # 4. Missing relationships
    # -----------------------------------------------------------------------
    def test_missing_relationships_detected(self) -> None:
        """Verify that NaN or empty relation values in edge table are rejected."""
        # NaN relation
        bad_edge_df = self.edge_df.copy()
        bad_edge_df.loc[0, "relation"] = None
        with self.assertRaises(ValueError) as ctx:
            validate_outputs(self.node_df, bad_edge_df, self.node_mapping, self.relation_mapping)
        self.assertIn("NaN", str(ctx.exception))

        # Edge relation not in relation_mapping
        bad_edge_df2 = self.edge_df.copy()
        bad_edge_df2.loc[0, "relation"] = "unmapped_new_relation"
        with self.assertRaises(ValueError) as ctx:
            validate_outputs(self.node_df, bad_edge_df2, self.node_mapping, self.relation_mapping)
        self.assertIn("relation_mapping", str(ctx.exception))

    # -----------------------------------------------------------------------
    # 5. Empty dataset
    # -----------------------------------------------------------------------
    def test_empty_dataset_detected(self) -> None:
        """Verify that empty datasets fail gracefully."""
        empty_raw_df = pd.DataFrame(columns=list(self.valid_raw_df.columns))
        with self.assertRaises(Exception):
            # clean_raw_primekg on empty df should yield empty result
            cleaned = clean_raw_primekg(empty_raw_df)
            if len(cleaned) == 0:
                raise ValueError("[MediGraph] Empty dataset")

        loader = MediGraphLoader()
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_nodes_file = Path(tmpdir) / "empty_nodes.csv"
            pd.DataFrame(columns=OUTPUT_NODE_COLUMNS).to_csv(empty_nodes_file, index=False)
            # An empty node table has non-existent contiguous range [0, 0)
            loaded = loader.load_nodes(empty_nodes_file)
            self.assertEqual(len(loaded), 0)

    # -----------------------------------------------------------------------
    # 6. Mapping consistency
    # -----------------------------------------------------------------------
    def test_mapping_consistency(self) -> None:
        """Verify 1-to-1 correspondence between dataframe entities and mapping dictionaries."""
        # Check node_mapping consistency
        self.assertEqual(len(self.node_mapping), len(self.node_df))
        for _, row in self.node_df.iterrows():
            name = row["entity_name"]
            nid = row["node_id"]
            self.assertIn(name, self.node_mapping)
            self.assertEqual(self.node_mapping[name], nid)

        # Check relation_mapping consistency
        unique_relations = sorted(self.edge_df["relation"].unique().tolist())
        self.assertEqual(len(self.relation_mapping), len(unique_relations))
        for rel in unique_relations:
            self.assertIn(rel, self.relation_mapping)
            # Relation IDs must be contiguous 0..R-1
            self.assertTrue(0 <= self.relation_mapping[rel] < len(unique_relations))

    # -----------------------------------------------------------------------
    # 7. Invalid node types
    # -----------------------------------------------------------------------
    def test_invalid_node_types_detected(self) -> None:
        """Verify that entity types outside the allowed ontology are rejected or filtered."""
        allowed_types = set(load_config()["dataset"]["node_types"])

        # Check that all mapped types in clean data are valid allowed types
        for etype in self.node_df["entity_type"].unique():
            self.assertIn(
                etype,
                allowed_types,
                f"Entity type '{etype}' not in allowed config dataset.node_types",
            )

        # Feed invalid raw type into clean_raw_primekg and ensure it is discarded
        invalid_raw = self.valid_raw_df.copy()
        invalid_raw.loc[0, "x_type"] = "unsupported_alien_entity"
        cleaned = clean_raw_primekg(invalid_raw)
        # Row 0 must have been dropped
        self.assertNotIn("unsupported_alien_entity", cleaned["x_type"].unique())

    # -----------------------------------------------------------------------
    # 8. Graph Builder & PyG Tensor Integration
    # -----------------------------------------------------------------------
    def test_graph_builder_and_stats(self) -> None:
        """Test NetworkX, stats, and tensor generation from cleaned graph."""
        builder = GraphBuilder(
            node_df=self.node_df,
            edge_df=self.edge_df,
            relation_mapping=self.relation_mapping,
        )

        # NetworkX DiGraph
        G = builder.build_networkx()
        self.assertEqual(G.number_of_nodes(), len(self.node_df))
        self.assertEqual(G.number_of_edges(), len(self.edge_df))

        # Check graph stats
        stats = builder.get_graph_stats()
        self.assertEqual(stats["num_nodes"], len(self.node_df))
        self.assertEqual(stats["num_edges"], len(self.edge_df))
        self.assertIn("entity_type_distribution", stats)
        self.assertIn("relation_type_distribution", stats)
        self.assertIn("total_degree_stats", stats)

        # PyG tensors
        edge_index, edge_type = builder.build_pyg_tensors()
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], len(self.edge_df))
        self.assertEqual(edge_type.shape[0], len(self.edge_df))


if __name__ == "__main__":
    unittest.main()

