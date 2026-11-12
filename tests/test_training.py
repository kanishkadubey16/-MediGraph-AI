"""
tests/test_training.py
----------------------
Unit and integration tests for MediGraph AI Member 3 deliverables:
- Negative sampling (TypedNegativeSampler, corrupt_triples)
- Evaluation metrics (MRR, Hits@K, ROC-AUC, classification metrics)
- Filtered link prediction evaluation
- DistMult baseline model and comparison
- Training pipeline and checkpoint generation
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.rgcn import RGCNEncoder
from src.models.decoder import DistMultDecoder, MediGraphModel
from src.training.negative_sampling import TypedNegativeSampler, corrupt_triples
from src.training.splitter import split_edges, sample_negatives
from src.training.evaluator import LinkPredictionEvaluator
from src.training.evaluate import evaluate_link_prediction
from src.training.baseline import DistMultBaseline, train_baseline, compare_rgcn_and_baseline
from src.training.train import train, _save_checkpoint
from src.inference.predict import predict_links, get_node_neighborhood, DISCLAIMER_TEXT
from src.utils.metrics import (
    compute_mrr,
    compute_hits_at_k,
    compute_roc_auc,
    compute_classification_metrics,
)


class TestDatasetSplitting(unittest.TestCase):
    """Test stratified edge splitting and leakage prevention."""

    def setUp(self):
        # 10 edges across 3 relation types
        self.edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
            [1, 2, 3, 4, 5, 0, 2, 3, 4, 5],
        ], dtype=torch.long)
        self.edge_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)

    def test_split_edges_no_leakage(self):
        splits = split_edges(
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        train_ei = splits["train"]["edge_index"]
        val_ei = splits["val"]["edge_index"]
        test_ei = splits["test"]["edge_index"]

        train_set = {(int(s), int(t)) for s, t in zip(train_ei[0].tolist(), train_ei[1].tolist())}
        val_set = {(int(s), int(t)) for s, t in zip(val_ei[0].tolist(), val_ei[1].tolist())}
        test_set = {(int(s), int(t)) for s, t in zip(test_ei[0].tolist(), test_ei[1].tolist())}

        # Check total edge count is preserved
        total = len(train_set) + len(val_set) + len(test_set)
        self.assertEqual(total, self.edge_index.shape[1])

        # Check strict zero overlap between splits
        self.assertEqual(len(train_set.intersection(val_set)), 0, "Leakage between train and val")
        self.assertEqual(len(train_set.intersection(test_set)), 0, "Leakage between train and test")
        self.assertEqual(len(val_set.intersection(test_set)), 0, "Leakage between val and test")

    def test_split_edges_reproducibility(self):
        splits_a = split_edges(self.edge_index, self.edge_type, seed=42)
        splits_b = split_edges(self.edge_index, self.edge_type, seed=42)
        self.assertTrue(torch.equal(splits_a["train"]["edge_index"], splits_b["train"]["edge_index"]))

    def test_split_edges_invalid_ratios(self):
        with self.assertRaises(ValueError):
            split_edges(self.edge_index, self.edge_type, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)

        with self.assertRaises(ValueError):
            split_edges(self.edge_index, self.edge_type, train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)


class TestMetrics(unittest.TestCase):
    """Test link prediction and classification evaluation metrics."""

    def test_mrr_perfect(self):
        ranks = [1, 1, 1, 1]
        self.assertAlmostEqual(compute_mrr(ranks), 1.0)

    def test_mrr_known_values(self):
        # 1/1 + 1/2 + 1/4 = 1.0 + 0.5 + 0.25 = 1.75 / 3 = 0.58333...
        ranks = [1, 2, 4]
        self.assertAlmostEqual(compute_mrr(ranks), 1.75 / 3, places=4)

    def test_mrr_empty(self):
        self.assertEqual(compute_mrr([]), 0.0)

    def test_hits_at_k(self):
        ranks = [1, 2, 5, 10, 20]
        self.assertAlmostEqual(compute_hits_at_k(ranks, 1), 1 / 5)
        self.assertAlmostEqual(compute_hits_at_k(ranks, 3), 2 / 5)
        self.assertAlmostEqual(compute_hits_at_k(ranks, 10), 4 / 5)
        self.assertAlmostEqual(compute_hits_at_k(ranks, 50), 1.0)

    def test_roc_auc_separable(self):
        pos_scores = np.array([5.0, 6.0, 7.0])
        neg_scores = np.array([0.1, -1.0, 0.5])
        auc = compute_roc_auc(pos_scores, neg_scores)
        self.assertAlmostEqual(auc, 1.0)

    def test_classification_metrics(self):
        pos_scores = np.array([2.0, 3.0, 1.5, 4.0])   # all sigmoid > 0.5 -> 1
        neg_scores = np.array([-2.0, -3.0, -1.5, -4.0]) # all sigmoid < 0.5 -> 0
        metrics = compute_classification_metrics(pos_scores, neg_scores, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)


class TestNegativeSampling(unittest.TestCase):
    """Test typed and filtered negative sampling."""

    def setUp(self):
        self.num_nodes = 10
        self.edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
        ], dtype=torch.long)
        self.edge_type = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

        self.node_df = pd.DataFrame({
            "node_id": list(range(10)),
            "entity_name": [f"E_{i}" for i in range(10)],
            "entity_type": ["disease"] * 5 + ["gene"] * 5,
            "source_id": [f"S_{i}" for i in range(10)],
        })

    def test_negative_sampling_shapes(self):
        sampler = TypedNegativeSampler(
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            num_nodes=self.num_nodes,
            node_df=self.node_df,
            seed=42,
        )

        neg_ei, neg_et = sampler.sample(self.edge_index, self.edge_type, num_neg_per_pos=2)
        self.assertEqual(neg_ei.shape, (2, 10))
        self.assertEqual(neg_et.shape, (10,))
        self.assertTrue(torch.all(neg_ei >= 0))
        self.assertTrue(torch.all(neg_ei < self.num_nodes))

    def test_corrupt_triples_helper(self):
        neg_ei, neg_et = corrupt_triples(
            pos_edge_index=self.edge_index,
            pos_edge_type=self.edge_type,
            num_nodes=self.num_nodes,
            num_neg_per_pos=1,
            seed=42,
        )
        self.assertEqual(neg_ei.shape, (2, 5))
        self.assertEqual(neg_et.shape, (5,))

    def test_invalid_negative_sampler_args(self):
        with self.assertRaises(ValueError):
            TypedNegativeSampler(self.edge_index, self.edge_type, num_nodes=1)

        sampler = TypedNegativeSampler(self.edge_index, self.edge_type, num_nodes=10)
        with self.assertRaises(ValueError):
            sampler.sample(self.edge_index, self.edge_type, num_neg_per_pos=0)

    def test_negative_sampling_avoids_positives(self):
        sampler = TypedNegativeSampler(
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            num_nodes=self.num_nodes,
            node_df=self.node_df,
            seed=42,
        )
        neg_ei, neg_et = sampler.sample(
            self.edge_index,
            self.edge_type,
            num_neg_per_pos=3,
            type_constrained=True,
        )
        for s, r, t in zip(neg_ei[0].tolist(), neg_et.tolist(), neg_ei[1].tolist()):
            self.assertNotIn((int(s), int(r), int(t)), sampler.known_triples)

    def test_negative_sampling_small_graph(self):
        # 3 nodes, 2 edges
        small_ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        small_et = torch.tensor([0, 0], dtype=torch.long)
        sampler = TypedNegativeSampler(
            edge_index=small_ei,
            edge_type=small_et,
            num_nodes=3,
            seed=123,
        )
        neg_ei, neg_et = sampler.sample(small_ei, small_et, num_neg_per_pos=1)
        self.assertEqual(neg_ei.shape, (2, 2))
        self.assertEqual(neg_et.shape, (2,))

    def test_negative_sampling_reproducibility(self):
        sampler1 = TypedNegativeSampler(self.edge_index, self.edge_type, num_nodes=self.num_nodes, seed=42)
        sampler2 = TypedNegativeSampler(self.edge_index, self.edge_type, num_nodes=self.num_nodes, seed=42)
        neg_ei1, neg_et1 = sampler1.sample(self.edge_index, self.edge_type)
        neg_ei2, neg_et2 = sampler2.sample(self.edge_index, self.edge_type)
        self.assertTrue(torch.equal(neg_ei1, neg_ei2))
        self.assertTrue(torch.equal(neg_et1, neg_et2))


class TestEvaluatorAndBaseline(unittest.TestCase):
    """Test evaluation logic and DistMult baseline."""

    def setUp(self):
        self.num_nodes = 8
        self.num_relations = 2
        self.embedding_dim = 16

        # Construct toy graph
        self.train_ei = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
        self.train_et = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        self.test_ei = torch.tensor([[0, 2], [5, 7]], dtype=torch.long)
        self.test_et = torch.tensor([0, 1], dtype=torch.long)

        encoder = RGCNEncoder(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            in_channels=self.embedding_dim,
            hidden_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            num_layers=2,
            num_bases=2,
            dropout=0.0,
        )
        decoder = DistMultDecoder(
            num_relations=self.num_relations,
            embedding_dim=self.embedding_dim,
        )
        self.rgcn_model = MediGraphModel(encoder, decoder)
        self.rgcn_model.eval()

    def test_evaluate_link_prediction(self):
        metrics = evaluate_link_prediction(
            model=self.rgcn_model,
            eval_edge_index=self.test_ei,
            eval_edge_type=self.test_et,
            mp_edge_index=self.train_ei,
            mp_edge_type=self.train_et,
            num_nodes=self.num_nodes,
            device="cpu",
        )
        self.assertIn("mrr", metrics)
        self.assertIn("hits@1", metrics)
        self.assertIn("hits@3", metrics)
        self.assertIn("hits@10", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("f1", metrics)
        self.assertTrue(0.0 <= metrics["mrr"] <= 1.0)

    def test_distmult_baseline_forward(self):
        baseline = DistMultBaseline(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            embedding_dim=self.embedding_dim,
        )
        z = baseline.encode()
        self.assertEqual(z.shape, (self.num_nodes, self.embedding_dim))
        scores = baseline(self.train_ei, self.train_et)
        self.assertEqual(scores.shape, (4,))

    def test_train_baseline_and_compare(self):
        baseline = train_baseline(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            train_edge_index=self.train_ei,
            train_edge_type=self.train_et,
            embedding_dim=self.embedding_dim,
            epochs=5,
            device="cpu",
        )
        self.assertIsInstance(baseline, DistMultBaseline)

        comparison = compare_rgcn_and_baseline(
            rgcn_model=self.rgcn_model,
            baseline_model=baseline,
            test_edge_index=self.test_ei,
            test_edge_type=self.test_et,
            train_edge_index=self.train_ei,
            train_edge_type=self.train_et,
            num_nodes=self.num_nodes,
            device="cpu",
        )
        self.assertIn("rgcn", comparison)
        self.assertIn("baseline_distmult", comparison)
        self.assertIn("delta_mrr", comparison)


class TestInference(unittest.TestCase):
    """Test link prediction and inference APIs."""

    def setUp(self):
        self.num_nodes = 5
        self.num_relations = 2
        self.embedding_dim = 8

        # 4 nodes connected, 1 node disconnected (node 4)
        self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        self.edge_type = torch.tensor([0, 0, 1], dtype=torch.long)

        self.node_df = pd.DataFrame({
            "node_id": [0, 1, 2, 3, 4],
            "entity_name": ["Disease_A", "Gene_B", "Drug_C", "Gene_D", "Isolated_E"],
            "entity_type": ["disease", "gene", "drug", "gene", "phenotype"],
            "source_id": ["D001", "G001", "C001", "G002", "P001"],
        })
        self.node_mapping = {
            "Disease_A": 0,
            "Gene_B": 1,
            "Drug_C": 2,
            "Gene_D": 3,
            "Isolated_E": 4,
        }
        self.rel_mapping = {
            "disease_protein": 0,
            "drug_protein": 1,
        }

        encoder = RGCNEncoder(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            in_channels=self.embedding_dim,
            hidden_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            num_layers=2,
            num_bases=2,
            dropout=0.0,
        )
        decoder = DistMultDecoder(
            num_relations=self.num_relations,
            embedding_dim=self.embedding_dim,
        )
        self.model = MediGraphModel(encoder, decoder)
        self.model.eval()

    def test_predict_links_success(self):
        preds = predict_links(
            query_node="Disease_A",
            relation_type="disease_protein",
            model=self.model,
            node_df=self.node_df,
            node_mapping=self.node_mapping,
            rel_mapping=self.rel_mapping,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            top_k=3,
            score_threshold=-1e5,
        )
        self.assertIsInstance(preds, list)
        self.assertGreaterEqual(len(preds), 1)
        first = preds[0]
        self.assertIn("rank", first)
        self.assertIn("target_node_id", first)
        self.assertIn("target_name", first)
        self.assertIn("target_type", first)
        self.assertIn("relation", first)
        self.assertIn("score", first)
        self.assertIn("is_known_edge", first)
        # Check self-prediction is excluded
        target_ids = [p["target_node_id"] for p in preds]
        self.assertNotIn(0, target_ids, "Self-loop should be excluded from predictions")

    def test_predict_links_known_edge_flag(self):
        # Disease_A (0) -> Gene_B (1) with rel 0 is known
        preds = predict_links(
            query_node="Disease_A",
            relation_type="disease_protein",
            model=self.model,
            node_df=self.node_df,
            node_mapping=self.node_mapping,
            rel_mapping=self.rel_mapping,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            top_k=4,
            score_threshold=-1e5,
        )
        for p in preds:
            if p["target_node_id"] == 1:
                self.assertTrue(p["is_known_edge"])
            else:
                self.assertFalse(p["is_known_edge"])

    def test_predict_links_unknown_entity(self):
        with self.assertRaises(KeyError):
            predict_links(
                query_node="NonExistentEntity",
                relation_type="disease_protein",
                model=self.model,
                node_df=self.node_df,
                node_mapping=self.node_mapping,
                rel_mapping=self.rel_mapping,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )

    def test_predict_links_unknown_relation(self):
        with self.assertRaises(KeyError):
            predict_links(
                query_node="Disease_A",
                relation_type="non_existent_relation",
                model=self.model,
                node_df=self.node_df,
                node_mapping=self.node_mapping,
                rel_mapping=self.rel_mapping,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )

    def test_predict_links_empty_inputs(self):
        with self.assertRaises(ValueError):
            predict_links(
                query_node="",
                relation_type="disease_protein",
                model=self.model,
                node_df=self.node_df,
                node_mapping=self.node_mapping,
                rel_mapping=self.rel_mapping,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )
        with self.assertRaises(ValueError):
            predict_links(
                query_node="Disease_A",
                relation_type="   ",
                model=self.model,
                node_df=self.node_df,
                node_mapping=self.node_mapping,
                rel_mapping=self.rel_mapping,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )

    def test_predict_links_invalid_top_k(self):
        with self.assertRaises(ValueError):
            predict_links(
                query_node="Disease_A",
                relation_type="disease_protein",
                model=self.model,
                node_df=self.node_df,
                node_mapping=self.node_mapping,
                rel_mapping=self.rel_mapping,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                top_k=0,
            )

    def test_predict_links_model_training_mode(self):
        self.model.train()
        with self.assertRaises(ValueError):
            predict_links(
                query_node="Disease_A",
                relation_type="disease_protein",
                model=self.model,
                node_df=self.node_df,
                node_mapping=self.node_mapping,
                rel_mapping=self.rel_mapping,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )
        self.model.eval()

    def test_predict_links_disconnected_node(self):
        # Isolated_E (node 4) has no edges in edge_index
        preds = predict_links(
            query_node="Isolated_E",
            relation_type="disease_protein",
            model=self.model,
            node_df=self.node_df,
            node_mapping=self.node_mapping,
            rel_mapping=self.rel_mapping,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            top_k=3,
            score_threshold=-1e5,
        )
        self.assertGreaterEqual(len(preds), 1)
        self.assertNotIn(4, [p["target_node_id"] for p in preds])

    def test_get_node_neighborhood(self):
        rel_inv = {0: "disease_protein", 1: "drug_protein"}
        nh = get_node_neighborhood(
            node_id=1,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            node_df=self.node_df,
            rel_mapping_inv=rel_inv,
        )
        self.assertEqual(nh["center"]["node_id"], 1)
        self.assertEqual(nh["center"]["entity_name"], "Gene_B")
        self.assertEqual(len(nh["neighbors"]), 2)


class TestTrainingIntegration(unittest.TestCase):
    """Test training loop execution and checkpoint persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_reload_checkpoint(self):
        ckpt_path = Path(self.temp_dir) / "test_model.pt"
        num_nodes = 6
        num_rels = 2
        encoder = RGCNEncoder(num_nodes=num_nodes, num_relations=num_rels, in_channels=8, hidden_channels=8, out_channels=8, num_layers=2)
        decoder = DistMultDecoder(num_relations=num_rels, embedding_dim=8)
        model = MediGraphModel(encoder, decoder)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        _save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_mrr=0.8,
            val_hits_at_10=1.0,
            model_cfg={"in_channels": 8, "hidden_channels": 8, "out_channels": 8, "num_layers": 2, "num_bases": 2, "dropout": 0.0},
            num_nodes=num_nodes,
            num_relations=num_rels,
            seed=42,
        )

        self.assertTrue(ckpt_path.exists())
        loaded_model, meta = MediGraphModel.load_checkpoint(str(ckpt_path))
        self.assertEqual(meta["epoch"], 1)
        self.assertEqual(meta["model_config"]["num_nodes"], num_nodes)
        self.assertEqual(meta["model_config"]["num_relations"], num_rels)

    def test_short_training_loop(self):
        cfg = {
            "project": {"name": "test_run", "seed": 42},
            "paths": {
                "nodes_csv": "data/processed/nodes.csv",
                "edges_csv": "data/processed/edges.csv",
                "node_mapping_json": "data/processed/node_mapping.json",
                "relation_mapping_json": "data/processed/relation_mapping.json",
                "checkpoints": str(Path(self.temp_dir) / "checkpoints"),
                "metrics": str(Path(self.temp_dir) / "metrics"),
            },
            "model": {
                "in_channels": 8,
                "hidden_channels": 8,
                "out_channels": 8,
                "num_layers": 2,
                "num_bases": 2,
                "dropout": 0.0,
            },
            "training": {
                "epochs": 2,
                "val_interval": 1,
                "learning_rate": 0.01,
                "weight_decay": 1e-4,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "negative_sampling_ratio": 1,
                "batch_size": 16,
            },
            "_device_override": "cpu",
        }
        train(cfg)

        best_ckpt = Path(self.temp_dir) / "checkpoints" / "best_model.pt"
        self.assertTrue(best_ckpt.exists(), "best_model.pt was not saved")

        loaded_model, meta = MediGraphModel.load_checkpoint(str(best_ckpt))
        self.assertIsInstance(loaded_model, MediGraphModel)
        self.assertIn(meta["epoch"], [1, 2])

        metrics_file = Path(self.temp_dir) / "metrics" / "metrics.json"
        self.assertTrue(metrics_file.exists(), "metrics.json was not saved")
        with open(metrics_file, "r") as fh:
            data = json.load(fh)
        self.assertIn("best", data)
        self.assertIn("test_mrr", data["best"])


if __name__ == "__main__":
    unittest.main()
