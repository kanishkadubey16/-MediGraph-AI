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
from src.training.splitter import split_edges
from src.training.evaluator import LinkPredictionEvaluator
from src.training.evaluate import evaluate_link_prediction
from src.training.baseline import DistMultBaseline, train_baseline, compare_rgcn_and_baseline
from src.training.train import train, _save_checkpoint
from src.utils.metrics import (
    compute_mrr,
    compute_hits_at_k,
    compute_roc_auc,
    compute_classification_metrics,
)


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


if __name__ == "__main__":
    unittest.main()
