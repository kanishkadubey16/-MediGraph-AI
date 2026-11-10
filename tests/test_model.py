"""
tests/test_model.py
-------------------
Unit and integration test suite for the MediGraph AI GNN architecture.

Tests:
  1. Node and relation embeddings initialization and forward lookup.
  2. MedicalRGCN / RGCNEncoder initialization and forward propagation.
  3. Output tensor shapes across multiple layers and channel dimensions.
  4. Proper handling of relation IDs and edge indices.
  5. DistMult decoder scoring (positive, negative, candidate ranking).
  6. End-to-end backpropagation and gradient flow.
  7. Model checkpoint serialization and reloading.
  8. Device handling (CPU execution).

Author: Member 2 — Graph Neural Network Engineer
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from src.models.embeddings import (
    NodeEmbedding,
    RelationEmbedding,
    BiomedicalEmbeddings,
)
from src.models.rgcn import RGCNEncoder, MedicalRGCN
from src.models.decoder import (
    DistMultDecoder,
    MedicalDistMultDecoder,
    MediGraphModel,
    MedicalRGCNModel,
)


class TestGNNArchitecture(unittest.TestCase):
    """Test suite for GNN architecture and DistMult decoder."""

    def setUp(self) -> None:
        """Create a tiny synthetic graph for testing."""
        torch.manual_seed(42)

        self.num_nodes = 12
        self.num_relations = 4
        self.in_channels = 16
        self.hidden_channels = 32
        self.out_channels = 16

        # Synthetic edge index (2, E)
        self.edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 0, 2],
            [1, 2, 0, 4, 5, 3, 3, 5],
        ], dtype=torch.long)

        # Synthetic edge types (E,) with values in [0, num_relations)
        self.edge_type = torch.tensor([0, 1, 2, 0, 1, 3, 2, 0], dtype=torch.long)

        self.encoder = MedicalRGCN(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=2,
            num_bases=4,
            dropout=0.1,
        )

        self.decoder = MedicalDistMultDecoder(
            num_relations=self.num_relations,
            embedding_dim=self.out_channels,
        )

        self.model = MedicalRGCNModel(self.encoder, self.decoder)

    # ----------------------------------------------------------------------- #
    # 1. Embeddings Tests
    # ----------------------------------------------------------------------- #

    def test_embeddings_initialization_and_forward(self) -> None:
        """Verify NodeEmbedding and RelationEmbedding shapes and lookups."""
        node_emb = NodeEmbedding(num_nodes=10, embedding_dim=24)
        rel_emb = RelationEmbedding(num_relations=5, embedding_dim=24)

        # Full table lookup (None argument)
        all_nodes = node_emb()
        self.assertEqual(all_nodes.shape, (10, 24))
        self.assertEqual(node_emb.weight.shape, (10, 24))

        all_rels = rel_emb()
        self.assertEqual(all_rels.shape, (5, 24))
        self.assertEqual(rel_emb.weight.shape, (5, 24))

        # Indexed batch lookup
        indices = torch.tensor([0, 2, 4], dtype=torch.long)
        batch_emb = node_emb(indices)
        self.assertEqual(batch_emb.shape, (3, 24))

        # Container
        bio_emb = BiomedicalEmbeddings(num_nodes=8, num_relations=3, embedding_dim=16)
        self.assertEqual(bio_emb.node_embeddings.weight.shape, (8, 16))
        self.assertEqual(bio_emb.relation_embeddings.weight.shape, (3, 16))

    def test_embeddings_invalid_dims_raise(self) -> None:
        """Verify that non-positive dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            NodeEmbedding(num_nodes=0, embedding_dim=16)
        with self.assertRaises(ValueError):
            NodeEmbedding(num_nodes=10, embedding_dim=-4)
        with self.assertRaises(ValueError):
            RelationEmbedding(num_relations=-1, embedding_dim=16)

    # ----------------------------------------------------------------------- #
    # 2. R-GCN Encoder Tests
    # ----------------------------------------------------------------------- #

    def test_rgcn_encoder_forward_shape(self) -> None:
        """Verify R-GCN forward output shape is (num_nodes, out_channels)."""
        z = self.encoder(self.edge_index, self.edge_type)

        self.assertIsInstance(z, Tensor)
        self.assertEqual(z.shape, (self.num_nodes, self.out_channels))
        self.assertEqual(z.dtype, torch.float32)

    def test_rgcn_alias_compatibility(self) -> None:
        """Verify MedicalRGCN is an alias/subclass of RGCNEncoder."""
        self.assertTrue(issubclass(MedicalRGCN, RGCNEncoder) or MedicalRGCN is RGCNEncoder)
        encoder_alt = RGCNEncoder(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
        )
        z = encoder_alt(self.edge_index, self.edge_type)
        self.assertEqual(z.shape, (self.num_nodes, self.out_channels))

    def test_rgcn_input_validation(self) -> None:
        """Verify invalid edge_index or edge_type tensor shapes raise ValueError."""
        # edge_index with wrong shape (3, E)
        bad_edge_index = torch.zeros((3, 5), dtype=torch.long)
        bad_edge_type = torch.zeros(5, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.encoder(bad_edge_index, bad_edge_type)

        # edge_type length mismatch
        mismatched_edge_type = torch.zeros(3, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.encoder(self.edge_index, mismatched_edge_type)

        # wrong dtype
        float_edge_index = self.edge_index.float()
        with self.assertRaises(ValueError):
            self.encoder(float_edge_index, self.edge_type)

    def test_rgcn_inference_mode_wrapper(self) -> None:
        """Verify get_embeddings() runs in eval mode without gradients."""
        self.encoder.train()
        z = self.encoder.get_embeddings(self.edge_index, self.edge_type)
        self.assertEqual(z.shape, (self.num_nodes, self.out_channels))
        self.assertFalse(z.requires_grad)
        # Verify training state was restored
        self.assertTrue(self.encoder.training)

    # ----------------------------------------------------------------------- #
    # 3. DistMult Decoder Tests
    # ----------------------------------------------------------------------- #

    def test_distmult_decoder_scoring(self) -> None:
        """Verify DistMult bilinear scoring calculation on triple batches."""
        z = self.encoder(self.edge_index, self.edge_type)

        cand_edges = torch.tensor([
            [0, 1, 2],
            [1, 2, 0],
        ], dtype=torch.long)
        cand_rel = torch.tensor([0, 1, 2], dtype=torch.long)

        scores = self.decoder(z, cand_edges, cand_rel)

        self.assertEqual(scores.shape, (3,))
        self.assertEqual(scores.dtype, torch.float32)

        # Manually compute expected DistMult score for triple (0, 0, 1)
        h_emb = z[0]
        r_emb = self.decoder.rel_weight[0]
        t_emb = z[1]
        expected_score_0 = (h_emb * r_emb * t_emb).sum().item()
        self.assertAlmostEqual(scores[0].item(), expected_score_0, places=5)

    def test_distmult_candidate_tail_ranking(self) -> None:
        """Verify score_candidates computes scores for all nodes in the graph."""
        z = self.encoder(self.edge_index, self.edge_type)

        head_idx = 3
        relation_idx = 1
        scores = self.decoder.score_candidates(head_idx, relation_idx, z)

        self.assertEqual(scores.shape, (self.num_nodes,))
        # Check individual computation matches
        for target_idx in range(self.num_nodes):
            expected = (z[head_idx] * self.decoder.rel_weight[relation_idx] * z[target_idx]).sum().item()
            self.assertAlmostEqual(scores[target_idx].item(), expected, places=5)

    def test_distmult_score_triples_helper(self) -> None:
        """Verify score_triples helper matches forward scoring."""
        z = self.encoder(self.edge_index, self.edge_type)

        heads = torch.tensor([0, 2], dtype=torch.long)
        rels = torch.tensor([1, 0], dtype=torch.long)
        tails = torch.tensor([3, 1], dtype=torch.long)

        scores = self.decoder.score_triples(heads, rels, tails, z)
        self.assertEqual(scores.shape, (2,))

    # ----------------------------------------------------------------------- #
    # 4. Backpropagation & Gradient Flow
    # ----------------------------------------------------------------------- #

    def test_backpropagation_and_parameter_update(self) -> None:
        """Verify backpropagation produces valid gradients and updates parameters."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Positive edges (from graph)
        pos_edges = self.edge_index[:, :4]
        pos_rels = self.edge_type[:4]

        # Negative edges (random corrupted tails)
        neg_edges = torch.tensor([
            [0, 1, 2, 3],
            [5, 4, 3, 2],
        ], dtype=torch.long)
        neg_rels = pos_rels.clone()

        # Forward
        z = self.model.encode(self.edge_index, self.edge_type)
        pos_scores = self.model.decode(z, pos_edges, pos_rels)
        neg_scores = self.model.decode(z, neg_edges, neg_rels)

        # BCE with logits: positive -> 1, negative -> 0
        criterion = nn.BCEWithLogitsLoss()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        all_scores = torch.cat([pos_scores, neg_scores])
        loss = criterion(all_scores, labels)

        self.assertTrue(torch.isfinite(loss))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check encoder and decoder parameters have gradients
        self.assertIsNotNone(self.encoder.node_emb.weight.grad)
        self.assertIsNotNone(self.decoder.rel_weight.grad)

        for param in self.encoder.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

        # Check step updates weights
        old_weight = self.encoder.node_emb.weight.clone()
        optimizer.step()
        self.assertFalse(torch.equal(old_weight, self.encoder.node_emb.weight))

    # ----------------------------------------------------------------------- #
    # 5. Model Checkpoint Serialization & Reloading
    # ----------------------------------------------------------------------- #

    def test_checkpoint_save_and_load(self) -> None:
        """Verify model checkpoint saving and loading maintains exact weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            metrics = {
                "train_loss": 0.4521,
                "val_loss": 0.5102,
                "val_mrr": 0.3245,
                "val_hits_at_10": 0.6500,
            }

            # Save checkpoint
            self.model.save_checkpoint(
                path=str(ckpt_path),
                epoch=15,
                optimizer=optimizer,
                metrics=metrics,
            )
            self.assertTrue(ckpt_path.exists())

            # Load checkpoint
            loaded_model, ckpt_dict = MediGraphModel.load_checkpoint(
                str(ckpt_path), map_location="cpu"
            )

            # Check metadata
            self.assertEqual(ckpt_dict["epoch"], 15)
            self.assertAlmostEqual(ckpt_dict["val_mrr"], 0.3245, places=4)
            self.assertFalse(loaded_model.training)  # Loaded model must be in eval mode

            # Check weights match exactly in eval mode
            self.model.eval()
            z_orig = self.model.encode(self.edge_index, self.edge_type)
            z_loaded = loaded_model.encode(self.edge_index, self.edge_type)
            self.assertTrue(torch.allclose(z_orig, z_loaded, atol=1e-6))



if __name__ == "__main__":
    unittest.main()
