"""
src/training/baseline.py
------------------------
DistMult baseline model (without R-GCN message passing) and model comparison.

Provides:
- DistMultBaseline: Knowledge graph embedding model using direct node and relation
  lookup embeddings with a DistMult bilinear decoder, omitting R-GCN message passing.
- train_baseline(): Trains the DistMult baseline using identical splits and negative sampling.
- compare_rgcn_and_baseline(): Side-by-side comparison of R-GCN vs DistMult baseline
  on the same test dataset, demonstrating the empirical advantage of graph neural message passing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Bootstrap sys.path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.decoder import DistMultDecoder, MediGraphModel
from src.data.loader import MediGraphLoader
from src.data.graph_builder import GraphBuilder
from src.training.splitter import split_edges
from src.training.negative_sampling import TypedNegativeSampler, corrupt_triples
from src.training.evaluate import evaluate_link_prediction
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class DistMultBaseline(nn.Module):
    """
    Standard DistMult Knowledge Graph Embedding baseline without GNN message passing.

    Node embeddings are learned directly as free lookup parameters.
    Score function: score(h, r, t) = <e_h, r_rel, e_t>
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        self.decoder = DistMultDecoder(num_relations, embedding_dim)

    def encode(self, edge_index: Optional[Tensor] = None, edge_type: Optional[Tensor] = None) -> Tensor:
        """Return node embeddings directly (ignoring edges, no message passing)."""
        return self.node_embeddings.weight

    def decode(self, z: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Compute scores for given triples using DistMult bilinear product."""
        return self.decoder(z, edge_index, edge_type)

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        z = self.encode()
        return self.decode(z, edge_index, edge_type)


def train_baseline(
    num_nodes: int,
    num_relations: int,
    train_edge_index: Tensor,
    train_edge_type: Tensor,
    embedding_dim: int = 64,
    epochs: int = 60,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seed: int = 42,
) -> DistMultBaseline:
    """
    Train a DistMult baseline model on the training edges.
    """
    torch.manual_seed(seed)
    device_obj = torch.device(device)

    model = DistMultBaseline(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
    ).to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_ei = train_edge_index.to(device_obj)
    train_et = train_edge_type.to(device_obj)

    log.info("Training DistMult baseline for %d epochs...", epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        neg_ei, neg_et = corrupt_triples(
            train_ei,
            train_et,
            num_nodes=num_nodes,
            num_neg_per_pos=1,
            seed=seed + epoch,
        )

        z = model.encode()
        pos_scores = model.decode(z, train_ei, train_et)
        neg_scores = model.decode(z, neg_ei, neg_et)

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    log.info("DistMult baseline training complete.")
    return model


def compare_rgcn_and_baseline(
    rgcn_model: MediGraphModel,
    baseline_model: DistMultBaseline,
    test_edge_index: Tensor,
    test_edge_type: Tensor,
    train_edge_index: Tensor,
    train_edge_type: Tensor,
    num_nodes: int,
    all_edge_index: Optional[Tensor] = None,
    all_edge_type: Optional[Tensor] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate and compare R-GCN against the DistMult baseline on identical test edges.

    Returns
    -------
    Dict[str, Any]
        Dictionary comparing metrics side-by-side.
    """
    rgcn_model.eval()
    baseline_model.eval()

    rgcn_metrics = evaluate_link_prediction(
        model=rgcn_model,
        eval_edge_index=test_edge_index,
        eval_edge_type=test_edge_type,
        mp_edge_index=train_edge_index,
        mp_edge_type=train_edge_type,
        num_nodes=num_nodes,
        all_edge_index=all_edge_index,
        all_edge_type=all_edge_type,
        device=device,
    )

    baseline_metrics = evaluate_link_prediction(
        model=baseline_model,
        eval_edge_index=test_edge_index,
        eval_edge_type=test_edge_type,
        mp_edge_index=train_edge_index,
        mp_edge_type=train_edge_type,
        num_nodes=num_nodes,
        all_edge_index=all_edge_index,
        all_edge_type=all_edge_type,
        device=device,
    )

    comparison = {
        "rgcn": rgcn_metrics,
        "baseline_distmult": baseline_metrics,
        "delta_mrr": round(rgcn_metrics["mrr"] - baseline_metrics["mrr"], 6),
        "delta_roc_auc": round(rgcn_metrics["roc_auc"] - baseline_metrics["roc_auc"], 6),
    }

    return comparison


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print clean comparison table in the terminal."""
    rgcn = comparison["rgcn"]
    base = comparison["baseline_distmult"]

    print("\n" + "=" * 70)
    print("      MEDIGRAPH AI — MODEL COMPARISON: R-GCN vs DISTMULT BASELINE")
    print("=" * 70)
    print(f"{'Metric':<20} {'DistMult (Baseline)':<22} {'R-GCN (Proposed)':<20} {'Delta':<10}")
    print("-" * 70)

    for metric_name in ["mrr", "hits@1", "hits@3", "hits@10", "roc_auc", "precision", "recall", "f1"]:
        b_val = base.get(metric_name, 0.0)
        r_val = rgcn.get(metric_name, 0.0)
        delta = r_val - b_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{metric_name.upper():<20} {b_val:<22.4f} {r_val:<20.4f} {delta_str:<10}")

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MediGraph R-GCN with DistMult baseline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_model.pt", help="R-GCN checkpoint")
    parser.add_argument("--epochs", type=int, default=60, help="Baseline training epochs")
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    parser.add_argument("--output", default="artifacts/metrics/baseline_comparison.json", help="Output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        print("Please train R-GCN first with: python src/training/train.py")
        sys.exit(1)

    loader = MediGraphLoader()
    node_df, edge_df, node_mapping, rel_mapping = loader.load_all(cfg)
    builder = GraphBuilder(node_df, edge_df, rel_mapping)
    edge_index, edge_type = builder.build_pyg_tensors(edge_df, rel_mapping)

    num_nodes = len(node_df)
    num_relations = len(rel_mapping)

    splits = split_edges(
        edge_index=edge_index,
        edge_type=edge_type,
        train_ratio=float(cfg.get("training", {}).get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("training", {}).get("val_ratio", 0.1)),
        test_ratio=float(cfg.get("training", {}).get("test_ratio", 0.1)),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )

    rgcn_model, _ = MediGraphModel.load_checkpoint(str(ckpt_path), map_location=args.device)
    rgcn_model.to(args.device)
    rgcn_model.eval()

    baseline_model = train_baseline(
        num_nodes=num_nodes,
        num_relations=num_relations,
        train_edge_index=splits["train"]["edge_index"],
        train_edge_type=splits["train"]["edge_type"],
        embedding_dim=int(cfg.get("model", {}).get("out_channels", 64)),
        epochs=args.epochs,
        device=args.device,
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )

    comparison = compare_rgcn_and_baseline(
        rgcn_model=rgcn_model,
        baseline_model=baseline_model,
        test_edge_index=splits["test"]["edge_index"],
        test_edge_type=splits["test"]["edge_type"],
        train_edge_index=splits["train"]["edge_index"],
        train_edge_type=splits["train"]["edge_type"],
        num_nodes=num_nodes,
        all_edge_index=edge_index,
        all_edge_type=edge_type,
        device=args.device,
    )

    print_comparison_table(comparison)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: {out_p}")


if __name__ == "__main__":
    main()
