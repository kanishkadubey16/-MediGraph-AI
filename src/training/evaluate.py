"""
src/training/evaluate.py
------------------------
Evaluation pipeline and CLI for MediGraph AI link prediction.

Computes:
- Filtered Mean Reciprocal Rank (MRR)
- Hits@1, Hits@3, Hits@10, Hits@50
- Precision, Recall, F1-score, and Accuracy on balanced test sets
- Receiver Operating Characteristic Area Under Curve (ROC-AUC)
- Per-relation breakdown of evaluation metrics

Can be executed as a standalone CLI script:
    python src/training/evaluate.py --checkpoint artifacts/checkpoints/best_model.pt

Or called programmatically:
    from src.training.evaluate import evaluate_model
    metrics = evaluate_model(model, test_edge_index, test_edge_type, train_edge_index, train_edge_type, num_nodes)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Bootstrap sys.path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch import Tensor

from src.models.decoder import MediGraphModel
from src.data.loader import MediGraphLoader
from src.data.graph_builder import GraphBuilder
from src.training.splitter import split_edges
from src.training.negative_sampling import TypedNegativeSampler, corrupt_triples
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.metrics import (
    compute_mrr,
    compute_hits_at_k,
    compute_roc_auc,
    compute_classification_metrics,
)

log = get_logger(__name__)


def evaluate_link_prediction(
    model: MediGraphModel,
    eval_edge_index: Tensor,
    eval_edge_type: Tensor,
    mp_edge_index: Tensor,
    mp_edge_type: Tensor,
    num_nodes: int,
    all_edge_index: Optional[Tensor] = None,
    all_edge_type: Optional[Tensor] = None,
    hits_k: List[int] = [1, 3, 10],
    device: str = "cpu",
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Perform filtered link prediction evaluation for tail entity ranking.

    For each test triple (h, r, t):
      1. Score (h, r, t') for all candidate nodes t' in [0, num_nodes).
      2. Filter out all other known true tails for the same (h, r) query.
      3. Compute 1-based rank of true tail t.
      4. Compute MRR, Hits@1, Hits@3, Hits@10.

    Parameters
    ----------
    model : MediGraphModel
        Trained model in eval mode.
    eval_edge_index : Tensor
        (2, E_eval) positive test edges.
    eval_edge_type : Tensor
        (E_eval,) positive test relation types.
    mp_edge_index : Tensor
        (2, E_train) message-passing graph edges.
    mp_edge_type : Tensor
        (E_train,) message-passing relation types.
    num_nodes : int
        Total number of nodes.
    all_edge_index : Optional[Tensor]
        All known edges (train + val + test) for filtered ranking.
    all_edge_type : Optional[Tensor]
        All known edge relation types.
    hits_k : List[int]
        List of k cutoffs (default [1, 3, 10]).
    device : str
        Device to run evaluation on.
    batch_size : int
        Batch size for candidate scoring.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing overall ranking metrics, classification metrics,
        and per-relation breakdowns.
    """
    if model.training:
        raise ValueError("[MediGraph] Model must be in eval() mode for evaluation.")

    eval_edge_index = eval_edge_index.to(device)
    eval_edge_type = eval_edge_type.to(device)
    mp_edge_index = mp_edge_index.to(device)
    mp_edge_type = mp_edge_type.to(device)

    # Build filtered dictionary: (h, r) -> set of known tails
    if all_edge_index is None:
        all_edge_index = torch.cat([mp_edge_index, eval_edge_index], dim=1)
        all_edge_type = torch.cat([mp_edge_type, eval_edge_type], dim=0)
    else:
        all_edge_index = all_edge_index.to(device)
        all_edge_type = all_edge_type.to(device)

    known_tails_map: Dict[Tuple[int, int], Set[int]] = {}
    srcs = all_edge_index[0].cpu().tolist()
    tgts = all_edge_index[1].cpu().tolist()
    rels = all_edge_type.cpu().tolist()
    for s, r, t in zip(srcs, rels, tgts):
        key = (int(s), int(r))
        if key not in known_tails_map:
            known_tails_map[key] = set()
        known_tails_map[key].add(int(t))

    # Obtain node embeddings from message passing on mp_edge_index
    with torch.no_grad():
        z = model.encode(mp_edge_index, mp_edge_type)

    all_nodes = torch.arange(num_nodes, device=device)
    E_eval = eval_edge_index.shape[1]
    all_ranks: List[int] = []
    rel_ranks: Dict[int, List[int]] = {}

    eval_srcs = eval_edge_index[0].cpu().tolist()
    eval_tgts = eval_edge_index[1].cpu().tolist()
    eval_rels = eval_edge_type.cpu().tolist()

    with torch.no_grad():
        for i in range(E_eval):
            h = int(eval_srcs[i])
            t = int(eval_tgts[i])
            r = int(eval_rels[i])

            cand_heads = torch.full((num_nodes,), h, dtype=torch.long, device=device)
            cand_ei = torch.stack([cand_heads, all_nodes], dim=0)
            cand_et = torch.full((num_nodes,), r, dtype=torch.long, device=device)

            scores = model.decode(z, cand_ei, cand_et).clone()

            # Filter setting: mask other known true tails
            known_t = known_tails_map.get((h, r), set())
            for other_t in known_t:
                if other_t != t:
                    scores[other_t] = float("-inf")

            true_score = scores[t].item()
            # 1-based rank: count candidates with score > true_score + 1
            rank = int((scores > true_score).sum().item()) + 1
            all_ranks.append(rank)

            if r not in rel_ranks:
                rel_ranks[r] = []
            rel_ranks[r].append(rank)

    # Compute ranking metrics
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    mrr = compute_mrr(ranks_arr)
    hits_dict = {f"hits@{k}": compute_hits_at_k(ranks_arr, k) for k in hits_k}

    # Binary classification metrics on positive + negative evaluation edges
    with torch.no_grad():
        pos_scores = model.decode(z, eval_edge_index, eval_edge_type)
        neg_ei, neg_et = corrupt_triples(
            eval_edge_index,
            eval_edge_type,
            num_nodes=num_nodes,
            num_neg_per_pos=1,
            seed=42,
        )
        neg_scores = model.decode(z, neg_ei, neg_et)

    roc_auc = compute_roc_auc(pos_scores, neg_scores)
    clf_metrics = compute_classification_metrics(pos_scores, neg_scores)

    # Per-relation breakdown
    per_relation: Dict[int, Dict[str, float]] = {}
    for r, r_ranks in rel_ranks.items():
        r_arr = np.array(r_ranks, dtype=np.float64)
        per_relation[r] = {
            "count": len(r_ranks),
            "mrr": compute_mrr(r_arr),
            **{f"hits@{k}": compute_hits_at_k(r_arr, k) for k in hits_k},
        }

    return {
        "mrr": round(float(mrr), 6),
        **{k: round(float(v), 6) for k, v in hits_dict.items()},
        "roc_auc": round(float(roc_auc), 6),
        "precision": round(float(clf_metrics["precision"]), 6),
        "recall": round(float(clf_metrics["recall"]), 6),
        "f1": round(float(clf_metrics["f1"]), 6),
        "accuracy": round(float(clf_metrics["accuracy"]), 6),
        "per_relation": per_relation,
    }


def evaluate_model(
    model: MediGraphModel,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    High-level evaluation function that loads data from config and runs evaluation.
    """
    if cfg is None:
        cfg = load_config()

    loader = MediGraphLoader()
    node_df, edge_df, node_mapping, rel_mapping = loader.load_all(cfg)
    builder = GraphBuilder(node_df, edge_df, rel_mapping)
    edge_index, edge_type = builder.build_pyg_tensors(edge_df, rel_mapping)

    num_nodes = len(node_df)
    train_cfg = cfg.get("training", {})
    splits = split_edges(
        edge_index=edge_index,
        edge_type=edge_type,
        train_ratio=float(train_cfg.get("train_ratio", 0.8)),
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
        test_ratio=float(train_cfg.get("test_ratio", 0.1)),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )

    model.eval()
    return evaluate_link_prediction(
        model=model,
        eval_edge_index=splits["test"]["edge_index"],
        eval_edge_type=splits["test"]["edge_type"],
        mp_edge_index=splits["train"]["edge_index"],
        mp_edge_type=splits["train"]["edge_type"],
        num_nodes=num_nodes,
        all_edge_index=edge_index,
        all_edge_type=edge_type,
        device=device,
    )


def print_evaluation_report(results: Dict[str, Any], rel_mapping_inv: Optional[Dict[int, str]] = None) -> None:
    """Print formatted markdown evaluation summary report."""
    print("\n" + "=" * 65)
    print("           MEDIGRAPH AI — LINK PREDICTION EVALUATION")
    print("=" * 65)
    print(f"  MRR (Mean Reciprocal Rank): {results['mrr']:.4f}")
    print(f"  Hits@1:                     {results.get('hits@1', 0.0):.4f}")
    print(f"  Hits@3:                     {results.get('hits@3', 0.0):.4f}")
    print(f"  Hits@10:                    {results.get('hits@10', 0.0):.4f}")
    print(f"  ROC-AUC:                    {results['roc_auc']:.4f}")
    print(f"  Precision:                  {results['precision']:.4f}")
    print(f"  Recall:                     {results['recall']:.4f}")
    print(f"  F1 Score:                   {results['f1']:.4f}")
    print(f"  Accuracy:                   {results['accuracy']:.4f}")
    print("-" * 65)

    per_rel = results.get("per_relation", {})
    if per_rel:
        print(f"{'Relation':<30} {'Count':<8} {'MRR':<10} {'Hits@10'}")
        print("-" * 65)
        for r_id, r_metrics in per_rel.items():
            rel_name = rel_mapping_inv.get(int(r_id), f"rel_{r_id}") if rel_mapping_inv else f"rel_{r_id}"
            print(f"{rel_name:<30} {r_metrics['count']:<8} {r_metrics['mrr']:<10.4f} {r_metrics.get('hits@10', 0.0):.4f}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MediGraph AI link prediction model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu or cuda)")
    parser.add_argument("--output", default="artifacts/metrics/eval_metrics.json", help="Output metrics path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        print("Please train the model first with: python src/training/train.py")
        sys.exit(1)

    model, meta = MediGraphModel.load_checkpoint(str(ckpt_path), map_location=args.device)
    model.to(args.device)
    model.eval()

    loader = MediGraphLoader()
    node_df, edge_df, node_mapping, rel_mapping = loader.load_all(cfg)
    rel_mapping_inv = {v: k for k, v in rel_mapping.items()}

    results = evaluate_model(model=model, cfg=cfg, device=args.device)
    print_evaluation_report(results, rel_mapping_inv)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {out_p}")


if __name__ == "__main__":
    main()
