"""
src/inference/predict.py
------------------------
Biomedical link prediction and neighborhood exploration for MediGraph AI.

Exposes:
  - predict_links(): computes candidate scores for all possible target entities
    using the trained R-GCN encoder and DistMult decoder.
  - get_node_neighborhood(): extracts 1-hop / multi-hop subgraphs for visualization.
  - save_predictions(): serializes prediction outputs with timestamp and medical disclaimer.

Author: Member 3 — Training, Evaluation & Inference
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Bootstrap project root into Python module search path
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import torch
from torch import Tensor


from src.models.decoder import MediGraphModel
from src.data.loader import MediGraphLoader
from src.data.graph_builder import GraphBuilder
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)

DISCLAIMER_TEXT = (
    "Scores are raw model outputs and must NOT be interpreted as clinical "
    "probabilities or medical advice. This is a research and educational prototype only."
)


def get_node_neighborhood(
    node_id: int,
    edge_index: Tensor,
    edge_type: Tensor,
    node_df: pd.DataFrame,
    rel_mapping_inv: dict[int, str],
    max_hops: int = 1,
) -> dict[str, Any]:
    """
    Extract the neighborhood of a given node for graph visualization.

    Parameters
    ----------
    node_id : int
        Center node ID.
    edge_index : Tensor
        (2, E) edge index tensor.
    edge_type : Tensor
        (E,) edge relation type tensor.
    node_df : pd.DataFrame
        DataFrame with columns [node_id, entity_name, entity_type, source_id].
    rel_mapping_inv : dict[int, str]
        Inverse relation mapping {relation_id: relation_str}.
    max_hops : int
        Number of hops to traverse (currently 1 hop supported).

    Returns
    -------
    dict[str, Any]
        Schema:
        {
          "center": {"node_id": int, "entity_name": str, "entity_type": str},
          "neighbors": [
            {
              "node_id": int,
              "entity_name": str,
              "entity_type": str,
              "relation": str,
              "direction": "outgoing" | "incoming"
            }, ...
          ]
        }
    """
    node_lookup = node_df.set_index("node_id")
    if node_id not in node_lookup.index:
        raise KeyError(f"[MediGraph] Node ID {node_id} not found in node dataframe.")

    center_row = node_lookup.loc[node_id]
    center_info = {
        "node_id": int(node_id),
        "entity_name": str(center_row["entity_name"]),
        "entity_type": str(center_row["entity_type"]),
    }

    srcs = edge_index[0].cpu().numpy()
    tgts = edge_index[1].cpu().numpy()
    rels = edge_type.cpu().numpy()

    neighbors: list[dict[str, Any]] = []

    # Outgoing edges: center -> target
    out_mask = (srcs == node_id)
    for tgt_id, rel_id in zip(tgts[out_mask], rels[out_mask]):
        tgt_id = int(tgt_id)
        if tgt_id in node_lookup.index:
            row = node_lookup.loc[tgt_id]
            neighbors.append({
                "node_id": tgt_id,
                "entity_name": str(row["entity_name"]),
                "entity_type": str(row["entity_type"]),
                "relation": rel_mapping_inv.get(int(rel_id), f"rel_{rel_id}"),
                "direction": "outgoing",
            })

    # Incoming edges: source -> center
    in_mask = (tgts == node_id)
    for src_id, rel_id in zip(srcs[in_mask], rels[in_mask]):
        src_id = int(src_id)
        if src_id in node_lookup.index:
            row = node_lookup.loc[src_id]
            neighbors.append({
                "node_id": src_id,
                "entity_name": str(row["entity_name"]),
                "entity_type": str(row["entity_type"]),
                "relation": rel_mapping_inv.get(int(rel_id), f"rel_{rel_id}"),
                "direction": "incoming",
            })

    return {
        "center": center_info,
        "neighbors": neighbors,
    }


def predict_links(
    query_node: str,
    relation_type: str,
    model: MediGraphModel,
    node_df: pd.DataFrame,
    node_mapping: dict[str, int],
    rel_mapping: dict[str, int],
    edge_index: Tensor,
    edge_type: Tensor,
    top_k: int = 20,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Predict top-K link candidates for a (query_node, relation_type, ?) triple.

    Parameters
    ----------
    query_node : str
        Entity name of the query head node.
    relation_type : str
        Name of the query relation.
    model : MediGraphModel
        Trained model instance (must be in eval mode).
    node_df : pd.DataFrame
        Nodes dataframe.
    node_mapping : dict[str, int]
        {entity_name: node_id}
    rel_mapping : dict[str, int]
        {relation_string: relation_id}
    edge_index : Tensor
        Message-passing graph edges (2, E).
    edge_type : Tensor
        Message-passing edge relation types (E,).
    top_k : int
        Number of top scoring predictions to return.
    score_threshold : float
        Minimum score threshold.

    Returns
    -------
    list[dict[str, Any]]
        List of ranked predictions sorted by score descending.

    Raises
    ------
    KeyError:
        If query_node or relation_type is unmapped.
    ValueError:
        If query_node or relation_type is empty/invalid, top_k is non-positive,
        or model is in training mode.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"[MediGraph] Expected PyTorch Module for model, got {type(model)}.")

    if model.training:
        raise ValueError(
            "[MediGraph] Model must be in eval() mode before calling predict_links()."
        )

    if not isinstance(query_node, str) or not query_node.strip():
        raise ValueError("[MediGraph] Query entity name must be a non-empty string.")

    if not isinstance(relation_type, str) or not relation_type.strip():
        raise ValueError("[MediGraph] Relation type must be a non-empty string.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"[MediGraph] top_k must be a positive integer, got {top_k}.")

    if not isinstance(score_threshold, (int, float)):
        raise ValueError(f"[MediGraph] score_threshold must be a float, got {score_threshold}.")

    if query_node not in node_mapping:
        raise KeyError(
            f"[MediGraph] Query entity '{query_node}' not found in node mapping."
        )

    if relation_type not in rel_mapping:
        raise KeyError(
            f"[MediGraph] Relation '{relation_type}' not found in relation mapping. "
            f"Available relations: {sorted(list(rel_mapping.keys()))}"
        )

    head_id = node_mapping[query_node]
    rel_id = rel_mapping[relation_type]
    num_nodes = len(node_df)
    effective_top_k = min(top_k, max(1, num_nodes - 1))

    device = next(model.parameters()).device
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # Compute node embeddings
    with torch.no_grad():
        z = model.encode(edge_index, edge_type)

        # Build candidate edges: (head_id, ?, candidate_target_id)
        all_targets = torch.arange(num_nodes, device=device)
        candidate_heads = torch.full((num_nodes,), head_id, dtype=torch.long, device=device)
        candidate_edges = torch.stack([candidate_heads, all_targets], dim=0)
        candidate_rel_types = torch.full((num_nodes,), rel_id, dtype=torch.long, device=device)

        # Compute DistMult raw scores
        scores = model.decode(z, candidate_edges, candidate_rel_types)

    scores_np = scores.cpu().numpy()

    # Identify existing known edges in full graph
    srcs = edge_index[0].cpu().numpy()
    tgts = edge_index[1].cpu().numpy()
    rels = edge_type.cpu().numpy()
    known_targets = set(tgts[(srcs == head_id) & (rels == rel_id)])

    # Exclude self-prediction (head -> head)
    scores_np[head_id] = -1e9

    # Rank target nodes
    top_indices = scores_np.argsort()[::-1][:effective_top_k]

    node_lookup = node_df.set_index("node_id")
    results: list[dict[str, Any]] = []

    for rank, tgt_id in enumerate(top_indices, start=1):
        tgt_id = int(tgt_id)
        score = float(scores_np[tgt_id])
        if score < score_threshold and rank > 1:
            # Respect threshold, but return at least 1 result if possible
            break

        tgt_row = node_lookup.loc[tgt_id]
        results.append({
            "rank": rank,
            "target_node_id": tgt_id,
            "target_name": str(tgt_row["entity_name"]),
            "target_type": str(tgt_row["entity_type"]),
            "relation": relation_type,
            "score": round(score, 4),
            "is_known_edge": tgt_id in known_targets,
        })

    return results


def save_predictions(
    predictions: list[dict[str, Any]],
    query_node: str,
    relation_type: str,
    checkpoint_path: str,
    cfg: dict[str, Any],
) -> Path:
    """
    Save predictions list to JSON matching API_CONTRACT.md §8.3 schema.
    """
    predictions_dir = Path(cfg["paths"]["predictions"])
    predictions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = predictions_dir / f"predictions_{timestamp}.json"

    payload = {
        "query": {
            "entity_name": query_node,
            "relation_type": relation_type,
            "top_k": len(predictions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "model_checkpoint": str(checkpoint_path),
        "predictions": predictions,
        "disclaimer": DISCLAIMER_TEXT,
    }

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    # Also save as CSV for tabular analysis and reporting
    csv_rows = []
    for p in predictions:
        csv_rows.append({
            "query_node": query_node,
            "relation": relation_type,
            "target_node": p["target_name"],
            "target_type": p["target_type"],
            "score": p["score"],
            "is_known_edge": p["is_known_edge"],
            "rank": p["rank"],
        })
    df_preds = pd.DataFrame(csv_rows)
    csv_file = predictions_dir / f"predictions_{timestamp}.csv"
    latest_csv = predictions_dir / "predictions.csv"
    df_preds.to_csv(csv_file, index=False)
    df_preds.to_csv(latest_csv, index=False)

    log.info("[MediGraph] Predictions saved to '%s' and '%s'", out_file, latest_csv)
    return out_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MediGraph AI — Link Prediction Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query-node", type=str, required=True,
                   help="Query entity name (e.g. 'Alzheimer disease')")
    p.add_argument("--relation", type=str, required=True,
                   help="Relation type to predict (e.g. 'indication')")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of top predictions to display (default: 20)")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Inference compute device")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config()

    loader = MediGraphLoader()
    node_df, edge_df, node_mapping, rel_mapping = loader.load_all(cfg)

    builder = GraphBuilder()
    edge_index, edge_type = builder.build_pyg_tensors(edge_df, rel_mapping)

    # Determine checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        ckpt_dir = Path(cfg["paths"]["checkpoints"])
        best_ckpt = ckpt_dir / "best_model.pt"
        if best_ckpt.exists():
            checkpoint_path = str(best_ckpt)
        else:
            log.warning("[MediGraph] No trained checkpoint found. Instantiating untrained model for testing.")
            from src.models.rgcn import RGCNEncoder
            from src.models.decoder import DistMultDecoder
            mc = cfg["model"]
            encoder = RGCNEncoder(
                num_nodes=len(node_df),
                num_relations=len(rel_mapping),
                in_channels=mc["in_channels"],
                hidden_channels=mc["hidden_channels"],
                out_channels=mc["out_channels"],
                num_layers=mc["num_layers"],
                num_bases=mc["num_bases"],
                dropout=mc["dropout"],
            )
            decoder = DistMultDecoder(
                num_relations=len(rel_mapping),
                embedding_dim=mc["out_channels"],
            )
            model = MediGraphModel(encoder, decoder)
            checkpoint_path = "untrained_model"

    if checkpoint_path != "untrained_model":
        model, _ = MediGraphModel.load_checkpoint(checkpoint_path, map_location=args.device)

    model.to(args.device)
    model.eval()

    log.info("[MediGraph] Running link prediction for ('%s', '%s', ?) ...",
             args.query_node, args.relation)

    predictions = predict_links(
        query_node=args.query_node,
        relation_type=args.relation,
        model=model,
        node_df=node_df,
        node_mapping=node_mapping,
        rel_mapping=rel_mapping,
        edge_index=edge_index,
        edge_type=edge_type,
        top_k=args.top_k,
    )

    print("\n" + "=" * 70)
    print(f"  PREDICTED LINKS FOR: '{args.query_node}' — [{args.relation}]")
    print("=" * 70)
    print(f"{'Rank':<5} {'Target Entity':<30} {'Type':<15} {'Score':<10} {'Known Edge?'}")
    print("-" * 70)
    for p in predictions:
        known_str = "✓ Yes" if p["is_known_edge"] else "No (New)"
        print(f"{p['rank']:<5} {p['target_name']:<30} {p['target_type']:<15} {p['score']:<10.4f} {known_str}")
    print("=" * 70)
    print(f"⚠️  {DISCLAIMER_TEXT}\n")

    saved_path = save_predictions(
        predictions=predictions,
        query_node=args.query_node,
        relation_type=args.relation,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
    )
    print(f"Saved predictions to: {saved_path}")


if __name__ == "__main__":
    main()
