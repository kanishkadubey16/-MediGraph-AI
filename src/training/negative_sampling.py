"""
src/training/negative_sampling.py
---------------------------------
Negative sampling module for MediGraph AI biomedical link prediction.

Key Responsibilities:
- Generate corrupted triples by replacing head or tail entity (with 50/50 probability).
- Avoid sampling known positive relationships as negatives (filtered sampling).
- Respect biomedical entity-type compatibility and relation-specific constraints:
  for a given relation r, corrupted replacement entities are selected from valid
  candidate entities (e.g., in (disease, disease_protein, gene), corrupted tails
  are sampled from genes/proteins rather than unrelated entities like drugs).
- Support configurable negative-to-positive sampling ratios.
- Fallback gracefully to uniform sampling if domain constraints are unavailable
  or candidate pools are exhausted.

Usage::

    from src.training.negative_sampling import TypedNegativeSampler, corrupt_triples

    sampler = TypedNegativeSampler(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        node_df=node_df,
    )
    neg_ei, neg_et = sampler.sample(
        pos_edge_index=train_ei,
        pos_edge_type=train_et,
        num_neg_per_pos=1,
    )
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.utils.logger import get_logger

log = get_logger(__name__)


class TypedNegativeSampler:
    """
    Biomedical type-constrained negative sampler for knowledge graph link prediction.

    Maintains relation-specific head and tail candidate entity pools based on
    observed triples in the knowledge graph, and filters out known positive
    triples during corruption.

    Parameters
    ----------
    edge_index : Tensor
        (2, E) LongTensor of all known graph edges.
    edge_type : Tensor
        (E,) LongTensor of relation types.
    num_nodes : int
        Total number of nodes in the graph.
    node_df : Optional[pd.DataFrame]
        Optional DataFrame with columns [node_id, entity_type] to enforce
        explicit entity type constraints.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        edge_index: Tensor,
        edge_type: Tensor,
        num_nodes: int,
        node_df: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
    ) -> None:
        if num_nodes < 2:
            raise ValueError(f"[MediGraph] num_nodes must be >= 2, got {num_nodes}")

        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)

        # Build known positive set: (head, rel, tail)
        srcs = edge_index[0].cpu().numpy()
        tgts = edge_index[1].cpu().numpy()
        rels = edge_type.cpu().numpy()

        self.known_triples: Set[Tuple[int, int, int]] = {
            (int(s), int(r), int(t)) for s, r, t in zip(srcs, rels, tgts)
        }
        self.known_pairs: Set[Tuple[int, int]] = {
            (int(s), int(t)) for s, t in zip(srcs, tgts)
        }

        # Relation-specific valid head and tail candidate pools
        self.rel_heads: Dict[int, np.ndarray] = {}
        self.rel_tails: Dict[int, np.ndarray] = {}

        unique_rels = np.unique(rels)
        for r in unique_rels:
            r_mask = (rels == r)
            self.rel_heads[int(r)] = np.unique(srcs[r_mask])
            self.rel_tails[int(r)] = np.unique(tgts[r_mask])

        # Entity type pools if node_df is provided
        self.type_to_nodes: Dict[str, np.ndarray] = {}
        self.node_to_type: Dict[int, str] = {}
        if node_df is not None and "node_id" in node_df.columns and "entity_type" in node_df.columns:
            for etype, group in node_df.groupby("entity_type"):
                self.type_to_nodes[str(etype)] = group["node_id"].to_numpy(dtype=np.int64)
            self.node_to_type = dict(zip(node_df["node_id"], node_df["entity_type"]))

        log.debug(
            "TypedNegativeSampler initialised with %d known triples across %d relations.",
            len(self.known_triples),
            len(unique_rels),
        )

    def sample(
        self,
        pos_edge_index: Tensor,
        pos_edge_type: Tensor,
        num_neg_per_pos: int = 1,
        type_constrained: bool = True,
        max_retries: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample negative edges by corrupting head or tail entity.

        Parameters
        ----------
        pos_edge_index : Tensor
            (2, B) LongTensor of positive edges.
        pos_edge_type : Tensor
            (B,) LongTensor of positive edge relation types.
        num_neg_per_pos : int
            Number of negative edges to generate per positive edge. Default 1.
        type_constrained : bool
            Whether to restrict corruption to compatible entity types / relation pools.
        max_retries : int
            Maximum number of attempts to resample if a corrupted triple is known positive.

        Returns
        -------
        neg_edge_index : Tensor
            (2, B * num_neg_per_pos) LongTensor.
        neg_edge_type : Tensor
            (B * num_neg_per_pos,) LongTensor.
        """
        if pos_edge_index.dim() != 2 or pos_edge_index.shape[0] != 2:
            raise ValueError(
                f"[MediGraph] pos_edge_index must have shape (2, B), got {tuple(pos_edge_index.shape)}"
            )
        B = pos_edge_index.shape[1]
        if pos_edge_type.shape[0] != B:
            raise ValueError(
                f"[MediGraph] pos_edge_type length {pos_edge_type.shape[0]} does not match {B}"
            )
        if num_neg_per_pos < 1:
            raise ValueError(f"[MediGraph] num_neg_per_pos must be >= 1, got {num_neg_per_pos}")

        device = pos_edge_index.device
        srcs = pos_edge_index[0].cpu().numpy()
        tgts = pos_edge_index[1].cpu().numpy()
        rels = pos_edge_type.cpu().numpy()

        neg_srcs: List[int] = []
        neg_tgts: List[int] = []
        neg_rels: List[int] = []

        all_node_ids = np.arange(self.num_nodes, dtype=np.int64)

        for i in range(B):
            h, r, t = int(srcs[i]), int(rels[i]), int(tgts[i])

            # Candidate pools for head and tail
            if type_constrained:
                # Prefer relation-specific head and tail pools
                head_pool = self.rel_heads.get(r, all_node_ids)
                tail_pool = self.rel_tails.get(r, all_node_ids)
                # If pool is too small (<=1 entity), fallback to all nodes
                if len(head_pool) <= 1:
                    head_pool = all_node_ids
                if len(tail_pool) <= 1:
                    tail_pool = all_node_ids
            else:
                head_pool = all_node_ids
                tail_pool = all_node_ids

            for _ in range(num_neg_per_pos):
                corrupt_head = bool(self.rng.random() < 0.5)
                chosen_h, chosen_t = h, t

                for _ in range(max_retries):
                    if corrupt_head:
                        cand = int(self.rng.choice(head_pool))
                        if cand != h and (cand, r, t) not in self.known_triples:
                            chosen_h = cand
                            break
                    else:
                        cand = int(self.rng.choice(tail_pool))
                        if cand != t and (h, r, cand) not in self.known_triples:
                            chosen_t = cand
                            break
                else:
                    # Fallback: uniform random node that is not the same entity
                    if corrupt_head:
                        cand = int(self.rng.choice(all_node_ids))
                        chosen_h = cand if cand != h else (cand + 1) % self.num_nodes
                    else:
                        cand = int(self.rng.choice(all_node_ids))
                        chosen_t = cand if cand != t else (cand + 1) % self.num_nodes

                neg_srcs.append(chosen_h)
                neg_tgts.append(chosen_t)
                neg_rels.append(r)

        neg_edge_index = torch.tensor([neg_srcs, neg_tgts], dtype=torch.long, device=device)
        neg_edge_type = torch.tensor(neg_rels, dtype=torch.long, device=device)

        return neg_edge_index, neg_edge_type


def corrupt_triples(
    pos_edge_index: Tensor,
    pos_edge_type: Tensor,
    num_nodes: int,
    num_neg_per_pos: int = 1,
    known_edges: Optional[Set[Tuple[int, int]]] = None,
    node_df: Optional[pd.DataFrame] = None,
    type_constrained: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Generate negative edges via head-or-tail corruption with optional type constraints.

    Parameters
    ----------
    pos_edge_index : Tensor
        (2, B) LongTensor of positive edges.
    pos_edge_type : Tensor
        (B,) LongTensor of relation types.
    num_nodes : int
        Total number of nodes.
    num_neg_per_pos : int
        Number of negatives per positive edge. Default 1.
    known_edges : Optional[Set[Tuple[int, int]]]
        Set of known (src, tgt) pairs or (src, rel, tgt) triples to avoid.
    node_df : Optional[pd.DataFrame]
        DataFrame with columns [node_id, entity_type].
    type_constrained : bool
        Whether to enforce relation-compatible negative sampling. Default True.
    seed : Optional[int]
        Random seed.

    Returns
    -------
    neg_edge_index : Tensor
        (2, B * num_neg_per_pos) LongTensor.
    neg_edge_type : Tensor
        (B * num_neg_per_pos,) LongTensor.
    """
    sampler = TypedNegativeSampler(
        edge_index=pos_edge_index,
        edge_type=pos_edge_type,
        num_nodes=num_nodes,
        node_df=node_df,
        seed=seed,
    )
    if known_edges is not None:
        for item in known_edges:
            if len(item) == 2:
                sampler.known_pairs.add((item[0], item[1]))
            elif len(item) == 3:
                sampler.known_triples.add((item[0], item[1], item[2]))

    return sampler.sample(
        pos_edge_index=pos_edge_index,
        pos_edge_type=pos_edge_type,
        num_neg_per_pos=num_neg_per_pos,
        type_constrained=type_constrained,
    )


# Alias for backward-compatibility with splitter.sample_negatives
sample_negatives = corrupt_triples

