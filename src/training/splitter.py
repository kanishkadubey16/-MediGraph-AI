"""
src/training/splitter.py
------------------------
Edge splitting and negative sampling utilities for MediGraph AI.

This module is responsible for:
  - Stratified train/val/test splitting of edges by relation type, so that
    each relation type maintains its proportion across all three splits.
  - Corrupt-based negative sampling (head or tail corruption, 50/50) with
    optional filtering against known positive edges.

Usage::

    from src.training.splitter import split_edges, sample_negatives

    splits = split_edges(edge_index, edge_type)
    neg_ei, neg_et = sample_negatives(
        splits["train"]["edge_index"],
        splits["train"]["edge_type"],
        num_nodes=num_nodes,
        known_edges=known_edges_set,
    )
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Split edges into stratified train / val / test partitions.

    Edges are shuffled deterministically (using ``seed``) and then split
    *per relation type* so that the type distribution is preserved across
    all three partitions.

    Parameters
    ----------
    edge_index : torch.Tensor
        Shape ``(2, E)``, dtype ``torch.long``.  Source / target node IDs.
    edge_type : torch.Tensor
        Shape ``(E,)``, dtype ``torch.long``.  Relation type for each edge.
    train_ratio : float
        Fraction of edges allocated to training.  Default ``0.80``.
    val_ratio : float
        Fraction of edges allocated to validation.  Default ``0.10``.
    test_ratio : float
        Fraction of edges allocated to testing.  Default ``0.10``.
    seed : int
        RNG seed for reproducible shuffling.  Default ``42``.

    Returns
    -------
    dict
        Nested dict with keys ``"train"``, ``"val"``, ``"test"``.  Each
        value is itself a dict with keys:

        - ``"edge_index"`` — ``torch.Tensor`` of shape ``(2, E_split)``
        - ``"edge_type"``  — ``torch.Tensor`` of shape ``(E_split,)``

    Raises
    ------
    ValueError
        If ratios do not sum to 1.0 (within floating-point tolerance), or if
        any ratio is non-positive, or if tensor shapes are inconsistent.

    Notes
    -----
    **Critical contract (§7.1 API_CONTRACT.md):**
    ``val`` and ``test`` edges must **never** be passed to
    ``RGCNEncoder.forward()`` during training.  Only ``train`` edges are used
    for message passing.
    """
    # ---- Input validation --------------------------------------------------
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"[MediGraph] split_edges: edge_index must have shape (2, E), "
            f"got {tuple(edge_index.shape)}."
        )
    num_edges = edge_index.shape[1]
    if edge_type.shape != (num_edges,):
        raise ValueError(
            f"[MediGraph] split_edges: edge_type shape {tuple(edge_type.shape)} "
            f"does not match edge_index column count {num_edges}."
        )
    for name, ratio in [("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)]:
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(
                f"[MediGraph] split_edges: {name}={ratio} must be in (0, 1)."
            )
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"[MediGraph] split_edges: ratios must sum to 1.0, "
            f"got {ratio_sum:.8f} (train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    log.info(
        "Splitting %d edges (train=%.0f%%, val=%.0f%%, test=%.0f%%, seed=%d).",
        num_edges, train_ratio * 100, val_ratio * 100, test_ratio * 100, seed,
    )

    # ---- Stratified split per relation type --------------------------------
    generator = torch.Generator()
    generator.manual_seed(seed)

    unique_types = edge_type.unique()
    train_indices: list[torch.Tensor] = []
    val_indices:   list[torch.Tensor] = []
    test_indices:  list[torch.Tensor] = []

    for rel in unique_types:
        # Indices belonging to this relation type
        mask: torch.Tensor = (edge_type == rel).nonzero(as_tuple=False).squeeze(1)

        # Shuffle within this relation type
        perm = torch.randperm(mask.numel(), generator=generator)
        mask = mask[perm]

        n = mask.numel()
        n_train = max(1, round(train_ratio * n))
        n_val   = max(1, round(val_ratio   * n))
        # test gets the remainder to ensure no edges are dropped
        n_test  = n - n_train - n_val
        if n_test < 1:
            # Edge case: not enough edges to populate all three splits.
            # Prioritise train, then val; test gets at least 0 (may be empty).
            n_test = max(0, n - n_train - n_val)

        train_indices.append(mask[:n_train])
        val_indices.append(  mask[n_train : n_train + n_val])
        test_indices.append( mask[n_train + n_val :])

        log.debug(
            "  Relation %d: total=%d  train=%d  val=%d  test=%d",
            rel.item(), n, n_train, n_val, n - n_train - n_val,
        )

    # Concatenate all per-type indices
    train_idx = torch.cat(train_indices)
    val_idx   = torch.cat(val_indices)
    test_idx  = torch.cat(test_indices)

    def _gather(idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "edge_index": edge_index[:, idx],
            "edge_type":  edge_type[idx],
        }

    splits = {
        "train": _gather(train_idx),
        "val":   _gather(val_idx),
        "test":  _gather(test_idx),
    }

    log.info(
        "Split complete — train: %d  val: %d  test: %d  (total: %d).",
        train_idx.numel(), val_idx.numel(), test_idx.numel(),
        train_idx.numel() + val_idx.numel() + test_idx.numel(),
    )
    return splits


# ---------------------------------------------------------------------------


def sample_negatives(
    pos_edge_index: torch.Tensor,
    pos_edge_type: torch.Tensor,
    num_nodes: int,
    num_neg_per_pos: int = 1,
    known_edges: Optional[set] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate negative edges by head-or-tail corruption of positive edges.

    For each positive triple ``(h, r, t)``, ``num_neg_per_pos`` negative
    triples are produced.  With probability 0.5 the **head** is corrupted
    (a random node replaces ``h``); otherwise the **tail** is corrupted.
    The sampling is fully vectorised (no Python loop over edges).

    If ``known_edges`` is provided, corrupted triples that happen to exist
    in that set are resampled up to a fixed number of retries (10 by
    default) to avoid leaking true positives.

    Parameters
    ----------
    pos_edge_index : torch.Tensor
        Shape ``(2, B)``, dtype ``torch.long``.  Positive edges to corrupt.
    pos_edge_type : torch.Tensor
        Shape ``(B,)``, dtype ``torch.long``.  Relation types for each edge.
    num_nodes : int
        Total number of nodes in the graph (used for uniform sampling).
    num_neg_per_pos : int
        Number of negatives to generate per positive edge.  Default ``1``.
    known_edges : set of (int, int) tuples, optional
        Set of ``(source, target)`` pairs that are known positive edges.
        Negatives that fall in this set will be resampled.

    Returns
    -------
    neg_edge_index : torch.Tensor
        Shape ``(2, B * num_neg_per_pos)``, dtype ``torch.long``.
    neg_edge_type : torch.Tensor
        Shape ``(B * num_neg_per_pos,)``, dtype ``torch.long``.

    Raises
    ------
    ValueError
        If tensor shapes are inconsistent or ``num_nodes`` < 2.
    """
    # ---- Input validation --------------------------------------------------
    if pos_edge_index.dim() != 2 or pos_edge_index.shape[0] != 2:
        raise ValueError(
            f"[MediGraph] sample_negatives: pos_edge_index must have shape (2, B), "
            f"got {tuple(pos_edge_index.shape)}."
        )
    B = pos_edge_index.shape[1]
    if pos_edge_type.shape != (B,):
        raise ValueError(
            f"[MediGraph] sample_negatives: pos_edge_type shape {tuple(pos_edge_type.shape)} "
            f"does not match pos_edge_index column count {B}."
        )
    if num_nodes < 2:
        raise ValueError(
            f"[MediGraph] sample_negatives: num_nodes must be >= 2, got {num_nodes}."
        )
    if num_neg_per_pos < 1:
        raise ValueError(
            f"[MediGraph] sample_negatives: num_neg_per_pos must be >= 1, "
            f"got {num_neg_per_pos}."
        )

    device = pos_edge_index.device
    total_neg = B * num_neg_per_pos

    # Repeat each positive edge num_neg_per_pos times
    # Shape after repeat: (2, total_neg)
    heads = pos_edge_index[0].repeat_interleave(num_neg_per_pos)  # (total_neg,)
    tails = pos_edge_index[1].repeat_interleave(num_neg_per_pos)  # (total_neg,)
    types = pos_edge_type.repeat_interleave(num_neg_per_pos)       # (total_neg,)

    _MAX_RETRIES = 10
    for _attempt in range(_MAX_RETRIES):
        # Decide which side to corrupt for each negative (0 → head, 1 → tail)
        corrupt_tail_mask = torch.randint(0, 2, (total_neg,), device=device).bool()

        # Sample random replacement nodes for every negative
        random_nodes = torch.randint(0, num_nodes, (total_neg,), device=device)

        # Apply corruption
        neg_heads = torch.where(corrupt_tail_mask, heads, random_nodes)
        neg_tails = torch.where(corrupt_tail_mask, random_nodes, tails)

        if known_edges is None:
            break

        # Filter: find negatives that accidentally landed on a known edge
        invalid_mask = torch.tensor(
            [
                (int(h), int(t)) in known_edges
                for h, t in zip(neg_heads.tolist(), neg_tails.tolist())
            ],
            dtype=torch.bool,
            device=device,
        )
        if not invalid_mask.any():
            break

        log.debug(
            "sample_negatives: retry %d — %d/%d negatives collided with known edges.",
            _attempt + 1, int(invalid_mask.sum()), total_neg,
        )
        # On the last retry keep whatever we have (some collisions acceptable)

    neg_edge_index = torch.stack([neg_heads, neg_tails], dim=0)  # (2, total_neg)

    return neg_edge_index, types
