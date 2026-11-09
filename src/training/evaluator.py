"""
src/training/evaluator.py
-------------------------
Evaluation metrics for MediGraph AI link prediction.

Implements the standard *filtered* ranking protocol used in knowledge graph
embedding literature (Bordes et al., 2013):

- For each positive triple ``(h, r, t)``, score every candidate tail entity
  ``t' ∈ [0, N)`` against the query ``(h, r, ?)``.
- Remove all *other* known positive tails for the same ``(h, r)`` pair from
  the ranking before computing the rank of the true answer.
- Report Mean Reciprocal Rank (MRR) and Hits@K.

Usage::

    from src.training.evaluator import LinkPredictionEvaluator

    evaluator = LinkPredictionEvaluator(num_nodes=N, device="cpu")
    metrics = evaluator.evaluate_mrr_hits(
        model, eval_edge_index, eval_edge_type,
        mp_edge_index, mp_edge_type,
    )
    auc = evaluator.compute_auc_roc(pos_scores, neg_scores)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sklearn.metrics import roc_auc_score

from src.utils.logger import get_logger

if TYPE_CHECKING:
    # Avoid circular imports; MediGraphModel lives in src/models/
    from src.models.rgcn import MediGraphModel  # type: ignore[attr-defined]

log = get_logger(__name__)


class LinkPredictionEvaluator:
    """
    Filtered MRR / Hits@K / AUC-ROC evaluator for link prediction.

    The evaluator follows the standard KG embedding filtered evaluation
    protocol.  For each positive test triple ``(h, r, t)``:

    1. Score ``(h, r, t')`` for every node ``t' ∈ [0, N)``.
    2. Set the scores of all *other* known tails for ``(h, r)`` to
       ``-∞`` so they are pushed below the true answer.
    3. Compute the rank of ``t`` among the remaining candidates.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    def __init__(self, num_nodes: int, device: str = "cpu") -> None:
        """
        Initialise the evaluator.

        Parameters
        ----------
        num_nodes : int
            Total number of nodes in the graph.  Must be >= 1.
        device : str
            PyTorch device string.  Default ``"cpu"``.

        Raises
        ------
        ValueError
            If ``num_nodes`` < 1.
        """
        if num_nodes < 1:
            raise ValueError(
                f"[MediGraph] LinkPredictionEvaluator: num_nodes must be >= 1, "
                f"got {num_nodes}."
            )
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        log.debug(
            "LinkPredictionEvaluator initialised: num_nodes=%d, device=%s.",
            num_nodes, device,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def evaluate_mrr_hits(
        self,
        model: "MediGraphModel",
        eval_edge_index: torch.Tensor,
        eval_edge_type: torch.Tensor,
        mp_edge_index: torch.Tensor,
        mp_edge_type: torch.Tensor,
        batch_size: int = 256,
        hits_k: list[int] = [1, 10, 50],
    ) -> dict[str, float]:
        """
        Compute filtered MRR and Hits@K on an evaluation edge set.

        The encoder is run **once** on the message-passing (training) edges to
        produce node embeddings; the decoder then scores every candidate tail
        for each query triple.

        Parameters
        ----------
        model : MediGraphModel
            The combined encoder+decoder wrapper.  Must already be in
            ``eval()`` mode; a ``ValueError`` is raised if not.
        eval_edge_index : torch.Tensor
            Shape ``(2, E_eval)``, dtype ``torch.long``.  Positive edges to
            rank.
        eval_edge_type : torch.Tensor
            Shape ``(E_eval,)``, dtype ``torch.long``.  Relation types for
            evaluation edges.
        mp_edge_index : torch.Tensor
            Shape ``(2, E_train)``, dtype ``torch.long``.  Message-passing
            (training-only) edges used for the encoder.
        mp_edge_type : torch.Tensor
            Shape ``(E_train,)``, dtype ``torch.long``.  Relation types for
            message-passing edges.
        batch_size : int
            Number of evaluation triples processed per batch.  Default 256.
        hits_k : list[int]
            List of K values for Hits@K.  Default ``[1, 10, 50]``.

        Returns
        -------
        dict[str, float]
            Keys: ``"mrr"``, ``"hits@1"``, ``"hits@10"``, ``"hits@50"``
            (and ``"hits@K"`` for any other K in *hits_k*).
            All values are floats in ``[0, 1]``.

        Raises
        ------
        ValueError
            If the model is not in eval mode, or if tensor shapes are
            inconsistent.
        """
        # ---- Guard: model must be in eval mode ----------------------------
        if model.training:
            raise ValueError(
                "[MediGraph] evaluate_mrr_hits: model must be in eval() mode. "
                "Call model.eval() before evaluation."
            )

        # ---- Input shape validation ----------------------------------------
        E_eval = eval_edge_index.shape[1]
        if eval_edge_type.shape != (E_eval,):
            raise ValueError(
                f"[MediGraph] evaluate_mrr_hits: eval_edge_type shape "
                f"{tuple(eval_edge_type.shape)} does not match eval_edge_index "
                f"column count {E_eval}."
            )

        log.info(
            "Starting filtered MRR/Hits@K evaluation on %d edges (batch_size=%d).",
            E_eval, batch_size,
        )

        # Move everything to the evaluator's device
        eval_edge_index = eval_edge_index.to(self.device)
        eval_edge_type  = eval_edge_type.to(self.device)
        mp_edge_index   = mp_edge_index.to(self.device)
        mp_edge_type    = mp_edge_type.to(self.device)

        # ---- Build filter: (src, rel) → set of known positive tails --------
        # This covers ALL edges (train + eval combined) used for filtering.
        all_ei   = torch.cat([mp_edge_index,  eval_edge_index],  dim=1)
        all_et   = torch.cat([mp_edge_type,   eval_edge_type])

        filter_dict: dict[tuple[int, int], list[int]] = {}
        srcs = all_ei[0].tolist()
        tgts = all_ei[1].tolist()
        rels = all_et.tolist()
        for s, r, t in zip(srcs, rels, tgts):
            key = (int(s), int(r))
            if key not in filter_dict:
                filter_dict[key] = []
            filter_dict[key].append(int(t))

        # ---- Encode once (no gradient) ------------------------------------
        with torch.no_grad():
            z: torch.Tensor = model.encode(mp_edge_index, mp_edge_type)
            # z shape: (num_nodes, out_channels)

        # ---- Evaluate in batches ------------------------------------------
        all_ranks: list[float] = []
        # Candidate tails tensor: (num_nodes,)
        all_nodes = torch.arange(self.num_nodes, device=self.device)

        for start in range(0, E_eval, batch_size):
            end = min(start + batch_size, E_eval)
            b_ei   = eval_edge_index[:, start:end]   # (2, B)
            b_et   = eval_edge_type[start:end]        # (B,)

            B = b_ei.shape[1]
            with torch.no_grad():
                for i in range(B):
                    head  = int(b_ei[0, i].item())
                    tail  = int(b_ei[1, i].item())
                    rel   = int(b_et[i].item())

                    # Build candidate edge index: head → all nodes  (2, N)
                    cand_heads = torch.full(
                        (self.num_nodes,), head, dtype=torch.long, device=self.device
                    )
                    cand_ei   = torch.stack([cand_heads, all_nodes], dim=0)   # (2, N)
                    cand_et   = torch.full(
                        (self.num_nodes,), rel, dtype=torch.long, device=self.device
                    )

                    scores: torch.Tensor = model.decode(z, cand_ei, cand_et)  # (N,)

                    # Filtered setting: mask out other known positive tails
                    known_tails = filter_dict.get((head, rel), [])
                    for kt in known_tails:
                        if kt != tail:
                            scores[kt] = float("-inf")

                    # Rank of the true tail (1-indexed, lower is better)
                    true_score = scores[tail]
                    rank = int((scores > true_score).sum().item()) + 1
                    all_ranks.append(float(rank))

        # ---- Aggregate metrics ---------------------------------------------
        ranks_tensor = torch.tensor(all_ranks, dtype=torch.float32)
        mrr = float((1.0 / ranks_tensor).mean().item())
        metrics: dict[str, float] = {"mrr": mrr}
        for k in hits_k:
            hits = float((ranks_tensor <= k).float().mean().item())
            metrics[f"hits@{k}"] = hits

        log.info(
            "Evaluation done: MRR=%.4f  %s",
            mrr,
            "  ".join(f"Hits@{k}={metrics[f'hits@{k}']:.4f}" for k in hits_k),
        )
        return metrics

    # ------------------------------------------------------------------

    def compute_auc_roc(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> float:
        """
        Compute Area Under the ROC Curve (AUC-ROC) using scikit-learn.

        Parameters
        ----------
        pos_scores : torch.Tensor
            Shape ``(B,)``.  Raw DistMult scores for **positive** edges.
        neg_scores : torch.Tensor
            Shape ``(B,)``.  Raw DistMult scores for **negative** edges.

        Returns
        -------
        float
            AUC-ROC value in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``pos_scores`` and ``neg_scores`` are empty or have
            incompatible shapes when concatenated.
        """
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            raise ValueError(
                "[MediGraph] compute_auc_roc: pos_scores and neg_scores must "
                "both be non-empty tensors."
            )

        scores_np = torch.cat([pos_scores.detach().cpu(),
                                neg_scores.detach().cpu()]).numpy()
        labels_np = torch.cat([
            torch.ones(pos_scores.numel(), dtype=torch.long),
            torch.zeros(neg_scores.numel(), dtype=torch.long),
        ]).numpy()

        auc = float(roc_auc_score(labels_np, scores_np))
        log.debug("AUC-ROC = %.4f", auc)
        return auc
