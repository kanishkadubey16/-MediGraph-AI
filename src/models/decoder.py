"""
src/models/decoder.py
---------------------
DistMult decoder and the top-level MediGraphModel wrapper for MediGraph AI.

DistMult (Yang et al., 2015) scores a candidate triple ``(h, r, t)`` as:

    score(h, r, t) = z_h ⊙ W_r ⊙ z_t  (element-wise product, then sum)

where ``z_h``, ``z_t`` are node embeddings produced by the R-GCN encoder and
``W_r`` is a learnable relation-specific diagonal weight vector.

``MediGraphModel`` composes :class:`RGCNEncoder` and :class:`DistMultDecoder`
into a single :class:`~torch.nn.Module` and adds checkpoint persistence
helpers that comply with the §6 checkpoint format in
``docs/API_CONTRACT.md``.

References
----------
- Yang et al. (2015), "Embedding Entities and Relations for Learning and
  Inference in Knowledge Bases", ICLR 2015.

Usage
-----
    from src.models.rgcn    import RGCNEncoder
    from src.models.decoder import DistMultDecoder, MediGraphModel
    from src.utils.config   import load_config

    cfg = load_config()
    mc  = cfg["model"]

    encoder = RGCNEncoder(num_nodes=N, num_relations=R, **mc_kwargs)
    decoder = DistMultDecoder(num_relations=R, embedding_dim=mc["out_channels"])
    model   = MediGraphModel(encoder, decoder)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

try:
    import torch_geometric
    _PYG_VERSION: str = torch_geometric.__version__
except ImportError:  # pragma: no cover
    _PYG_VERSION = "unknown"

from src.models.rgcn import RGCNEncoder
from src.utils.logger import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  DistMult Decoder
# --------------------------------------------------------------------------- #

class DistMultDecoder(nn.Module):
    """DistMult bilinear decoder for knowledge-graph link prediction.

    For each candidate triple ``(h, r, t)`` the decoder computes a scalar
    score via the DistMult factorisation:

        score(h, r, t) = Σ_d  z_h[d] · W_r[d] · z_t[d]

    where ``W_r`` is a learnable relation embedding (diagonal matrix stored
    as a vector) initialised with Xavier uniform.

    **Important:** This module returns *raw* scores.  Apply
    :func:`torch.sigmoid` externally before computing binary cross-entropy
    loss (or use :func:`~torch.nn.BCEWithLogitsLoss`).

    Parameters
    ----------
    num_relations : int
        Number of distinct relation types (i.e. ``R``).  Must be > 0.
    embedding_dim : int
        Dimensionality of node embeddings produced by the encoder, and
        therefore also of each relation weight vector ``W_r``.  Must be > 0.

    Attributes
    ----------
    rel_weight : nn.Parameter
        Shape ``(num_relations, embedding_dim)``.  One weight vector per
        relation type, initialised with Xavier uniform.
    """

    def __init__(self, num_relations: int, embedding_dim: int) -> None:
        super().__init__()

        if num_relations <= 0:
            raise ValueError(
                f"[MediGraph] DistMultDecoder: num_relations must be > 0, "
                f"got {num_relations}."
            )
        if embedding_dim <= 0:
            raise ValueError(
                f"[MediGraph] DistMultDecoder: embedding_dim must be > 0, "
                f"got {embedding_dim}."
            )

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Learnable diagonal relation weight matrix (num_relations × D)
        self.rel_weight: nn.Parameter = nn.Parameter(
            torch.empty(num_relations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.rel_weight)


        log.info(
            "DistMultDecoder constructed: num_relations=%d, embedding_dim=%d",
            num_relations,
            embedding_dim,
        )

    def forward(self, z: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Compute DistMult scores for a batch of candidate edges.

        Parameters
        ----------
        z : Tensor
            Node embedding matrix, shape ``(N, D)``, dtype ``torch.float32``.
            Row ``i`` corresponds to ``node_id`` ``i`` from ``nodes.csv``.
        edge_index : Tensor
            Candidate edges, shape ``(2, B)``, dtype ``torch.long``.
            Row 0 = source node IDs, row 1 = target node IDs.
        edge_type : Tensor
            Relation type indices for each candidate edge, shape ``(B,)``,
            dtype ``torch.long``.  Values must be in ``[0, num_relations)``.

        Returns
        -------
        Tensor
            Raw DistMult scores, shape ``(B,)``, dtype ``torch.float32``.
            **Do not** apply sigmoid here — let the caller handle it for
            loss computation (e.g. :class:`~torch.nn.BCEWithLogitsLoss`).

        Raises
        ------
        ValueError
            If any input tensor has an unexpected shape or dtype.
        """
        self._validate_inputs(z, edge_index, edge_type)

        src_ids: Tensor = edge_index[0]   # (B,)
        tgt_ids: Tensor = edge_index[1]   # (B,)

        z_src: Tensor = z[src_ids]                     # (B, D)
        z_tgt: Tensor = z[tgt_ids]                     # (B, D)
        w_rel: Tensor = self.rel_weight[edge_type]     # (B, D)

        # Element-wise product then sum over embedding dimension → (B,)
        scores: Tensor = (z_src * w_rel * z_tgt).sum(dim=-1)
        return scores

    def score_candidates(
        self,
        head_idx: int,
        relation_idx: int,
        z: Tensor,
    ) -> Tensor:
        """Compute DistMult scores for all candidate tails against a query (head, relation).

        Parameters
        ----------
        head_idx : int
            Head node index in [0, num_nodes).
        relation_idx : int
            Relation type index in [0, num_relations).
        z : Tensor
            Node embeddings matrix of shape (num_nodes, embedding_dim).

        Returns
        -------
        Tensor
            Raw DistMult scores for all candidate tails, shape (num_nodes,).
        """
        h_emb = z[head_idx]                    # (D,)
        w_rel = self.rel_weight[relation_idx]  # (D,)
        hr = h_emb * w_rel                     # (D,)
        return torch.matmul(z, hr)             # (num_nodes,)

    def score_triples(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
        z: Tensor,
    ) -> Tensor:
        """Compute DistMult scores for a batch of triples (h, r, t).

        Parameters
        ----------
        heads : Tensor
            (B,) LongTensor of head IDs.
        relations : Tensor
            (B,) LongTensor of relation IDs.
        tails : Tensor
            (B,) LongTensor of tail IDs.
        z : Tensor
            (num_nodes, D) node embeddings.

        Returns
        -------
        Tensor
            (B,) raw DistMult scores.
        """
        edge_index = torch.stack([heads, tails], dim=0)
        return self.forward(z, edge_index, relations)


    def reset_parameters(self) -> None:
        """Re-initialise relation weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.rel_weight)
        log.info("DistMultDecoder parameters have been reset.")


    def _validate_inputs(
        self, z: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> None:
        """Validate input tensors before scoring.

        Parameters
        ----------
        z : Tensor
            Node embeddings, expected shape ``(N, D)``.
        edge_index : Tensor
            Expected shape ``(2, B)``, dtype ``torch.long``.
        edge_type : Tensor
            Expected shape ``(B,)``, dtype ``torch.long``.

        Raises
        ------
        ValueError
            On shape or dtype mismatch.
        """
        if z.ndim != 2:
            raise ValueError(
                f"[MediGraph] DistMultDecoder: z must be 2-D (N, D), "
                f"got shape {tuple(z.shape)}."
            )
        if z.shape[1] != self.embedding_dim:
            raise ValueError(
                f"[MediGraph] DistMultDecoder: z has embedding dim "
                f"{z.shape[1]} but decoder expects {self.embedding_dim}."
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"[MediGraph] DistMultDecoder: edge_index must have shape "
                f"(2, B), got {tuple(edge_index.shape)}."
            )
        if edge_index.dtype != torch.long:
            raise ValueError(
                f"[MediGraph] DistMultDecoder: edge_index must have dtype "
                f"torch.long, got {edge_index.dtype}."
            )

        batch_size = edge_index.shape[1]
        if edge_type.shape != (batch_size,):
            raise ValueError(
                f"[MediGraph] DistMultDecoder: edge_type must have shape "
                f"({batch_size},) to match edge_index, got {tuple(edge_type.shape)}."
            )
        if edge_type.dtype != torch.long:
            raise ValueError(
                f"[MediGraph] DistMultDecoder: edge_type must have dtype "
                f"torch.long, got {edge_type.dtype}."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_relations={self.num_relations}, "
            f"embedding_dim={self.embedding_dim})"
        )


# --------------------------------------------------------------------------- #
#  MediGraphModel — combined encoder + decoder wrapper
# --------------------------------------------------------------------------- #

class MediGraphModel(nn.Module):
    """Combined encoder–decoder model for knowledge-graph link prediction.

    This wrapper composes an :class:`RGCNEncoder` (message-passing encoder)
    with a :class:`DistMultDecoder` (scoring function) and exposes a clean
    ``encode`` / ``decode`` / ``forward`` API consistent with §5.3 of
    ``docs/API_CONTRACT.md``.

    It also provides :meth:`save_checkpoint` and :meth:`load_checkpoint` that
    read/write the exact checkpoint dict format specified in §6.

    Parameters
    ----------
    encoder : RGCNEncoder
        Instantiated R-GCN encoder.
    decoder : DistMultDecoder
        Instantiated DistMult decoder.

    Notes
    -----
    The ``embedding_dim`` of the decoder must equal the ``out_channels`` of
    the encoder.  This is validated at construction time.
    """

    def __init__(self, encoder: RGCNEncoder, decoder: DistMultDecoder) -> None:
        super().__init__()

        if not isinstance(encoder, RGCNEncoder):
            raise TypeError(
                f"[MediGraph] MediGraphModel: encoder must be an instance of "
                f"RGCNEncoder, got {type(encoder).__name__}."
            )
        if not isinstance(decoder, DistMultDecoder):
            raise TypeError(
                f"[MediGraph] MediGraphModel: decoder must be an instance of "
                f"DistMultDecoder, got {type(decoder).__name__}."
            )
        if encoder.out_channels != decoder.embedding_dim:
            raise ValueError(
                f"[MediGraph] MediGraphModel: encoder.out_channels "
                f"({encoder.out_channels}) must equal decoder.embedding_dim "
                f"({decoder.embedding_dim})."
            )
        if encoder.num_relations != decoder.num_relations:
            raise ValueError(
                f"[MediGraph] MediGraphModel: encoder.num_relations "
                f"({encoder.num_relations}) must equal decoder.num_relations "
                f"({decoder.num_relations})."
            )

        self.encoder = encoder
        self.decoder = decoder

        log.info(
            "MediGraphModel constructed: encoder=%r, decoder=%r",
            encoder,
            decoder,
        )

    # ---------------------------------------------------------------------- #
    # Core API
    # ---------------------------------------------------------------------- #

    def encode(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Run the encoder to produce node embeddings.

        Parameters
        ----------
        edge_index : Tensor
            Message-passing (training) edges, shape ``(2, E)``,
            dtype ``torch.long``.  See :meth:`RGCNEncoder.forward` warning.
        edge_type : Tensor
            Relation types for message-passing edges, shape ``(E,)``,
            dtype ``torch.long``.

        Returns
        -------
        Tensor
            Node embedding matrix, shape ``(num_nodes, out_channels)``,
            dtype ``torch.float32``.
        """
        return self.encoder(edge_index, edge_type)

    def decode(
        self, z: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> Tensor:
        """Score candidate edges using the DistMult decoder.

        Parameters
        ----------
        z : Tensor
            Node embeddings from :meth:`encode`, shape ``(N, D)``.
        edge_index : Tensor
            Candidate edges to score, shape ``(2, B)``, dtype ``torch.long``.
        edge_type : Tensor
            Relation types for candidate edges, shape ``(B,)``,
            dtype ``torch.long``.

        Returns
        -------
        Tensor
            Raw DistMult scores, shape ``(B,)``, dtype ``torch.float32``.
        """
        return self.decoder(z, edge_index, edge_type)

    def score_candidates(
        self, head_idx: int, relation_idx: int, z: Tensor
    ) -> Tensor:
        """Score candidate tails using the DistMult decoder."""
        return self.decoder.score_candidates(head_idx, relation_idx, z)


    def forward(
        self,
        mp_edge_index: Tensor,
        mp_edge_type: Tensor,
        pred_edge_index: Optional[Tensor] = None,
        pred_edge_type: Optional[Tensor] = None,
    ) -> Tensor:
        """Full forward pass: encode then decode.

        Parameters
        ----------
        mp_edge_index : Tensor
            Message-passing (training) edge index, shape ``(2, E)``,
            dtype ``torch.long``.  Passed to the encoder.
        mp_edge_type : Tensor
            Message-passing edge types, shape ``(E,)``, dtype ``torch.long``.
        pred_edge_index : Optional[Tensor]
            Candidate edges to score (positive + negative samples),
            shape ``(2, B)``, dtype ``torch.long``. Defaults to ``mp_edge_index``.
        pred_edge_type : Optional[Tensor]
            Relation types for candidate edges, shape ``(B,)``,
            dtype ``torch.long``. Defaults to ``mp_edge_type``.

        Returns
        -------
        Tensor
            Raw DistMult scores, shape ``(B,)``, dtype ``torch.float32``.
            Apply sigmoid externally to obtain probabilities / binary loss.
        """
        if pred_edge_index is None:
            pred_edge_index = mp_edge_index
        if pred_edge_type is None:
            pred_edge_type = mp_edge_type

        z: Tensor = self.encode(mp_edge_index, mp_edge_type)
        scores: Tensor = self.decode(z, pred_edge_index, pred_edge_type)
        return scores

    # ---------------------------------------------------------------------- #
    # Checkpoint persistence  (API CONTRACT §6)
    # ---------------------------------------------------------------------- #

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, Any],
    ) -> None:
        """Save a training checkpoint to disk.

        The checkpoint dict matches the exact format specified in
        ``docs/API_CONTRACT.md`` §6.

        Parameters
        ----------
        path : str
            Destination file path (e.g.
            ``"artifacts/checkpoints/rgcn_epoch_10.pt"``).
        epoch : int
            Current epoch number (1-indexed).
        optimizer : torch.optim.Optimizer
            Optimiser whose state dict will be saved.
        metrics : dict
            Must contain the following float-valued keys:

            - ``"train_loss"``
            - ``"val_loss"``
            - ``"val_mrr"``
            - ``"val_hits_at_10"``

            Additional keys are allowed and will be persisted.

        Raises
        ------
        KeyError
            If a required metric key is missing from ``metrics``.
        OSError
            If the target directory cannot be created or the file cannot be
            written.
        """
        required_metric_keys = ("train_loss", "val_loss", "val_mrr", "val_hits_at_10")
        for key in required_metric_keys:
            if key not in metrics:
                raise KeyError(
                    f"[MediGraph] MediGraphModel.save_checkpoint: required "
                    f"metric key '{key}' missing from metrics dict."
                )

        enc = self.encoder
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "train_loss": float(metrics["train_loss"]),
            "val_loss": float(metrics["val_loss"]),
            "val_mrr": float(metrics["val_mrr"]),
            "val_hits_at_10": float(metrics["val_hits_at_10"]),
            "model_config": {
                "num_nodes": enc.num_nodes,
                "num_relations": enc.num_relations,
                "in_channels": enc.in_channels,
                "hidden_channels": enc.hidden_channels,
                "out_channels": enc.out_channels,
                "num_layers": enc.num_layers,
                "num_bases": enc.num_bases,
                "dropout": enc.dropout,
            },
            "random_seed": _get_random_seed(),
            "torch_version": torch.__version__,
            "pyg_version": _PYG_VERSION,
        }

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)

        log.info(
            "Checkpoint saved to '%s' (epoch=%d, val_mrr=%.4f)",
            save_path,
            epoch,
            metrics["val_mrr"],
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        map_location: str = "cpu",
    ) -> tuple["MediGraphModel", dict[str, Any]]:
        """Load a MediGraphModel from a checkpoint file.

        Reconstructs the full model (encoder + decoder) from the
        ``"model_config"`` stored in the checkpoint, loads the saved state
        dict, and returns the model in **eval** mode together with the raw
        checkpoint dictionary.

        Parameters
        ----------
        path : str
            Path to the ``.pt`` checkpoint file.
        map_location : str
            Device string passed to :func:`torch.load`
            (e.g. ``"cpu"``, ``"cuda:0"``).

        Returns
        -------
        tuple[MediGraphModel, dict]
            ``(model, checkpoint_dict)`` where ``model`` is in eval mode.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        KeyError
            If the checkpoint dict is missing required keys.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"[MediGraph] MediGraphModel.load_checkpoint: checkpoint file "
                f"not found at '{checkpoint_path}'. "
                f"Run training first or verify the path."
            )

        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

        # Validate required top-level keys
        required_keys = (
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "train_loss",
            "val_loss",
            "val_mrr",
            "val_hits_at_10",
            "model_config",
            "random_seed",
            "torch_version",
            "pyg_version",
        )
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(
                    f"[MediGraph] MediGraphModel.load_checkpoint: required "
                    f"checkpoint key '{key}' is missing.  "
                    f"Checkpoint may have been produced by an incompatible "
                    f"version of MediGraph AI."
                )

        cfg: dict[str, Any] = checkpoint["model_config"]

        # Validate required model_config keys
        required_cfg_keys = (
            "num_nodes",
            "num_relations",
            "in_channels",
            "hidden_channels",
            "out_channels",
            "num_layers",
            "num_bases",
            "dropout",
        )
        for key in required_cfg_keys:
            if key not in cfg:
                raise KeyError(
                    f"[MediGraph] MediGraphModel.load_checkpoint: required "
                    f"model_config key '{key}' is missing."
                )

        # Reconstruct encoder and decoder from saved config
        encoder = RGCNEncoder(
            num_nodes=cfg["num_nodes"],
            num_relations=cfg["num_relations"],
            in_channels=cfg["in_channels"],
            hidden_channels=cfg["hidden_channels"],
            out_channels=cfg["out_channels"],
            num_layers=cfg["num_layers"],
            num_bases=cfg["num_bases"],
            dropout=cfg["dropout"],
        )
        decoder = DistMultDecoder(
            num_relations=cfg["num_relations"],
            embedding_dim=cfg["out_channels"],
        )

        model = cls(encoder, decoder)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        log.info(
            "Checkpoint loaded from '%s' "
            "(epoch=%d, val_mrr=%.4f, torch=%s, pyg=%s)",
            checkpoint_path,
            checkpoint["epoch"],
            checkpoint["val_mrr"],
            checkpoint.get("torch_version", "?"),
            checkpoint.get("pyg_version", "?"),
        )

        return model, checkpoint

    # ---------------------------------------------------------------------- #
    # Dunder helpers
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  encoder={self.encoder},\n"
            f"  decoder={self.decoder}\n"
            f")"
        )


# --------------------------------------------------------------------------- #
#  Module-private utilities
# --------------------------------------------------------------------------- #

def _get_random_seed() -> int:
    """Return a best-effort current random seed from :mod:`torch`.

    PyTorch does not expose the global seed directly after it has been set, so
    we sample the initial state of the default generator.  This is used only
    for checkpoint reproducibility metadata — it is not guaranteed to be
    perfectly accurate if the seed was set outside PyTorch.

    Returns
    -------
    int
        The initial seed of the default PyTorch random number generator.
    """
    return int(torch.initial_seed() & 0xFFFF_FFFF)


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #
MedicalDistMultDecoder = DistMultDecoder
MedicalRGCNModel = MediGraphModel

