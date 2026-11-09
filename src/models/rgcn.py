"""
src/models/rgcn.py
------------------
Relational Graph Convolutional Network (R-GCN) encoder for MediGraph AI.

The encoder maps every node in the biomedical knowledge graph to a dense
embedding vector via stacked RGCNConv layers (Schlichtkrull et al., 2018).
Node features are learned from scratch through a trainable embedding table
so that no external node-feature matrix is required at inference time.

References
----------
- Schlichtkrull et al. (2018), "Modeling Relational Data with Graph
  Convolutional Networks", ESWC 2018.
- PyTorch Geometric RGCNConv:
  https://pytorch-geometric.readthedocs.io/en/latest/generated/
  torch_geometric.nn.conv.RGCNConv.html

Usage
-----
    from src.models.rgcn import RGCNEncoder
    from src.utils.config import load_config

    cfg = load_config()
    mc  = cfg["model"]

    encoder = RGCNEncoder(
        num_nodes        = num_nodes,
        num_relations    = num_relations,
        in_channels      = mc["in_channels"],
        hidden_channels  = mc["hidden_channels"],
        out_channels     = mc["out_channels"],
        num_layers       = mc["num_layers"],
        num_bases        = mc["num_bases"],
        dropout          = mc["dropout"],
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import RGCNConv
except ImportError:
    class RGCNConv(nn.Module):
        """Fallback Relational Graph Convolution layer in pure PyTorch."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_relations: int,
            num_bases: int | None = None,
            **kwargs,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_relations = num_relations
            self.num_bases = num_bases

            if num_bases is not None and num_bases < num_relations:
                self.basis = nn.Parameter(torch.empty(num_bases, in_channels, out_channels))
                self.att = nn.Parameter(torch.empty(num_relations, num_bases))
            else:
                self.weight = nn.Parameter(torch.empty(num_relations, in_channels, out_channels))
                self.basis = None
                self.att = None

            self.root = nn.Parameter(torch.empty(in_channels, out_channels))
            self.bias = nn.Parameter(torch.empty(out_channels))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            if self.basis is not None:
                nn.init.xavier_uniform_(self.basis)
                nn.init.xavier_uniform_(self.att)
            else:
                nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.root)
            nn.init.zeros_(self.bias)

        def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
            num_nodes = x.size(0)
            out = torch.matmul(x, self.root)

            if edge_index.numel() > 0:
                src, tgt = edge_index[0], edge_index[1]
                if self.basis is not None:
                    w = torch.einsum("rb,bio->rio", self.att, self.basis)
                else:
                    w = self.weight

                msg = torch.bmm(x[src].unsqueeze(1), w[edge_type]).squeeze(1)
                out = out.clone()
                out.index_add_(0, tgt, msg)

            return out + self.bias


from src.utils.logger import get_logger

log = get_logger(__name__)


class RGCNEncoder(nn.Module):
    """Relational Graph Convolutional Network encoder.

    Builds a stack of :class:`~torch_geometric.nn.conv.RGCNConv` layers
    with basis decomposition.  Node features are **not** provided externally;
    instead, a learnable :class:`~torch.nn.Embedding` table is maintained
    for all ``num_nodes`` nodes and updated jointly during training.

    Architecture
    ------------
    Embedding(num_nodes, in_channels)
        → [RGCNConv → ReLU → Dropout] × (num_layers - 1)
        → RGCNConv
        → Tensor(num_nodes, out_channels)

    The final layer does **not** apply activation or dropout so that the raw
    embeddings can be freely used by whatever decoder follows.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph (i.e. ``N``).  Must be > 0.
    num_relations : int
        Total number of distinct edge/relation types (i.e. ``R``).  Must be > 0.
    in_channels : int
        Dimensionality of the initial node embedding (input to layer 0).
    hidden_channels : int
        Dimensionality of all intermediate RGCN layers.
        Ignored when ``num_layers == 1``.
    out_channels : int
        Dimensionality of the output node embeddings.
    num_layers : int
        Number of RGCN message-passing layers.  Must be >= 1.
    num_bases : int
        Number of basis matrices used in the basis-decomposition variant of
        RGCNConv.  Reduces parameter count when ``num_relations`` is large.
        Set to ``None`` to disable decomposition (not recommended for large R).
    dropout : float
        Dropout probability applied between hidden layers.  Range [0, 1).
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Input validation
        # ------------------------------------------------------------------ #
        if num_nodes <= 0:
            raise ValueError(
                f"[MediGraph] RGCNEncoder: num_nodes must be > 0, got {num_nodes}."
            )
        if num_relations <= 0:
            raise ValueError(
                f"[MediGraph] RGCNEncoder: num_relations must be > 0, got {num_relations}."
            )
        if num_layers < 1:
            raise ValueError(
                f"[MediGraph] RGCNEncoder: num_layers must be >= 1, got {num_layers}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"[MediGraph] RGCNEncoder: dropout must be in [0, 1), got {dropout}."
            )

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.dropout = dropout

        # ------------------------------------------------------------------ #
        # Learnable node feature table (replaces external node features)
        # ------------------------------------------------------------------ #
        self.node_emb: nn.Embedding = nn.Embedding(num_nodes, in_channels)

        # ------------------------------------------------------------------ #
        # Build the stack of RGCNConv layers
        # ------------------------------------------------------------------ #
        self.convs: nn.ModuleList = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_in = in_channels if layer_idx == 0 else hidden_channels
            layer_out = out_channels if layer_idx == num_layers - 1 else hidden_channels

            self.convs.append(
                RGCNConv(
                    in_channels=layer_in,
                    out_channels=layer_out,
                    num_relations=num_relations,
                    num_bases=num_bases,
                )
            )

        log.info(
            "RGCNEncoder constructed: num_nodes=%d, num_relations=%d, "
            "in=%d, hidden=%d, out=%d, layers=%d, bases=%d, dropout=%.2f",
            num_nodes,
            num_relations,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            num_bases,
            dropout,
        )

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Compute node embeddings by message-passing over the given edges.

        .. warning::
            ``edge_index`` and ``edge_type`` **must** contain only the
            *training* message-passing edges (i.e. the ``"train"`` split
            returned by ``src.training.splitter.split_edges``).

            **Never** pass validation or test edges here.  Doing so would
            cause data leakage and invalidate evaluation metrics.

        Parameters
        ----------
        edge_index : Tensor
            Shape ``(2, E)``, dtype ``torch.long``.  Row 0 contains source
            node IDs; row 1 contains target node IDs.  All IDs must be in
            ``[0, num_nodes)``.
        edge_type : Tensor
            Shape ``(E,)``, dtype ``torch.long``.  Each value is a relation
            type integer in ``[0, num_relations)``.

        Returns
        -------
        Tensor
            Shape ``(num_nodes, out_channels)``, dtype ``torch.float32``.
            Row ``i`` is the embedding for node ``i`` (matches ``node_id``
            from ``nodes.csv``).
        """
        self._validate_inputs(edge_index, edge_type)

        # Retrieve all node embeddings (shape: num_nodes × in_channels)
        x: Tensor = self.node_emb.weight

        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            is_last_layer = layer_idx == self.num_layers - 1
            if not is_last_layer:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # shape: (num_nodes, out_channels)

    # ---------------------------------------------------------------------- #
    # Convenience helpers
    # ---------------------------------------------------------------------- #

    def get_embeddings(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Return node embeddings in inference mode (no gradient, eval mode).

        This is a convenience wrapper around :meth:`forward` that temporarily
        sets the model to ``eval`` mode and disables gradient computation.
        The model is restored to its previous training state afterwards.

        Parameters
        ----------
        edge_index : Tensor
            Shape ``(2, E)``, dtype ``torch.long``.  Training message-passing
            edges only — see :meth:`forward` docstring warning.
        edge_type : Tensor
            Shape ``(E,)``, dtype ``torch.long``.

        Returns
        -------
        Tensor
            Shape ``(num_nodes, out_channels)``, dtype ``torch.float32``.
        """
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                embeddings = self.forward(edge_index, edge_type)
        finally:
            # Always restore previous mode even if forward raises
            if was_training:
                self.train()

        log.debug(
            "get_embeddings() produced tensor of shape %s on device %s",
            tuple(embeddings.shape),
            embeddings.device,
        )
        return embeddings

    def reset_parameters(self) -> None:
        """Re-initialise all learnable weights.

        Resets the node embedding table with a standard normal distribution
        and delegates to each ``RGCNConv`` layer's own ``reset_parameters``
        method.  Call this before re-training a model from scratch.
        """
        nn.init.normal_(self.node_emb.weight)
        for conv in self.convs:
            conv.reset_parameters()
        log.info("RGCNEncoder parameters have been reset.")

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _validate_inputs(self, edge_index: Tensor, edge_type: Tensor) -> None:
        """Validate tensor shapes and dtypes before the forward pass.

        Parameters
        ----------
        edge_index : Tensor
            Expected shape ``(2, E)``, dtype ``torch.long``.
        edge_type : Tensor
            Expected shape ``(E,)``, dtype ``torch.long``.

        Raises
        ------
        ValueError
            On shape or dtype mismatch.
        """
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"[MediGraph] RGCNEncoder: edge_index must have shape (2, E), "
                f"got {tuple(edge_index.shape)}."
            )
        if edge_index.dtype != torch.long:
            raise ValueError(
                f"[MediGraph] RGCNEncoder: edge_index must have dtype torch.long, "
                f"got {edge_index.dtype}."
            )

        num_edges = edge_index.shape[1]

        if edge_type.ndim != 1 or edge_type.shape[0] != num_edges:
            raise ValueError(
                f"[MediGraph] RGCNEncoder: edge_type must have shape ({num_edges},) "
                f"to match edge_index, got {tuple(edge_type.shape)}."
            )
        if edge_type.dtype != torch.long:
            raise ValueError(
                f"[MediGraph] RGCNEncoder: edge_type must have dtype torch.long, "
                f"got {edge_type.dtype}."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_nodes={self.num_nodes}, "
            f"num_relations={self.num_relations}, "
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_bases={self.num_bases}, "
            f"dropout={self.dropout})"
        )
