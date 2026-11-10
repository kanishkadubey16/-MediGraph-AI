"""
src/models/embeddings.py
------------------------
Entity and relation embedding modules for MediGraph AI.

Provides:
  - NodeEmbedding: Learnable lookup table for all biomedical entities (nodes).
  - RelationEmbedding: Learnable lookup table for typed relations.
  - BiomedicalEmbeddings: Container module holding both node and relation embeddings.

Initializations supported:
  - xavier_uniform (default, optimal for DistMult/R-GCN)
  - kaiming_uniform
  - normal
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def init_embeddings(weight: nn.Parameter, method: str = "xavier_uniform") -> None:
    """
    Initialize an embedding parameter tensor according to the specified method.

    Parameters
    ----------
    weight : nn.Parameter
        Tensor to initialize in-place.
    method : str
        Initialization method: 'xavier_uniform', 'kaiming_uniform', or 'normal'.
    """
    if method == "xavier_uniform":
        nn.init.xavier_uniform_(weight)
    elif method == "kaiming_uniform":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    elif method == "normal":
        nn.init.normal_(weight, mean=0.0, std=0.01)
    else:
        raise ValueError(
            f"[MediGraph] Unknown initialization method '{method}'. "
            f"Expected one of ['xavier_uniform', 'kaiming_uniform', 'normal']."
        )


class NodeEmbedding(nn.Module):
    """
    Trainable node embedding table for biomedical knowledge graph entities.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph. Must be > 0.
    embedding_dim : int
        Dimensionality of each node embedding vector. Default: 64.
    init_method : str
        Method used to initialize weights. Default: 'xavier_uniform'.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 64,
        init_method: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError(f"[MediGraph] num_nodes must be positive, got {num_nodes}.")
        if embedding_dim <= 0:
            raise ValueError(f"[MediGraph] embedding_dim must be positive, got {embedding_dim}.")

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.init_method = init_method

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset embedding weights using the configured initialization method."""
        init_embeddings(self.embedding.weight, self.init_method)

    @property
    def weight(self) -> nn.Parameter:
        """Return the underlying embedding weight tensor of shape (num_nodes, embedding_dim)."""
        return self.embedding.weight

    def forward(self, indices: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        indices : Tensor, optional
            Node index tensor of shape (B,). If None, returns the entire
            embedding table of shape (num_nodes, embedding_dim).

        Returns
        -------
        Tensor
            Node embeddings of shape (B, embedding_dim) or (num_nodes, embedding_dim).
        """
        if indices is None:
            return self.embedding.weight
        return self.embedding(indices)


class RelationEmbedding(nn.Module):
    """
    Trainable relation embedding table for typed biomedical relations.

    Parameters
    ----------
    num_relations : int
        Total number of distinct relation types. Must be > 0.
    embedding_dim : int
        Dimensionality of each relation embedding vector. Default: 64.
    init_method : str
        Method used to initialize weights. Default: 'xavier_uniform'.
    """

    def __init__(
        self,
        num_relations: int,
        embedding_dim: int = 64,
        init_method: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if num_relations <= 0:
            raise ValueError(f"[MediGraph] num_relations must be positive, got {num_relations}.")
        if embedding_dim <= 0:
            raise ValueError(f"[MediGraph] embedding_dim must be positive, got {embedding_dim}.")

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.init_method = init_method

        self.embedding = nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset relation weights using the configured initialization method."""
        init_embeddings(self.embedding.weight, self.init_method)

    @property
    def weight(self) -> nn.Parameter:
        """Return the underlying relation weight tensor of shape (num_relations, embedding_dim)."""
        return self.embedding.weight

    def forward(self, indices: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        indices : Tensor, optional
            Relation index tensor of shape (B,). If None, returns the entire
            relation matrix of shape (num_relations, embedding_dim).

        Returns
        -------
        Tensor
            Relation embeddings of shape (B, embedding_dim) or (num_relations, embedding_dim).
        """
        if indices is None:
            return self.embedding.weight
        return self.embedding(indices)


class BiomedicalEmbeddings(nn.Module):
    """
    Unified container holding both entity (node) and relation embeddings.

    Parameters
    ----------
    num_nodes : int
        Number of biomedical entities.
    num_relations : int
        Number of biomedical relation types.
    embedding_dim : int
        Dimensionality of embeddings (default: 64).
    init_method : str
        Weight initialization method.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 64,
        init_method: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.node_embeddings = NodeEmbedding(num_nodes, embedding_dim, init_method)
        self.relation_embeddings = RelationEmbedding(num_relations, embedding_dim, init_method)

    def reset_parameters(self) -> None:
        """Reset all node and relation parameters."""
        self.node_embeddings.reset_parameters()
        self.relation_embeddings.reset_parameters()

