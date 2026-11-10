"""
src.models — Relational Graph Convolutional Network and DistMult modules.

Exposes:
  - RGCNEncoder, MedicalRGCN: Relational GCN encoder
  - DistMultDecoder, MedicalDistMultDecoder: Knowledge graph triple decoder
  - MediGraphModel, MedicalRGCNModel: Combined encoder + decoder wrapper
  - NodeEmbedding, RelationEmbedding, BiomedicalEmbeddings: Embedding tables
"""

from src.models.embeddings import (
    NodeEmbedding,
    RelationEmbedding,
    BiomedicalEmbeddings,
)
from src.models.rgcn import (
    RGCNEncoder,
    MedicalRGCN,
)
from src.models.decoder import (
    DistMultDecoder,
    MedicalDistMultDecoder,
    MediGraphModel,
    MedicalRGCNModel,
)

__all__ = [
    "NodeEmbedding",
    "RelationEmbedding",
    "BiomedicalEmbeddings",
    "RGCNEncoder",
    "MedicalRGCN",
    "DistMultDecoder",
    "MedicalDistMultDecoder",
    "MediGraphModel",
    "MedicalRGCNModel",
]
