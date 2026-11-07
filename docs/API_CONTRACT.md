# MediGraph AI — API Contract

> **Version:** 0.1.0  
> **Status:** Authoritative — all team members must follow this document.  
> **Last updated:** 2026-11-07

This document defines every inter-module interface in the MediGraph AI system.  
Different developers implement different modules; this contract ensures they integrate correctly.

---

## Table of Contents

1. [File Locations](#1-file-locations)
2. [CSV Schemas](#2-csv-schemas)
3. [JSON Mapping Schemas](#3-json-mapping-schemas)
4. [PyTorch Tensor Formats](#4-pytorch-tensor-formats)
5. [Model Interface](#5-model-interface)
6. [Checkpoint Format](#6-checkpoint-format)
7. [Training Module Interface](#7-training-module-interface)
8. [Inference / Prediction Interface](#8-inference--prediction-interface)
9. [Streamlit App Interface](#9-streamlit-app-interface)
10. [Configuration Interface](#10-configuration-interface)
11. [Error Handling Standards](#11-error-handling-standards)
12. [Versioning and Compatibility](#12-versioning-and-compatibility)

---

## 1. File Locations

All paths are **relative to the project root** unless stated otherwise.  
Never hardcode absolute paths; read from `config.yaml` via `src/utils/config.py`.

```
MediGraph-AI/
│
├── config.yaml
│
├── data/
│   ├── raw/                              # Raw PrimeKG download (NOT in Git)
│   │   └── kg.csv                        # Primary PrimeKG edge file
│   ├── processed/
│   │   ├── nodes.csv                     # CONTRACT: see §2.1
│   │   ├── edges.csv                     # CONTRACT: see §2.2
│   │   ├── node_mapping.json             # CONTRACT: see §3.1
│   │   └── relation_mapping.json         # CONTRACT: see §3.2
│   └── sample/
│       ├── nodes_sample.csv              # ≤ 2000 rows subset of nodes.csv
│       └── edges_sample.csv             # ≤ 10000 rows subset of edges.csv
│
├── artifacts/
│   ├── checkpoints/
│   │   └── rgcn_epoch_{N}.pt            # CONTRACT: see §6
│   ├── predictions/
│   │   └── predictions_{timestamp}.json # CONTRACT: see §8.3
│   └── metrics/
│       └── metrics_{timestamp}.json     # CONTRACT: see §7.3
│
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py                     # loads nodes.csv + edges.csv
    │   ├── preprocess.py                 # entry point: raw → processed
    │   └── graph_builder.py             # builds PyG HeteroData / Data object
    ├── models/
    │   ├── __init__.py
    │   ├── rgcn.py                       # RGCNEncoder class
    │   └── decoder.py                    # DistMultDecoder class
    ├── training/
    │   ├── __init__.py
    │   ├── train.py                      # entry point: trains model
    │   ├── splitter.py                   # train/val/test edge split
    │   └── evaluator.py                  # MRR, Hits@K, AUC-ROC
    ├── inference/
    │   ├── __init__.py
    │   └── predict.py                    # entry point: link prediction
    └── utils/
        ├── __init__.py
        ├── config.py                     # load_config()
        ├── logger.py                     # get_logger()
        └── seed.py                       # set_seed()
```

---

## 2. CSV Schemas

### 2.1 `data/processed/nodes.csv`

**Produced by:** Member 1 (`src/data/preprocess.py`)  
**Consumed by:** Member 2, Member 3, Member 4

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `node_id` | `int64` | Required. Unique. Zero-indexed, contiguous (0 … N-1). | Canonical integer ID used in all tensors. |
| `entity_name` | `str` | Required. Non-empty. | Human-readable name (e.g. `"Alzheimer disease"`). |
| `entity_type` | `str` | Required. Must be one of the allowed types (see §2.3). | Ontological category. |
| `source_id` | `str` | Required. | Original identifier from PrimeKG (e.g. `"MONDO:0004975"`). |

**Example rows:**

```
node_id,entity_name,entity_type,source_id
0,Alzheimer disease,disease,MONDO:0004975
1,metformin,drug,DB00331
2,APOE,gene,NCBIGene:348
3,tau protein,protein,UniProt:P10636
4,memory impairment,phenotype,HP:0002354
```

**Rules:**
- `node_id` must be a contiguous integer range `[0, N)`.
- `entity_name` values must match the keys in `node_mapping.json`.
- No `NaN` or empty values in any column.
- Duplicate `node_id` values are forbidden.

---

### 2.2 `data/processed/edges.csv`

**Produced by:** Member 1 (`src/data/preprocess.py`)  
**Consumed by:** Member 2, Member 3, Member 4

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `source` | `int64` | Required. Must exist in `nodes.csv.node_id`. | Integer ID of the source node. |
| `relation` | `str` | Required. Non-empty. | Relationship type string (e.g. `"associated_with"`). |
| `target` | `int64` | Required. Must exist in `nodes.csv.node_id`. | Integer ID of the target node. |

**Example rows:**

```
source,relation,target
0,associated_with,4
0,associated_with,2
1,treats,0
2,interacts_with,3
```

**Rules:**
- `source` and `target` must both appear in `nodes.csv.node_id`.
- `relation` values must match the keys in `relation_mapping.json`.
- Self-loops (`source == target`) should be removed during preprocessing.
- No `NaN` values in any column.
- Edge direction is **preserved as-is** from PrimeKG. Do not reverse or canonicalise.
- Relationship meanings must not be altered from the original dataset.

---

### 2.3 Allowed Entity Types

The following `entity_type` values are valid in `nodes.csv`:

```
disease
drug
gene
protein
phenotype
pathway
anatomy
biological_process
molecular_function
cellular_component
exposure
```

Add new types only by updating this document and `config.yaml` (under `dataset.node_types`).

---

## 3. JSON Mapping Schemas

### 3.1 `data/processed/node_mapping.json`

**Purpose:** Maps entity names to integer node IDs (inverse of `nodes.csv`).

```json
{
  "Alzheimer disease": 0,
  "metformin": 1,
  "APOE": 2,
  "tau protein": 3,
  "memory impairment": 4
}
```

- Keys are `entity_name` strings (must match `nodes.csv` exactly, case-sensitive).
- Values are `node_id` integers.
- Every node in `nodes.csv` must appear here.

---

### 3.2 `data/processed/relation_mapping.json`

**Purpose:** Maps relation type strings to integer relation IDs.

```json
{
  "associated_with": 0,
  "treats": 1,
  "interacts_with": 2,
  "causes": 3,
  "expressed_in": 4
}
```

- Keys are `relation` strings (must match `edges.csv` exactly, case-sensitive).
- Values are contiguous integers `[0, R)` where R = number of unique relation types.
- Every relation in `edges.csv` must appear here.

---

## 4. PyTorch Tensor Formats

### 4.1 Edge Index Tensor

Standard PyTorch Geometric format.

```
Shape:  (2, num_edges)
Dtype:  torch.long
Device: same as model
Layout:
  edge_index[0, :]  →  source node IDs
  edge_index[1, :]  →  target node IDs
```

### 4.2 Edge Type Tensor

```
Shape:  (num_edges,)
Dtype:  torch.long
Values: integers in [0, num_relations)
        — mapped via relation_mapping.json
```

### 4.3 Node Embedding Matrix

Output of `RGCNEncoder.forward()`.

```
Shape:  (num_nodes, out_channels)
Dtype:  torch.float32
Indexing: row i corresponds to node_id i
```

### 4.4 Link Prediction Score Tensor

Output of `DistMultDecoder.forward()` (batch mode).

```
Shape:  (batch_size,)
Dtype:  torch.float32
Values: raw DistMult scores (NOT probabilities)
        Apply sigmoid for binary classification loss.
```

---

## 5. Model Interface

### 5.1 `RGCNEncoder`

**File:** `src/models/rgcn.py`

```python
class RGCNEncoder(torch.nn.Module):
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
    ) -> None: ...

    def forward(
        self,
        edge_index: torch.Tensor,   # shape (2, E), dtype long
        edge_type:  torch.Tensor,   # shape (E,),   dtype long
    ) -> torch.Tensor:              # shape (N, out_channels), dtype float32
        """
        Returns node embedding matrix.
        edge_index and edge_type must correspond only to
        MESSAGE-PASSING edges (training split).
        Held-out validation/test edges must NOT be passed here.
        """
        ...
```

### 5.2 `DistMultDecoder`

**File:** `src/models/decoder.py`

```python
class DistMultDecoder(torch.nn.Module):
    def __init__(
        self,
        num_relations: int,
        embedding_dim: int,
    ) -> None: ...

    def forward(
        self,
        z:           torch.Tensor,  # node embeddings (N, D)
        edge_index:  torch.Tensor,  # candidate edges  (2, B)
        edge_type:   torch.Tensor,  # relation types   (B,)
    ) -> torch.Tensor:              # scores            (B,)
        """
        Computes DistMult scores:
            score(h, r, t) = z[h] ⊙ W_r ⊙ z[t]  (element-wise, then sum)
        Returns raw scores. Apply sigmoid externally for loss computation.
        """
        ...
```

### 5.3 Combined Model Wrapper (optional helper)

```python
class MediGraphModel(torch.nn.Module):
    """Convenience wrapper: encoder + decoder as one module."""

    def __init__(self, encoder: RGCNEncoder, decoder: DistMultDecoder) -> None: ...

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type:  torch.Tensor,
    ) -> torch.Tensor: ...

    def decode(
        self,
        z:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_type:  torch.Tensor,
    ) -> torch.Tensor: ...
```

---

## 6. Checkpoint Format

**Location:** `artifacts/checkpoints/rgcn_epoch_{N}.pt`  
**Best model:** `artifacts/checkpoints/best_model.pt`  
**Format:** Python dict saved with `torch.save()` / loaded with `torch.load()`.

```python
checkpoint = {
    # Model state
    "model_state_dict":   model.state_dict(),       # OrderedDict
    "optimizer_state_dict": optimizer.state_dict(), # dict

    # Training metadata
    "epoch":              int,          # epoch number (1-indexed)
    "train_loss":         float,
    "val_loss":           float,
    "val_mrr":            float,        # Mean Reciprocal Rank on val set
    "val_hits_at_10":     float,

    # Architecture config (must match to reload)
    "model_config": {
        "num_nodes":        int,
        "num_relations":    int,
        "in_channels":      int,
        "hidden_channels":  int,
        "out_channels":     int,
        "num_layers":       int,
        "num_bases":        int,
        "dropout":          float,
    },

    # Reproducibility
    "random_seed":        int,
    "torch_version":      str,
    "pyg_version":        str,
}
```

**Loading example:**

```python
checkpoint = torch.load("artifacts/checkpoints/best_model.pt", map_location="cpu")
model_cfg  = checkpoint["model_config"]
model      = RGCNEncoder(**model_cfg)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## 7. Training Module Interface

### 7.1 Edge Splitter (`src/training/splitter.py`)

```python
def split_edges(
    edge_index:  torch.Tensor,   # (2, E)
    edge_type:   torch.Tensor,   # (E,)
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.10,
    seed:        int   = 42,
) -> dict:
    """
    Returns:
    {
        "train": {"edge_index": Tensor(2, E_train), "edge_type": Tensor(E_train)},
        "val":   {"edge_index": Tensor(2, E_val),   "edge_type": Tensor(E_val)},
        "test":  {"edge_index": Tensor(2, E_test),  "edge_type": Tensor(E_test)},
    }

    CRITICAL: val and test edges must NEVER appear in the
    edge_index passed to RGCNEncoder.forward() during training.
    """
```

### 7.2 Training Entry Point (`src/training/train.py`)

```
CLI usage:
    python src/training/train.py [--config config.yaml] [--epochs N]

Reads:
    data/processed/nodes.csv
    data/processed/edges.csv
    data/processed/node_mapping.json
    data/processed/relation_mapping.json

Writes:
    artifacts/checkpoints/rgcn_epoch_{N}.pt   (every val_interval epochs)
    artifacts/checkpoints/best_model.pt        (best val MRR)
    artifacts/metrics/metrics_{timestamp}.json
```

### 7.3 Metrics File Schema

```json
{
  "run_id": "2026-11-07T23:00:00",
  "config": { "...": "..." },
  "per_epoch": [
    {
      "epoch": 1,
      "train_loss": 0.693,
      "val_loss": 0.701,
      "val_mrr": 0.112,
      "val_hits_at_1": 0.045,
      "val_hits_at_10": 0.198,
      "val_hits_at_50": 0.321
    }
  ],
  "best": {
    "epoch": 87,
    "val_mrr": 0.341,
    "test_mrr": 0.338,
    "test_hits_at_1": 0.189,
    "test_hits_at_10": 0.512,
    "test_hits_at_50": 0.701,
    "test_auc_roc": 0.874
  }
}
```

---

## 8. Inference / Prediction Interface

### 8.1 `predict_links` Function

**File:** `src/inference/predict.py`

```python
def predict_links(
    query_node:    str,              # entity_name string (must exist in node_mapping)
    relation_type: str,              # relation string    (must exist in relation_mapping)
    model:         MediGraphModel,
    node_df:       pd.DataFrame,     # nodes.csv loaded as DataFrame
    node_mapping:  dict,             # {entity_name: node_id}
    rel_mapping:   dict,             # {relation_str: relation_id}
    edge_index:    torch.Tensor,     # training edges (2, E_train)
    edge_type:     torch.Tensor,     # training edge types (E_train,)
    top_k:         int = 20,
    score_threshold: float = 0.5,
) -> list[dict]:
    """
    Returns a ranked list of predicted target nodes.

    Return schema (list of dicts, sorted by score descending):
    [
      {
        "rank":          int,
        "target_node_id":   int,
        "target_name":   str,
        "target_type":   str,
        "relation":      str,
        "score":         float,  # raw DistMult score — NOT a probability
        "is_known_edge": bool,   # True if this edge exists in the full graph
      },
      ...
    ]

    Raises:
        KeyError: if query_node or relation_type is not in mappings
        ValueError: if model is not in eval() mode
    """
```

### 8.2 `get_node_neighborhood` Function

```python
def get_node_neighborhood(
    node_id:    int,
    edge_index: torch.Tensor,  # full graph edges (2, E_full)
    edge_type:  torch.Tensor,  # full graph edge types (E_full,)
    node_df:    pd.DataFrame,
    rel_mapping_inv: dict,     # {relation_id: relation_str}
    max_hops:   int = 1,
) -> dict:
    """
    Returns the neighbourhood of a node for visualisation.

    Return schema:
    {
      "center": {"node_id": int, "entity_name": str, "entity_type": str},
      "neighbors": [
        {
          "node_id":    int,
          "entity_name": str,
          "entity_type": str,
          "relation":   str,
          "direction":  "outgoing" | "incoming",
        },
        ...
      ]
    }
    """
```

### 8.3 Predictions File Schema

Saved to `artifacts/predictions/predictions_{timestamp}.json`.

```json
{
  "query": {
    "entity_name":   "Alzheimer disease",
    "relation_type": "associated_with",
    "top_k":         20,
    "timestamp":     "2026-11-07T23:00:00"
  },
  "model_checkpoint": "artifacts/checkpoints/best_model.pt",
  "predictions": [
    {
      "rank":            1,
      "target_node_id":  42,
      "target_name":     "APOE",
      "target_type":     "gene",
      "relation":        "associated_with",
      "score":           3.721,
      "is_known_edge":   true
    }
  ],
  "disclaimer": "Scores are raw model outputs and must NOT be interpreted as clinical probabilities or medical advice."
}
```

---

## 9. Streamlit App Interface

### 9.1 State Keys

The Streamlit session state uses the following reserved keys:

| Key | Type | Description |
|-----|------|-------------|
| `st.session_state.model` | `MediGraphModel \| None` | Loaded model (None until checkpoint loaded) |
| `st.session_state.node_df` | `pd.DataFrame \| None` | Loaded nodes.csv |
| `st.session_state.edge_index` | `torch.Tensor \| None` | Full graph edge index |
| `st.session_state.edge_type` | `torch.Tensor \| None` | Full graph edge types |
| `st.session_state.node_mapping` | `dict \| None` | entity_name → node_id |
| `st.session_state.rel_mapping` | `dict \| None` | relation_str → relation_id |
| `st.session_state.predictions` | `list[dict] \| None` | Last prediction result |

### 9.2 Page Structure

```
app/
├── main.py                        # st.navigation() entry point
└── pages/
    ├── 01_dashboard.py            # Summary statistics, graph overview
    ├── 02_entity_search.py        # Node lookup + neighbourhood viewer
    ├── 03_graph_visualisation.py  # Interactive graph (Plotly / PyVis)
    ├── 04_gnn_prediction.py       # Link prediction query UI
    └── 05_model_performance.py    # Training curves, metrics table
```

### 9.3 Visualisation Rules

- **Existing edges** (from `edges.csv`): displayed in **blue**.
- **Predicted edges** (from model): displayed in **orange dashed**.
- **Colour legend** must always be visible.
- Raw model scores must be labelled "Model Score" — never "Probability", "Confidence", or "Certainty".
- Include the disclaimer: *"This is a research prototype. Outputs are not medical advice."*

---

## 10. Configuration Interface

### 10.1 `load_config` Function

**File:** `src/utils/config.py`

```python
def load_config(path: str = "config.yaml") -> dict:
    """
    Loads and returns the config.yaml as a nested dict.
    Resolves all path keys to absolute paths relative to project root.

    Raises:
        FileNotFoundError: if config.yaml is not found
        yaml.YAMLError:    if the file is malformed
    """
```

**Usage:**

```python
from src.utils.config import load_config

cfg = load_config()
nodes_path  = cfg["paths"]["nodes_csv"]          # "data/processed/nodes.csv"
hidden_dim  = cfg["model"]["hidden_channels"]    # 256
top_k       = cfg["inference"]["top_k"]          # 20
seed        = cfg["project"]["seed"]             # 42
```

---

## 11. Error Handling Standards

All modules must follow these conventions:

### 11.1 Exception Types

| Situation | Exception to raise |
|-----------|-------------------|
| Missing required file | `FileNotFoundError` with descriptive path |
| Column missing from CSV | `KeyError` with column name and file path |
| Node not found in mapping | `KeyError` with node name and mapping file path |
| Tensor shape mismatch | `ValueError` with expected vs actual shape |
| Model not in eval() mode | `ValueError` with explicit message |
| Config key missing | `KeyError` with key name and config path |
| Invalid entity type | `ValueError` with allowed types listed |

### 11.2 Error Message Format

```python
raise FileNotFoundError(
    f"[MediGraph] nodes.csv not found at '{path}'. "
    f"Run `python src/data/preprocess.py` first."
)
```

Always prefix messages with `[MediGraph]` for easy log filtering.

### 11.3 Logging

Use `src/utils/logger.py` throughout:

```python
from src.utils.logger import get_logger
log = get_logger(__name__)
log.info("Loaded %d nodes and %d edges", len(node_df), len(edge_df))
```

Do not use bare `print()` in module code (only in CLI entry points if needed).

---

## 12. Versioning and Compatibility

- **Config version** (`project.version` in `config.yaml`) must be bumped when any schema in this document changes.
- **Checkpoint `model_config`** must match the architecture instantiated at load time — the loader must validate this.
- **CSV schemas** are fixed for v0.x. Additional columns may be added in future versions without breaking existing consumers (consumers should select only required columns by name, not by position).
- All changes to this document must be communicated to all team members before implementation.

