# MediGraph AI — System Architecture

> **Version:** 0.1.0

## Overview

MediGraph AI is a biomedical knowledge graph platform built on PrimeKG.  
It applies a Relational Graph Convolutional Network (R-GCN) to perform link prediction across biomedical entity pairs.

---

## High-Level Data Flow

```
PrimeKG (raw CSV)
        │
        ▼
[ Member 1: src/data/preprocess.py ]
        │  nodes.csv
        │  edges.csv
        │  node_mapping.json
        │  relation_mapping.json
        ▼
[ Member 2: src/models/rgcn.py + decoder.py ]
   RGCNEncoder ──────────────────────────────────────► Node Embeddings (N × D)
   DistMultDecoder ◄── (embeddings + candidate edges) ► Link Scores (B,)
        │
        ▼
[ Member 3: src/training/train.py ]
   Edge split (train / val / test)
   Negative sampling
   Training loop
   Checkpoint saving ──► artifacts/checkpoints/best_model.pt
   Metrics saving   ──► artifacts/metrics/metrics_*.json
        │
        ▼
[ Member 3: src/inference/predict.py ]
   predict_links()
   get_node_neighborhood()
        │
        ▼
[ Member 4: app/ ]
   Streamlit dashboard
   Entity search
   Graph visualisation
   GNN prediction UI
   Model performance dashboard
```

---

## Component Descriptions

### `src/data/`
Responsible for downloading, cleaning, and transforming PrimeKG into the standard CSV format and PyTorch tensors.

### `src/models/`
Defines the R-GCN encoder and DistMult decoder as `torch.nn.Module` subclasses.

### `src/training/`
Implements the training loop, edge splitting, negative sampling, evaluation metrics (MRR, Hits@K, AUC-ROC), and checkpoint management.

### `src/inference/`
Loads a trained checkpoint and exposes `predict_links()` for link prediction queries and `get_node_neighborhood()` for graph traversal.

### `src/utils/`
Shared utilities: config loader, logger, random seed setter.

### `app/`
Multi-page Streamlit application. All pages load data through `src/` modules; no direct file I/O in the app layer.

---

## R-GCN Architecture

```
Node IDs (0 … N-1)
        │
        ▼ Learnable Embedding Table (N × in_channels)
        │
  ┌─────▼──────────────────────────┐
  │   R-GCN Layer 1                │
  │   For each relation r:         │
  │     H' += A_r · H · W_r       │
  │   (W_r via basis decomposition)│
  └─────┬──────────────────────────┘
        │ ReLU + Dropout
  ┌─────▼──────────────────────────┐
  │   R-GCN Layer 2                │
  └─────┬──────────────────────────┘
        │
        ▼ Node Embeddings Z (N × out_channels)
        │
  ┌─────▼──────────────────────────┐
  │   DistMult Decoder             │
  │   score(h,r,t) = z_h⊙W_r⊙z_t │
  └─────┬──────────────────────────┘
        │
        ▼ Link Scores (B,) — raw, not probabilities
```

---

## Security and Ethics Notes

- Raw data files are excluded from Git (`.gitignore`).
- No patient-identifiable information is processed.
- Model scores are labelled as "Model Score", not as clinical probabilities.
- The disclaimer "Research prototype — not medical advice" must appear in the UI.

