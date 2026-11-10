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

## R-GCN & DistMult Model Deep-Dive

### 1. What is an R-GCN?
The **Relational Graph Convolutional Network (R-GCN)** (Schlichtkrull et al., 2018) is a specialized graph neural network architecture designed for multi-relational directed knowledge graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$. Unlike standard GCNs which assume homogeneous, single-typed edges, R-GCN performs relation-specific transformations on incoming neighbor messages.

### 2. Message Passing Mechanics
At layer $l$, the forward message-passing update for entity $i \in \mathcal{V}$ is defined as:

$$h_i^{(l+1)} = \sigma \left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} \right)$$

where:
- $\mathcal{N}_i^r$ denotes the set of immediate neighbors of node $i$ under relation $r \in \mathcal{R}$.
- $W_r^{(l)} \in \mathbb{R}^{d^{(l+1)} \times d^{(l)}}$ is a relation-specific transformation matrix.
- $W_0^{(l)}$ represents the self-loop transformation accumulating the node's prior state.
- $c_{i,r} = |\mathcal{N}_i^r|$ is a normalization constant (or degree normalizer).
- $\sigma(\cdot)$ is a non-linear activation function (ReLU for intermediate layers, linear for the final layer).

#### Basis Decomposition
To prevent parameter explosion when scaling across dozens of biomedical relation types, relation matrices $W_r^{(l)}$ are parameterized via basis decomposition:

$$W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)}$$

where $V_b^{(l)}$ are $B$ shared basis matrices and $a_{rb}^{(l)}$ are relation-specific coefficients. This acts as an effective regularizer, encouraging weight sharing across structurally similar biological relationships.

### 3. Why Relation Types Matter in Biomedicine
In a medical knowledge graph, edges possess radically different clinical meanings:
- `drug_treats_disease` indicates a therapeutic indication.
- `drug_contraindicated_disease` indicates a dangerous, prohibited interaction.
- `disease_associated_gene` indicates genetic risk or causal etiology.
- `protein_interacts_protein` indicates physical binding affinity.

Collapsing these edges into a single homogeneous graph would confound contradictory biomedical facts (e.g., treating vs. contraindicating). R-GCN preserves semantic distinctness during aggregation.

### 4. What Node Embeddings Represent
Because PrimeKG does not furnish uniform raw tabular features across all 10 entity scales, the model initializes a trainable lookup table $E \in \mathbb{R}^{N \times d_{in}}$ (`NodeEmbedding`).
Following 2 layers of relational convolution, each node's final embedding vector $z_i \in \mathbb{R}^{d_{out}}$ encapsulates:
1. **Intrinsic identity:** Learned representations tailored to the entity's functional behavior.
2. **2-hop multi-relational context:** Pharmacological, genetic, and phenotypic neighborhood context (e.g., a drug embedding encodes which pathways its targets modulate).

### 5. Why DistMult Decoder?
DistMult (Yang et al., 2015) is a bilinear scoring decoder that models each relation as a diagonal matrix $W_r = \text{diag}(r) \in \mathbb{R}^{d \times d}$.
For a candidate triple $(h, r, t)$ where $h, t \in \mathcal{V}$ and $r \in \mathcal{R}$:

$$\text{score}(h, r, t) = z_h^\top \text{diag}(r) z_t = \sum_{d=1}^{D} z_{h}[d] \cdot r[d] \cdot z_{t}[d]$$

**Why DistMult for MediGraph AI?**
- **Symmetry & Efficiency:** DistMult requires only $\mathcal{O}(D)$ parameters per relation type, computing scores with fast element-wise multiplication and dot products.
- **GNN Synergy:** While vanilla DistMult alone cannot capture asymmetric directed relations well, pairing it with an R-GCN encoder eliminates this limitation because directional information is already baked into the non-symmetric node representations $z_h$ and $z_t$.

### 6. Triple Scoring & Logit Interpretation
- **Raw Scores (Logits):** $\text{score}(h, r, t) \in (-\infty, +\infty)$ are uncalibrated bilinear compatibility scores.
- **Loss Computation:** $\hat{y} = \sigma(\text{score}(h, r, t)) \in (0, 1)$ is fed into Binary Cross-Entropy loss (`BCEWithLogitsLoss`) with positive edges labeled $1$ and corrupted negative edges labeled $0$.
- **Clinical Non-Equivalence:** In user interfaces, scores are strictly labeled **"Model Score"** — never "Probability" or "Confidence" — to avoid misleading clinical interpretations.

---

## Model Architecture & Tensor Dimensions

```
Entity Index (Head/Tail) ──► NodeEmbedding: (N, in_channels)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ RGCN Layer 1 (RGCNConv with basis decomposition)                        │
│ Input:  x ∈ (N, in_channels), edge_index ∈ (2, E), edge_type ∈ (E,)    │
│ Output: h^(1) ∈ (N, hidden_channels)                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ ReLU + Dropout(p=0.1)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ RGCN Layer 2 (RGCNConv with basis decomposition)                        │
│ Input:  h^(1) ∈ (N, hidden_channels), edge_index, edge_type            │
│ Output: Node Embeddings Z ∈ (N, out_channels)                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
               Candidate Triple (h, r, t) with r ∈ (B,)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ DistMult Decoder (Bilinear Scoring)                                    │
│ Input:  z_h ∈ (B, out_channels), W_r ∈ (B, out_channels), z_t ∈ (B, out_channels)│
│ Operation: sum(z_h * W_r * z_t, dim=-1)                                │
│ Output: Raw Logit Scores ∈ (B,)                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Tensor Dimensions Summary
| Stage | Component | Input Shape | Output Shape | Dtype |
| :--- | :--- | :--- | :--- | :--- |
| **Node Embeddings** | `NodeEmbedding` | None (table) | `(num_nodes, in_channels)` | `torch.float32` |
| **Relation Embeddings**| `RelationEmbedding`| None (table) | `(num_relations, in_channels)`| `torch.float32` |
| **Layer 1** | `RGCNConv` | `(N, in_channels)`, `(2, E)`, `(E,)` | `(N, hidden_channels)` | `torch.float32` |
| **Layer 2** | `RGCNConv` | `(N, hidden_channels)`, `(2, E)`, `(E,)`| `(N, out_channels)` | `torch.float32` |
| **Batch Triple Scoring**| `DistMultDecoder` | `z`: `(N, D)`, `edges`: `(2, B)`, `rel`: `(B,)` | `(B,)` | `torch.float32` |
| **Candidate Ranking** | `score_candidates`| `head_id`: `int`, `rel_id`: `int`, `z`: `(N, D)`| `(num_nodes,)` | `torch.float32` |


---

## Security and Ethics Notes

- Raw data files are excluded from Git (`.gitignore`).
- No patient-identifiable information is processed.
- Model scores are labelled as "Model Score", not as clinical probabilities.
- The disclaimer "Research prototype — not medical advice" must appear in the UI.

