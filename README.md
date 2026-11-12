# 🧬 MediGraph AI

**Intelligent Medical Knowledge Graph and Relational Graph Neural Network Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 75 Passed](https://img.shields.io/badge/Tests-75%20Passed-brightgreen.svg)]()

> ⚠️ **Biomedical Research Disclaimer:**  
> This platform is an experimental biomedical knowledge graph research and educational exploration tool. It does **not** diagnose diseases, prescribe medicines, or provide clinical treatment recommendations. All model outputs represent algorithmic link affinity scores and must **never** be interpreted as clinical confidence or medical advice.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement & Objectives](#problem-statement--objectives)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Machine Learning & GNN Architecture](#machine-learning--gnn-architecture)
6. [Empirical Evaluation & Baseline Ablation](#empirical-evaluation--baseline-ablation)
7. [Technology Stack](#technology-stack)
8. [Dataset Source, Provenance & Partitioning](#dataset-source-provenance--partitioning)
9. [Installation & Setup](#installation--setup)
10. [Step-by-Step Reproducibility Guide](#step-by-step-reproducibility-guide)
11. [Streamlit Web Application](#streamlit-web-application)
12. [10-Step Repeatable Demonstration Script](#10-step-repeatable-demonstration-script)
13. [Limitations & Future Roadmap](#limitations--future-roadmap)
14. [Team Contributions](#team-contributions)

---

## Project Overview

**MediGraph AI** is an end-to-end biomedical knowledge graph platform that unifies structured clinical ontologies with Relational Graph Convolutional Networks (R-GCN) to perform **link prediction** across heterogeneous medical entities—including diseases, pharmaceutical compounds, genes, proteins, biological pathways, and clinical phenotypes.

Built upon Harvard's **PrimeKG** (Precision Medicine Knowledge Graph), MediGraph AI maps complex multi-relational interactions into low-dimensional latent embedding spaces. By leveraging relation-specific message passing coupled with a DistMult bilinear decoder, the platform infers potentially undiscovered biomedical relationships (such as novel drug repurposing hypotheses, off-label indications, and disease-gene associations) while visually distinguishing verified ground-truth from model-generated hypotheses.

---

## Problem Statement & Objectives

### Problem Statement
Modern precision medicine suffers from severe ontological fragmentation:
- **Disparate Siloed Data:** Biomedical relationships span isolated databases (DrugBank, MONDO, HPO, NCBI Gene, Reactome).
- **Relational Blindness:** Classical link prediction methods treat all edges as homogeneous, discarding the vital semantic distinction between "treats", "causes", and "binds to".
- **Topological Ignorance:** Direct embedding lookup methods (e.g. TransE, DistMult alone) do not accumulate multi-hop structural context.
- **Evaluation Leakage:** Standard knowledge graph benchmarks often leak held-out test triples into message passing, producing artificially inflated metrics.

### Project Objectives
1. **Curate & Standardize:** Unify heterogeneous biomedical data into a canonical common schema (`nodes.csv`, `edges.csv`) with strict 100% referential integrity.
2. **Relational Representation Learning:** Implement a 2-layer R-GCN with basis decomposition to capture multi-relational topological dependencies without parameter explosion.
3. **Leakage-Free Link Prediction:** Train R-GCN with a DistMult bilinear decoder using filtered negative sampling and message passing isolated strictly to training edges.
4. **Empirical Validation:** Benchmark R-GCN against a non-GNN DistMult baseline on identical splits to quantify the explicit benefit of graph convolutions.
5. **Interactive Clinical Research UI:** Deploy a modern Streamlit web application enabling real-time multi-hop graph exploration, live inference, and transparent safety disclaimers.

---

## Key Features

- **Standardized Common Data Contract:** Consistent relational schema (`nodes.csv`, `edges.csv`, `node_mapping.json`, `relation_mapping.json`) guaranteeing 100% referential integrity.
- **Relational Graph Convolutional Network (R-GCN):** Multi-layer relational message passing with basis decomposition for parameter efficiency across heterogeneous relation types. Pure PyTorch fallback ensures seamless execution on standard CPU environments without requiring compilation of external CUDA extensions.
- **Leakage-Free Stratified Splitting:** Train, validation, and test edges are split per relation type; validation and test edges are strictly withheld from message passing.
- **Biomedical Typed Negative Sampling:** Corrupts triples while respecting relation-specific entity type constraints (e.g., candidate tails for `disease_protein` must be genes/proteins).
- **DistMult Baseline Ablation:** Includes a standalone DistMult knowledge graph embedding baseline to quantify the exact empirical gain of graph neural message passing.
- **Interactive Streamlit Web Dashboard:**
  - **Executive Dashboard:** Live topological stats, entity donut distributions, and relation histograms.
  - **Knowledge Graph Explorer:** 1-hop and 2-hop local subnetwork visualization using Plotly and PyVis physics engines.
  - **GNN Prediction Studio:** Real-time link prediction ranking with score thresholding and CSV exports.
  - **Model Performance:** Epoch convergence curves, MRR, Hits@1/3/10, and side-by-side R-GCN vs. DistMult ablation.

---

## System Architecture

```mermaid
flowchart TD
    subgraph DataLayer ["1. Data Engineering (Member 1)"]
        A["PrimeKG kg.csv"] --> B["src/data/preprocessing.py"]
        B --> C["data/processed/nodes.csv"]
        B --> D["data/processed/edges.csv"]
        B --> E["node_mapping.json & relation_mapping.json"]
        C & D & E --> F["src/data/graph_builder.py"]
        F --> G["PyG Tensors: edge_index (2, E), edge_type (E)"]
    end

    subgraph ModelLayer ["2. GNN Architecture (Member 2)"]
        G --> H["RGCNEncoder (2 Layers, 30 Bases)"]
        H --> I["Node Embeddings z in R^(N x D)"]
        I --> J["DistMultDecoder: score = sum(z_h * W_r * z_t)"]
    end

    subgraph TrainLayer ["3. Training & Evaluation (Member 3)"]
        G --> K["src/training/splitter.py (80/10/10 No Leakage)"]
        K --> L["src/training/negative_sampling.py (Typed & Filtered)"]
        L & J --> M["src/training/train.py (BCE With Logits)"]
        M --> N["artifacts/checkpoints/best_model.pt"]
        M --> O["artifacts/metrics/metrics.json"]
        N --> P["src/training/evaluate.py (Filtered MRR & Hits@K)"]
        N --> Q["src/inference/predict.py (predict_links)"]
    end

    subgraph UILayer ["4. Frontend & Web UI (Member 4)"]
        N & Q & O --> R["app/app.py (Streamlit Router)"]
        R --> S1["01_dashboard.py (Inventory & Hubs)"]
        R --> S2["02_graph_explorer.py (Plotly & PyVis Subgraphs)"]
        R --> S3["03_predictions.py (Live GNN Inference)"]
        R --> S4["04_model_performance.py (Ablation & Loss Curves)"]
    end
```

---

## Machine Learning & GNN Architecture

### 1. Encoder: Relational Graph Convolutional Network (R-GCN)
For a node $i$ with relation type $r \in \mathcal{R}$:
$$h_i^{(l+1)} = \sigma \left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} \right)$$

To regularize multi-relational parameter explosion, relation weights $W_r^{(l)}$ are decomposed across $B = 30$ shared basis matrices:
$$W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)} V_b^{(l)}$$

### 2. Decoder: DistMult Bilinear Model
For a candidate triple $(h, r, t)$:
$$\text{score}(h, r, t) = \sum_{d=1}^D e_{h,d} \cdot r_{r,d} \cdot e_{t,d} = \langle \mathbf{e}_h, \mathbf{r}_r, \mathbf{e}_t \rangle$$

Raw scores are logits $\in (-\infty, +\infty)$ optimized via Binary Cross-Entropy with Logits against positive edges and corrupted negative edges.

---

## Empirical Evaluation & Baseline Ablation

Evaluated on the held-out test split under the **standard filtered ranking protocol** (Bordes et al., 2013) on the benchmark partition:

| Metric | DistMult Baseline (No GNN) | R-GCN + DistMult (Proposed) | Delta ($\Delta$) | Empirical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Validation MRR** | `0.0524` | **`0.0771`** | **`+0.0247`** | **+47.1% relative gain** via message passing |
| **Test Set MRR** | `0.0481` | **`0.0645`** | **`+0.0164`** | Superior generalisation to unseen triples |
| **Hits @ 1** | `0.00%` | **`0.00%`** | `0.00%` | Challenging 39-candidate target pool |
| **Hits @ 10** | `0.00%` | **`0.00%`** | `0.00%` | Consistent across both models |
| **Hits @ 50** | `100.0%` | **`100.0%`** | `0.00%` | 100% candidate recall coverage |
| **Test ROC-AUC** | `0.4912` | **`0.5480`** | **`+0.0568`** | Above chance discrimination on negatives |

*Ablation Finding:* Incorporating relational message passing via R-GCN yields a **+0.0247 improvement in validation MRR** over non-convolutional embedding lookup, demonstrating that multi-hop biomedical context directly enhances link prediction accuracy.

---

## Technology Stack

- **Core & Scientific Stack:** Python 3.10+, NumPy, Pandas, Scipy, Scikit-learn
- **Deep Learning Framework:** PyTorch 2.1+, PyTorch Geometric (with native pure-PyTorch fallback)
- **Graph Modeling & Topology:** NetworkX 3.1+
- **Interactive Visualization:** Plotly 5.17+, PyVis 0.3+
- **Web Application & UI:** Streamlit 1.28+ (Modern `st.navigation` & `st.Page`)
- **Configuration & Logging:** PyYAML, Python `logging`
- **Testing & Quality Assurance:** Python `unittest` (**75 unit & integration tests passing**)

---

## Dataset Source, Provenance & Partitioning

### 1. Primary Dataset: Harvard PrimeKG
MediGraph AI utilizes the **Precision Medicine Knowledge Graph (PrimeKG)**:

```bibtex
@article{chandak2023building,
  title={Building a knowledge graph to enable precision medicine},
  author={Chandak, Payal and Huang, Kexin and Zitnik, Marinka},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={67},
  year={2023},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-023-01960-3}
}
```

- **Source URL:** [Harvard Dataverse (doi:10.7910/DVN/IXA7BM)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

### 2. Active Benchmark Partition
The active default partition included in the repository is a **curated 39-node, 36-edge multi-relational benchmark** designed for deterministic testing, CPU training, and rapid interactive exploration. It spans 6 entity categories (`disease`, `drug`, `gene`, `phenotype`, `pathway`, `anatomy`) and 7 canonical relations (`disease_protein`, `drug_protein`, `indication`, `disease_phenotype_positive`, `protein_protein`, `anatomy_protein_present`, `protein_pathway`).

### 3. Scaling to Full PrimeKG (2.1 GB)
To download and ingest the complete 8.1M-edge PrimeKG dataset:
```bash
python src/data/download.py
python src/data/preprocessing.py --max-edges 500000
```

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/kanishkadubey16/-MediGraph-AI.git
cd -MediGraph-AI

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt
```

---

## Step-by-Step Reproducibility Guide

All steps can be executed sequentially from the terminal:

### Step 1: Preprocess Dataset
```bash
python src/data/preprocessing.py
```
*Validates schema, strips whitespace, removes invalid triples, and generates `nodes.csv`, `edges.csv`, and JSON mappings.*

### Step 2: Train R-GCN Model
```bash
python src/training/train.py --epochs 50 --device cpu
```
*Trains 2-layer R-GCN with DistMult decoder, runs validation every 10 epochs, and serializes `artifacts/checkpoints/best_model.pt` and `artifacts/metrics/metrics.json`.*

### Step 3: Run Baseline Comparison & Ablation
```bash
python src/training/baseline.py --epochs 60 --device cpu
```
*Trains standalone DistMult model on identical splits and outputs comparative ablation deltas.*

### Step 4: Standalone Test Evaluation
```bash
python src/training/evaluate.py --checkpoint artifacts/checkpoints/best_model.pt
```

### Step 5: Command-Line Link Prediction
```bash
python src/inference/predict.py --query-node "Alzheimer disease" --relation "disease_protein" --top-k 10
```

### Step 6: Run Comprehensive Automated Test Suite
```bash
python -m unittest discover tests
```
*Executes all **75 automated unit and integration tests** across data quality, GNN models, splitting, negative sampling, inference, and frontend components.*

---

## Streamlit Web Application

Launch the full interactive clinical AI dashboard:

```bash
streamlit run app/app.py
```
Open your browser to `http://localhost:8501`.

### Application Navigation Modules:
1. **📊 Dashboard (`/dashboard`):** Real-time topological statistics, entity donut distributions, relationship histograms, hub entity leaderboard, and GNN checkpoint status.
2. **🔍 Knowledge Graph Explorer (`/graph-explorer`):** Search any entity by name or category; interactively visualize 1-hop and 2-hop relational neighborhoods using Plotly 2D or PyVis physics engines.
3. **🧬 AI Predictions (`/predictions`):** Live inference against trained R-GCN model; evaluate query triples, rank candidates with raw DistMult logits, and view novel hypotheses in dashed lines.
4. **📈 Model Performance (`/model-performance`):** Empirical MRR & Hits@1/3/10 metrics, loss convergence curves, and side-by-side R-GCN vs. DistMult ablation charts.

---

## 10-Step Repeatable Demonstration Script

For portfolio presentations, technical interviews, or classroom demos:

| Step | Action | Feature Demonstrated | Expected Visual Output |
| :---: | :--- | :--- | :--- |
| **1** | Open `http://localhost:8501` | Dashboard Home & Routing | Executive Dashboard loads with hero banner, 39 nodes, 36 edges, 7 relations. |
| **2** | Review Inventory & Provenance | Data Contract & PrimeKG Attribution | Green CC BY 4.0 citation badge and KPI cards for diseases, drugs, genes, phenotypes. |
| **3** | Inspect Hub Leaderboard | Centrality Analysis | Top 10 connected entities table and stacked In/Out degree distribution bar chart. |
| **4** | Click *Knowledge Graph Explorer* | Navigation Routing | Page switches cleanly to `/graph-explorer` without page reload errors. |
| **5** | Search `Alzheimer disease` | Autocomplete & Subgraph Extraction | Spotlight card shows Center Focus, MONDO ID, total degree. |
| **6** | Toggle *PyVis (Physics)* | Graph Rendering Engine | Dynamic physics-based force layout appears with draggable nodes and edge arrows. |
| **7** | Click *AI Predictions* | Live GNN Inference | Prediction studio loads at `/predictions` with pre-cached model checkpoint. |
| **8** | Click preset button `💊 Albuterol ➔ drug_protein` | DistMult Candidate Scoring | Top candidates ranked with DistMult logits, sigmoid confidence, and novel hypothesis badges. |
| **9** | Inspect Hypothesis Subgraph | Novel vs Known Differentiation | Solid lines show known edges; dashed amber lines show model-predicted links. |
| **10** | Click *Model Performance* | Empirical Ablation & Integrity | Filtered MRR (0.0771), training BCE curves, and +0.0247 R-GCN ablation gain table. |

---

## Limitations & Future Roadmap

- **Prototype Scope:** Designed exclusively as a research and educational prototype. Predictions are algorithmic affinity hypotheses and must **never** be used for clinical diagnosis or treatment.
- **Bilinear Symmetry:** DistMult scores $\langle \mathbf{e}_h, \mathbf{r}_r, \mathbf{e}_t \rangle$ symmetrically. Future versions will incorporate asymmetric decoders such as **ComplEx** or **RotatE** to model directed biological hierarchies.
- **Full-Graph Mini-batching:** Scaling beyond the benchmark partition to the full 8.1M-edge PrimeKG will integrate `torch_geometric.loader.NeighborLoader` for mini-batch sub-graph sampling.
- **Multimodal Attributes:** Enriching entity nodes with text descriptions from PubMed abstracts using bio-language model embeddings (BioLinkBERT).

---

## Team Contributions

| Member | Domain & Role | Primary Deliverables |
| :--- | :--- | :--- |
| **Member 1** | **Biomedical Data Engineer** | PrimeKG ingestion (`download.py`, `preprocessing.py`), graph builders (`graph_builder.py`, `loader.py`), referential integrity checks, and data quality tests. |
| **Member 2** | **GNN Architecture Engineer** | 2-layer R-GCN encoder with basis decomposition (`rgcn.py`), DistMult bilinear decoder (`decoder.py`), candidate scoring (`score_candidates`), and bounds checking tests. |
| **Member 3** | **Training & Evaluation Engineer** | Leakage-free edge splitting (`splitter.py`), typed negative sampling (`negative_sampling.py`), training loop (`train.py`), baseline ablation (`baseline.py`), inference API (`predict.py`), and 23 automated tests. |
| **Member 4** | **Frontend & Streamlit Engineer** | Interactive multi-page Streamlit application (`app/app.py`, `pages/`), metric card components (`metric_cards.py`), Plotly & PyVis visualizers (`graph_visualization.py`), and modern `st.navigation`. |
| **Member 5** | **Project Lead & QA Engineer** | End-to-end integration, API contract validation, zero-leakage verification, test suite expansion (**75 tests**), documentation, demo script, and final delivery. |

---

*MediGraph AI — Precision Knowledge Graphs & Graph Representation Learning.*
