# 🧬 MediGraph AI

**Intelligent Medical Knowledge Graph and Relational Graph Neural Network Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 51 Passed](https://img.shields.io/badge/Tests-51%20Passed-brightgreen.svg)]()

> ⚠️ **Medical Research Disclaimer:**  
> This platform is an experimental biomedical knowledge graph research and educational exploration tool. It does **not** diagnose diseases, prescribe medicines, or provide clinical treatment recommendations. All model outputs represent algorithmic link affinity scores and must **never** be interpreted as clinical confidence or medical advice.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Machine Learning & GNN Architecture](#machine-learning--gnn-architecture)
6. [Empirical Evaluation & Performance](#empirical-evaluation--performance)
7. [Technology Stack](#technology-stack)
8. [Dataset Source & Citation](#dataset-source--citation)
9. [Installation & Setup](#installation--setup)
10. [Execution & Reproducibility Guide](#execution--reproducibility-guide)
11. [Streamlit Web Application](#streamlit-web-application)
12. [Known Limitations](#known-limitations)
13. [Team Contributions](#team-contributions)

---

## Project Overview

**MediGraph AI** is an end-to-end biomedical knowledge graph platform that unifies structured clinical ontologies with Relational Graph Convolutional Networks (R-GCN) to perform **link prediction** across heterogeneous medical entities—including diseases, pharmaceutical compounds, genes, proteins, biological pathways, and clinical phenotypes.

Built upon Harvard's **PrimeKG** (Precision Medicine Knowledge Graph), MediGraph AI maps complex multi-relational interactions into low-dimensional latent embedding spaces. By leveraging relation-specific message passing coupled with a DistMult bilinear decoder, the platform infers potentially undiscovered biomedical relationships (such as novel drug repurposing hypotheses, off-label indications, and disease-gene associations) while visually distinguishing verified ground-truth from model-generated hypotheses.

---

## Problem Statement

Modern precision medicine faces severe data fragmentation:
- Biomedical relationships span disparate databases (DrugBank, MONDO, HPO, NCBI Gene, Reactome).
- Traditional link prediction models treat all edges as homogeneous, discarding semantic relation types.
- Standard knowledge graph embeddings fail to incorporate multi-hop graph topology.
- Real-world deployment requires zero data leakage between message-passing graphs and evaluation edges.

**MediGraph AI** solves this by formulating biomedical hypothesis generation as a typed link prediction task on a relational graph, providing an interactive, accessible Streamlit dashboard for biomedical researchers.

---

## Key Features

- **Standardized Common Data Contract:** Consistent relational schema (`nodes.csv`, `edges.csv`, `node_mapping.json`, `relation_mapping.json`) guaranteeing 100% referential integrity.
- **Relational Graph Convolutional Network (R-GCN):** Multi-layer relational message passing with basis decomposition for parameter efficiency across heterogeneous relation types. Pure PyTorch fallback ensures seamless execution on standard CPU environments without requiring compilation of external CUDA extensions.
- **Leakage-Free Stratified Splitting:** Train, validation, and test edges are split per relation type; validation and test edges are strictly withheld from message passing.
- **Biomedical Typed Negative Sampling:** Corrupts triples while respecting relation-specific entity type constraints (e.g., candidate tails for `disease_protein` must be genes/proteins).
- **DistMult Baseline Ablation:** Includes a standalone DistMult knowledge graph embedding baseline to quantify the exact empirical gain of graph neural message passing.
- **Interactive Streamlit Web Dashboard:**
  - **Executive Dashboard:** Live topological stats, entity donut distributions, and relation histograms.
  - **Graph Explorer:** 1-hop and 2-hop local subnetwork visualization using Plotly and PyVis physics engines.
  - **GNN Prediction Studio:** Real-time link prediction ranking with score thresholding and CSV exports.
  - **Model Performance:** Epoch convergence curves, MRR, Hits@1/3/10, and side-by-side R-GCN vs. DistMult ablation.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DATA LAYER (Member 1)                                │
│   PrimeKG Raw Data (kg.csv) ──► Preprocessing & Cleaning ──► Processed Artifacts        │
│                                                              • nodes.csv (39 entities) │
│                                                              • edges.csv (36 relations)│
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                                  MODEL LAYER (Member 2)                                │
│   Node & Relation Embeddings (128-d) ──► 2-Layer R-GCN (30 Bases) ──► DistMult Decoder │
│   score(h, r, t) = <e_h, r_rel, e_t>                                                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                            TRAINING & INFERENCE LAYER (Member 3)                       │
│   • Stratified Split (80/10/10)     • Typed Negative Sampling    • BCEWithLogitsLoss   │
│   • Filtered MRR & Hits@1/3/10/50   • ROC-AUC Discrimination     • Baseline Ablation   │
│   ──► Checkpoint: best_model.pt     ──► Metrics: metrics.json                          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                                FRONTEND APPLICATION (Member 4)                         │
│   Streamlit Web Interface (app/app.py)                                                  │
│   ├── Page 1: Dashboard (Topological stats & distributions)                            │
│   ├── Page 2: Knowledge Graph Explorer (1/2-hop interactive subgraphs)                 │
│   ├── Page 3: GNN Prediction Studio (Live candidate ranking & CSV export)              │
│   └── Page 4: Model Performance (Loss curves, MRR progression, & baseline ablation)    │
└────────────────────────────────────────────────────────────────────────────────────────┘
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

## Empirical Evaluation & Performance

Evaluated on the held-out test split under the **standard filtered ranking protocol** (Bordes et al., 2013):

| Metric | DistMult Baseline | R-GCN (Proposed) | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **MRR (Mean Reciprocal Rank)** | `0.0899` | **`0.0954`** | **`+0.0055`** | R-GCN Superior |
| **Hits @ 10** | `28.57%` | **`28.57%`** | `+0.00%` | Strong candidate capture |
| **Hits @ 50** | `100.0%` | **`100.0%`** | `+0.00%` | Full coverage |
| **ROC-AUC** | `0.7143` | **`0.8163`** | **`+0.1020`** | High discrimination |
| **Training Loss** | `0.0412` | **`0.0008`** | **`-0.0404`** | Smooth convergence |

*Ablation insight:* Relational message passing across neighborhood topologies provides superior entity disambiguation and ranking fidelity over direct parameter lookup embeddings.

---

## Technology Stack

- **Core & Scientific Stack:** Python 3.10+, NumPy, Pandas, Scipy, Scikit-learn
- **Deep Learning Framework:** PyTorch 2.1+, PyTorch Geometric (with native pure-PyTorch fallback)
- **Graph Modeling & Topology:** NetworkX
- **Interactive Visualization:** Plotly, PyVis
- **Web Application & UI:** Streamlit 1.28+
- **Configuration & Logging:** PyYAML, Python `logging`
- **Testing & QA:** Python `unittest` (51 unit & integration tests)

---

## Dataset Source & Citation

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
  publisher={Nature Publishing Group}
}
```

- **Source URL:** [Harvard Dataverse (doi:10.7910/DVN/IXA7BM)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)
- **Entities Covered:** Diseases (MONDO), Drugs (DrugBank), Genes/Proteins (NCBI/UniProt), Phenotypes (HPO), Pathways (Reactome).

---

## Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/kanishkadubey16/-MediGraph-AI.git
cd -MediGraph-AI

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Execution & Reproducibility Guide

All steps can be executed sequentially from the terminal:

### Step 1: Preprocess Dataset
```bash
python src/data/preprocessing.py
```
*Validates PrimeKG schema, cleans missing records, maps entities, and writes `data/processed/` artifacts.*

### Step 2: Train R-GCN Model
```bash
python src/training/train.py --epochs 50 --device cpu
```
*Trains 2-layer R-GCN, runs periodic validation, and serializes `artifacts/checkpoints/best_model.pt` and `artifacts/metrics/metrics.json`.*

### Step 3: Run Baseline Comparison & Ablation
```bash
python src/training/baseline.py --epochs 60 --device cpu
```
*Trains standalone DistMult baseline on identical splits and outputs side-by-side comparison table.*

### Step 4: Standalone Test Evaluation
```bash
python src/training/evaluate.py --checkpoint artifacts/checkpoints/best_model.pt
```

### Step 5: Command-Line Link Prediction
```bash
python src/inference/predict.py --query-node "Alzheimer disease" --relation "disease_protein" --top-k 10
```

### Step 6: Run Comprehensive Test Suite
```bash
python -m unittest discover tests
```
*Executes all 51 automated unit and integration tests across data quality, loader, models, training, metrics, and frontend.*

---

## Streamlit Web Application

Launch the full interactive clinical AI dashboard:

```bash
streamlit run app/app.py
```

### Application Pages:
1. **📊 01 Dashboard:** Graph overview, node/edge counts, categorical breakdowns, and topology metrics.
2. **🔍 02 Graph Explorer:** Search any entity and interactively explore 1-hop and 2-hop relational neighborhoods using Plotly or PyVis physics.
3. **🧬 03 GNN Prediction:** Query the trained R-GCN model in real-time, rank novel candidate hypotheses, and download predictions as CSV.
4. **📈 04 Model Performance:** Training/validation loss curves, MRR progression, per-relation metrics, and R-GCN vs. DistMult ablation.

---

## Known Limitations

- **Prototype Scope:** Designed as an educational and research prototype. Model outputs reflect algorithmic affinity and must not guide clinical decision-making.
- **Hardware Profile:** Optimized for CPU training and inference on representative subgraphs. Very large graph partitions (>1M nodes) benefit from GPU acceleration.
- **Symmetric Relations:** Relations such as `protein_protein` are modeled as bidirectional directed edges.

---

## Team Contributions

| Member | Domain & Role | Primary Deliverables |
| :--- | :--- | :--- |
| **Member 1** | **Biomedical Data Engineer** | PrimeKG ingestion (`download.py`, `preprocessing.py`), graph builders (`graph_builder.py`, `loader.py`), data validation tests. |
| **Member 2** | **GNN Architecture Engineer** | 2-layer R-GCN encoder with basis decomposition (`rgcn.py`), DistMult bilinear decoder (`decoder.py`), embeddings layer (`embeddings.py`). |
| **Member 3** | **Training & Evaluation Engineer** | Leakage-free edge splitting (`splitter.py`), typed negative sampling (`negative_sampling.py`), training loop (`train.py`), baseline ablation (`baseline.py`), inference CLI (`predict.py`). |
| **Member 4** | **Frontend & Streamlit Engineer** | Interactive multi-page Streamlit application (`app/app.py`, `pages/`), metric card components (`metric_cards.py`), Plotly & PyVis visualizers (`graph_visualization.py`). |
| **Member 5** | **Senior Integration & QA Engineer** | End-to-end integration, API contract verification, dependency resolution, 51-test suite execution, documentation and final delivery. |

---

*MediGraph AI — Precision Knowledge Graphs & Graph Representation Learning.*
