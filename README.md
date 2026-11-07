# MediGraph AI

**Intelligent Medical Knowledge Graph and Graph Neural Network Platform**

> ⚠️ **Research & Educational Prototype Only.**  
> This system is NOT a clinical diagnosis or treatment recommendation tool.  
> Model outputs must NOT be interpreted as medical advice.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Team Responsibilities](#team-responsibilities)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Data Contract](#data-contract)
6. [Quick Start](#quick-start)
7. [Running the Application](#running-the-application)
8. [Configuration](#configuration)
9. [Documentation](#documentation)

---

## Project Overview

MediGraph AI constructs a biomedical knowledge graph from [PrimeKG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) and trains a Relational Graph Convolutional Network (R-GCN) to perform **link prediction** across biomedical entity pairs (diseases, drugs, genes, proteins, phenotypes, etc.).

The platform provides:
- A processed knowledge graph (nodes + typed edges)
- An R-GCN encoder with DistMult decoder
- Training, evaluation and inference pipelines
- An interactive Streamlit dashboard

---

## Team Responsibilities

| Member | Role |
|--------|------|
| Member 1 | Dataset download, preprocessing, graph construction (`src/data/`) |
| Member 2 | R-GCN model and DistMult decoder (`src/models/`) |
| Member 3 | Training loop, evaluation, inference (`src/training/`, `src/inference/`) |
| Member 4 | Streamlit frontend and visualisation (`app/`) |
| Project Lead | Repo setup, integration, final testing |

Each member works independently against the **API Contract** (`docs/API_CONTRACT.md`).

---

## Technology Stack

| Category | Library |
|----------|---------|
| Deep learning | PyTorch, PyTorch Geometric |
| Data processing | Pandas, NumPy |
| Graph analysis | NetworkX |
| ML utilities | Scikit-learn |
| Frontend | Streamlit, Plotly, PyVis |
| Config | PyYAML |
| Testing | Pytest |

---

## Project Structure

```
MediGraph-AI/
├── config.yaml                  # Central configuration (all paths & hyperparams)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/                     # Original PrimeKG files (NOT in Git)
│   ├── processed/               # nodes.csv, edges.csv, mappings (NOT in Git)
│   └── sample/                  # Small deterministic sample for CI / demos
│
├── notebooks/                   # Exploratory Jupyter notebooks
│
├── src/
│   ├── data/                    # Member 1 — data loading & graph construction
│   ├── models/                  # Member 2 — R-GCN + DistMult
│   ├── training/                # Member 3 — training loop & evaluation
│   ├── inference/               # Member 3 — inference & link prediction
│   └── utils/                   # Shared helpers (config loader, logging, etc.)
│
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── pages/                   # Multi-page Streamlit pages
│   └── components/              # Reusable UI components
│
├── artifacts/
│   ├── checkpoints/             # Saved model weights (.pt)
│   ├── predictions/             # Exported predictions (.json / .csv)
│   └── metrics/                 # Training & eval metrics (.json)
│
├── tests/                       # Unit and integration tests
└── docs/
    ├── architecture.md
    ├── dataset.md
    └── API_CONTRACT.md          # ← READ THIS FIRST
```

---

## Data Contract

All modules exchange data through two standardised CSV files plus two JSON mappings.

| File | Description |
|------|-------------|
| `data/processed/nodes.csv` | Nodes with `node_id`, `entity_name`, `entity_type`, `source_id` |
| `data/processed/edges.csv` | Edges with `source`, `relation`, `target` (integer node IDs) |
| `data/processed/node_mapping.json` | `{entity_name: node_id}` |
| `data/processed/relation_mapping.json` | `{relation_string: relation_id}` |

See [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) for full schemas and tensor formats.

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd MediGraph-AI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download PrimeKG (Member 1 task)

```bash
# See docs/dataset.md for download instructions
# Place files in data/raw/
```

### 3. Process data (Member 1 task)

```bash
python src/data/preprocess.py
```

### 4. Train the model (Member 3 task)

```bash
python src/training/train.py
```

### 5. Run inference (Member 3 task)

```bash
python src/inference/predict.py --query_node "Alzheimer disease" --relation "associated_with"
```

### 6. Launch the dashboard (Member 4 task)

```bash
streamlit run app/main.py
```

---

## Configuration

All hyperparameters, paths and flags live in [`config.yaml`](config.yaml).  
Never hardcode paths in source files — always read from the config.

```python
from src.utils.config import load_config
cfg = load_config()
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) | Full inter-module contract |
| [`docs/architecture.md`](docs/architecture.md) | System design and data-flow |
| [`docs/dataset.md`](docs/dataset.md) | PrimeKG description and download guide |

