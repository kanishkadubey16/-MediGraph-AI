# MediGraph AI — Dataset Guide

> **Version:** 0.1.0  
> **Responsible:** Member 1

---

## Dataset: PrimeKG

**Full name:** Precision Medicine Knowledge Graph  
**Paper:** [Chandak et al., Nature Scientific Data, 2023](https://www.nature.com/articles/s41597-023-01960-3)  
**Download:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)

PrimeKG integrates 20 high-quality biomedical databases into a single knowledge graph with:
- ~129,000 nodes
- ~8,100,000 edges
- 10 node types
- 30+ relation types

---

## Downloading PrimeKG

```bash
# Option 1: Direct download from Harvard Dataverse
# Visit the URL and download kg.csv (the primary edge file)
# Place it at: data/raw/kg.csv

# Option 2: Using the Harvard Dataverse API
curl -L "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/IXA7BM/PYHJMF" \
     -o data/raw/kg.csv
```

> ⚠️ The raw file is large (~500 MB). Do NOT commit it to Git.

---

## PrimeKG Column Schema (kg.csv)

The raw file contains these columns (do not rename them in preprocessing):

| Column | Description |
|--------|-------------|
| `relation` | Relation type string |
| `display_relation` | Human-readable relation label |
| `x_id` | Source entity external ID |
| `x_type` | Source entity type |
| `x_name` | Source entity name |
| `x_source` | Source database |
| `y_id` | Target entity external ID |
| `y_type` | Target entity type |
| `y_name` | Target entity name |
| `y_source` | Target database |

---

## Preprocessing Steps (Member 1)

Implemented in `src/data/preprocess.py`:

1. Load `data/raw/kg.csv`.
2. Extract unique nodes from both `x_name`/`x_type`/`x_id` and `y_name`/`y_type`/`y_id` columns.
3. Assign contiguous integer `node_id` values (0-indexed).
4. Save `data/processed/nodes.csv` per the schema in `docs/API_CONTRACT.md §2.1`.
5. Map edge `x_name` → `source` integer, `y_name` → `target` integer.
6. Save `data/processed/edges.csv` per the schema in `docs/API_CONTRACT.md §2.2`.
7. Build and save `node_mapping.json` and `relation_mapping.json`.
8. Validate: check no NaN, check node_id contiguity, check edge referential integrity.

---

## Sample Data

For development without the full dataset, a small deterministic sample is provided:

```
data/sample/nodes_sample.csv   — ≤ 2000 nodes
data/sample/edges_sample.csv  — ≤ 10000 edges
```

The sample must follow the same schema as the full processed files.

---

## Licence

PrimeKG is distributed under CC BY 4.0.  
Cite: Chandak et al. (2023) when publishing results derived from this dataset.

