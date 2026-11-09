# MediGraph AI — Dataset Guide & Data Engineering Documentation

> **Version:** 0.1.0  
> **Author:** Member 1 — Biomedical Data Engineer  
> **Last Updated:** 2026-11-09

---

## 1. Dataset Source

* **Dataset Name:** PrimeKG (Precision Medicine Knowledge Graph)
* **Hosting Repository:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)
* **Direct File Access ID:** File 6180620 (`kg.csv`)
* **Persistent DOI:** `doi:10.7910/DVN/IXA7BM`
* **Direct Download URL:** `https://dataverse.harvard.edu/api/access/datafile/6180620`
* **Python TDC Resource:** `from tdc.resource import PrimeKG; PrimeKG(path='data/raw')`

---

## 2. Dataset Citation

When publishing, reporting, or citing experiments derived from PrimeKG, cite the primary publication:

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

**APA Citation:**  
Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10(1), 67. https://doi.org/10.1038/s41597-023-01960-3

---

## 3. License and Usage Restrictions

* **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
* **Usage Restrictions:** Free to share and adapt for research, educational, and commercial purposes with appropriate credit given to the original authors.
* **Clinical Disclaimer:** MediGraph AI and PrimeKG are research and educational prototypes only. They are **NOT** medical devices, clinical diagnostic tools, or therapeutic recommendation systems. Model predictions should never be used as medical advice.

---

## 4. Original PrimeKG Schema (`kg.csv`)

The raw PrimeKG file contains **12 columns** spanning 8,100,498 directed rows (representing ~4,050,249 unique biomedical relationship pairs):

| Column Name | Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `relation` | `str` | Standardized broad relation type | `indication`, `disease_protein`, `protein_protein` |
| `display_relation` | `str` | Granular relation label from source database | `indication`, `associated with`, `target`, `ppi` |
| `x_index` | `int` | Internal integer index for source entity | `0`, `4102`, `12045` |
| `x_id` | `str` | External database ID for source entity | `MONDO:0004975`, `DB00998`, `348` |
| `x_type` | `str` | Biological category of source entity | `disease`, `drug`, `gene/protein`, `effect/phenotype` |
| `x_name` | `str` | Canonical display name of source entity | `Alzheimer disease`, `Albuterol`, `APOE` |
| `x_source` | `str` | Originating database of source entity | `MONDO`, `DrugBank`, `NCBI`, `HPO` |
| `y_index` | `int` | Internal integer index for target entity | `1`, `8889`, `18321` |
| `y_id` | `str` | External database ID for target entity | `MONDO:0004979`, `56992`, `HP:0001962` |
| `y_type` | `str` | Biological category of target entity | `disease`, `gene/protein`, `effect/phenotype` |
| `y_name` | `str` | Canonical display name of target entity | `asthma`, `PHYHIPL`, `Palpitations` |
| `y_source` | `str` | Originating database of target entity | `MONDO`, `NCBI`, `HPO` |

### Primary Biological Scales (Entity Types) in PrimeKG
1. `disease` (from Mondo, Orphanet, OMIM)
2. `drug` (from DrugBank)
3. `gene/protein` (unified genomic/proteomic entities from NCBI Entrez / STRING)
4. `effect/phenotype` (clinical signs and symptoms from Human Phenotype Ontology)
5. `pathway` (signaling/metabolic pathways from Reactome and WikiPathways)
6. `biological_process` (Gene Ontology biological process)
7. `molecular_function` (Gene Ontology molecular function)
8. `cellular_component` (Gene Ontology cellular component)
9. `anatomy` (anatomical localizations from UBERON)
10. `exposure` (environmental and toxicological factors from CTD)

---

## 5. Preprocessing Steps

Implemented in `src/data/preprocessing.py`:

```
┌────────────────────────────────────────────────────────┐
│               data/raw/kg.csv (PrimeKG)                │
└──────────────────────────┬─────────────────────────────┘
                           │ 1. Load & validate 12 required columns
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Data Cleaning                        │
│ - Drop rows with NaN in critical columns               │
│ - Strip whitespace across string fields                │
│ - Filter unknown entity types                          │
│ - Remove self-loops (x_name == y_name & x_type==y_type)│
│ - Deduplicate (x_name, relation, y_name) triples       │
│ - Configurable filtering (--types, --max-edges)        │
└──────────────────────────┬─────────────────────────────┘
                           │ 2. Extract & assign canonical IDs
                           ▼
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌──────────────────────┐          ┌──────────────────────┐
│  Node Construction   │          │  Edge Construction   │
│ - Unique (name, type)│          │ - Map names → IDs    │
│ - Contiguous node_ids│          │ - Integer source/tgt │
│ - Type normalization │          │ - Preserve direction │
└──────────┬───────────┘          └──────────┬───────────┘
           │                                 │
           ▼                                 ▼
┌────────────────────────────────────────────────────────┐
│                  Validation Engine                     │
│ - Check no NaNs                                        │
│ - Verify node_id contiguity [0, N)                     │
│ - Check referential integrity (sources/targets valid)  │
│ - Check node_mapping and relation_mapping consistency  │
└──────────────────────────┬─────────────────────────────┘
                           │ 3. Save standardized outputs
                           ▼
┌────────────────────────────────────────────────────────┐
│ - data/processed/nodes.csv                             │
│ - data/processed/edges.csv                             │
│ - data/processed/node_mapping.json                     │
│ - data/processed/relation_mapping.json                 │
│ - data/sample/nodes_sample.csv                         │
│ - data/sample/edges_sample.csv                         │
└────────────────────────────────────────────────────────┘
```

### Entity Type Normalization
PrimeKG merges genes and proteins under `"gene/protein"`. In our standardized output schema, these are normalized to `"gene"` to conform to the API contract while maintaining source external identifiers.

---

## 6. Filtering Decisions (5-Day Prototype Scope)

Full PrimeKG is ~8.1 million edges (~500 MB). For rapid development and memory efficiency during local model iteration, our pipeline provides configurable filtering:

1. **Entity Type Selection (`--types`):** Select specific biological scales of interest (e.g., `disease drug gene/protein effect/phenotype`).
2. **Stratified Edge Budget (`--max-edges`):** When subsampling, edges are sampled proportionally across relation types to prevent rare but critical biomedical relationships (e.g. drug indications) from vanishing.
3. **Reproducibility Seed (`--seed`):** All sampling and sorting operations are deterministic based on `cfg["project"]["seed"]` (default: 42).

---

## 7. Graph Statistics

### A. Development Sample Graph Benchmark (`artifacts/metrics/graph_stats.json`)
Generated from authentic PrimeKG records across disease, drug, gene/protein, phenotype, pathway, and anatomy categories:

| Metric | Value |
| :--- | :--- |
| **Total Nodes** | 39 |
| **Total Directed Edges** | 36 |
| **Entity Types** | 6 |
| **Relation Types** | 7 |
| **Graph Density** | 0.024291 |
| **Isolated Nodes** | 0 (0.0%) |
| **Mean Degree** | 1.85 (min: 1.0, max: 5.0, median: 1.0) |

#### Entity Type Distribution
* `gene`: 14
* `drug`: 7
* `disease`: 6
* `phenotype`: 6
* `pathway`: 3
* `anatomy`: 3

#### Relation Type Distribution
* `disease_protein`: 7 (19.4%)
* `indication`: 6 (16.7%)
* `disease_phenotype_positive`: 6 (16.7%)
* `protein_protein`: 6 (16.7%)
* `drug_protein`: 5 (13.9%)
* `protein_pathway`: 3 (8.3%)
* `anatomy_protein_present`: 3 (8.3%)

#### Top Hub Entities by Total Degree
1. **Alzheimer disease** (disease): 5 connections (out: 4, in: 1)
2. **APP** (gene): 4 connections (out: 1, in: 3)
3. **ESR1** (gene): 4 connections (out: 1, in: 3)
4. **breast cancer** (disease): 4 connections (out: 3, in: 1)
5. **APOE** (gene): 3 connections (out: 2, in: 1)
6. **BRCA1** (gene): 3 connections (out: 2, in: 1)
7. **Parkinson disease** (disease): 3 connections (out: 2, in: 1)
8. **SNCA** (gene): 3 connections (out: 1, in: 2)
9. **asthma** (disease): 3 connections (out: 2, in: 1)
10. **ADRB2** (gene): 2 connections (out: 0, in: 2)

---

## 8. Known Dataset Limitations

1. **Gene/Protein Unification:** PrimeKG represents genes and proteins as unified nodes rather than distinct ontological layers. While common in systems biology, this merges transcriptional regulation and physical protein binding into the same node coordinate.
2. **Relation Sparsity:** While `protein_protein` interactions dominate PrimeKG (~642,000 interactions), clinical relation categories like `contraindication` or `off-label use` are much sparser. Stratified negative sampling is required to prevent model prediction bias.
3. **Biomedical Literature Bias:** Frequently studied diseases (e.g., Alzheimer's, Breast Cancer, Type 2 Diabetes) have dramatically higher node degrees than rare orphan diseases, reflecting publication density rather than biological complexity.
4. **Directionality Interpretation:** Some relations in PrimeKG (like PPIs) are inherently symmetric/undirected, whereas others (like drug indications or gene-phenotype causality) are directed. The pipeline preserves original PrimeKG directions without arbitrary reversals.
