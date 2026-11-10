"""
src/data/preprocessing.py
--------------------------
PrimeKG raw data preprocessing pipeline for MediGraph AI.

Responsibilities:
  - Load raw kg.csv with actual PrimeKG column schema
  - Inspect and report actual column values
  - Clean missing / invalid / duplicate records
  - Normalize entity types to our internal schema
  - Preserve original relationship direction and names
  - Build a configurable, manageable subset
  - Produce nodes.csv and edges.csv conforming to API_CONTRACT.md §2

PrimeKG raw kg.csv schema (12 columns):
  relation, display_relation,
  x_index, x_id, x_type, x_name, x_source,
  y_index, y_id, y_type, y_name, y_source

PrimeKG x_type / y_type raw values → our internal entity_type mapping:
  "disease"               → "disease"
  "drug"                  → "drug"
  "gene/protein"          → "gene"      (genes and proteins merged in PrimeKG)
  "effect/phenotype"      → "phenotype"
  "pathway"               → "pathway"
  "biological_process"    → "biological_process"
  "molecular_function"    → "molecular_function"
  "cellular_component"    → "cellular_component"
  "anatomy"               → "anatomy"
  "exposure"              → "exposure"

Usage:
    python src/data/preprocessing.py [--config config.yaml] [--subset] [--max-edges N]

Author: Member 1 — Data Engineering
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# PrimeKG schema constants — based on actual published dataset
# ---------------------------------------------------------------------------

# All 12 columns in the raw kg.csv
PRIMEKG_REQUIRED_COLUMNS = [
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
]

# Map PrimeKG raw x_type / y_type strings → our internal entity_type strings
# Reference: Chandak et al. (2023) Table 1 and supplementary
PRIMEKG_TYPE_MAP: dict[str, str] = {
    "disease":              "disease",
    "drug":                 "drug",
    "gene/protein":         "gene",          # PrimeKG unifies genes and proteins
    "effect/phenotype":     "phenotype",
    "pathway":              "pathway",
    "biological_process":   "biological_process",
    "molecular_function":   "molecular_function",
    "cellular_component":   "cellular_component",
    "anatomy":              "anatomy",
    "exposure":             "exposure",
}

# Entity types we care about for biomedical focus
# Keep all 10 by default; configurable via subset filtering
FOCUS_ENTITY_TYPES = set(PRIMEKG_TYPE_MAP.keys())   # all PrimeKG types

# Node type column names in our output schema
OUTPUT_NODE_COLUMNS   = ["node_id", "entity_name", "entity_type", "source_id"]
OUTPUT_EDGE_COLUMNS   = ["source", "relation", "target"]


# ---------------------------------------------------------------------------
# Step 1: Load raw PrimeKG
# ---------------------------------------------------------------------------

def load_raw_primekg(raw_path: str | Path) -> pd.DataFrame:
    """
    Load the raw PrimeKG kg.csv file and validate its schema.

    Parameters
    ----------
    raw_path : str or Path
        Absolute or relative path to kg.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with 12 columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required columns are missing.
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"[MediGraph] PrimeKG kg.csv not found at '{path}'. "
            f"Run `python src/data/download.py` for download instructions. "
            f"See also: docs/dataset.md"
        )

    log.info("[MediGraph] Loading raw PrimeKG from '%s'...", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("[MediGraph] Raw file loaded: %d rows × %d columns.", len(df), len(df.columns))

    # Validate required columns
    actual_cols = set(df.columns)
    missing_cols = set(PRIMEKG_REQUIRED_COLUMNS) - actual_cols
    if missing_cols:
        raise KeyError(
            f"[MediGraph] kg.csv is missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {sorted(actual_cols)}. "
            f"Ensure you are using the official PrimeKG kg.csv from Harvard Dataverse."
        )

    return df


# ---------------------------------------------------------------------------
# Step 2: Inspect and report schema
# ---------------------------------------------------------------------------

def inspect_raw_schema(df: pd.DataFrame) -> dict:
    """
    Inspect the raw PrimeKG dataframe and return a schema summary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PrimeKG dataframe.

    Returns
    -------
    dict
        Schema summary with entity types, relation types, counts, etc.
    """
    log.info("[MediGraph] === Raw PrimeKG Schema Inspection ===")

    x_types = df["x_type"].value_counts()
    y_types = df["y_type"].value_counts()
    relations = df["relation"].value_counts()
    display_relations = df["display_relation"].value_counts()

    all_types = pd.concat([x_types, y_types]).groupby(level=0).sum().sort_values(ascending=False)

    log.info("[MediGraph] Total rows (edges): %d", len(df))
    log.info("[MediGraph] Unique x_type values: %s", sorted(df["x_type"].dropna().unique().tolist()))
    log.info("[MediGraph] Unique y_type values: %s", sorted(df["y_type"].dropna().unique().tolist()))
    log.info("[MediGraph] Unique relation values (%d total):\n%s", len(relations), relations.to_string())
    log.info("[MediGraph] Unique display_relation values (%d total):\n%s",
             len(display_relations), display_relations.to_string())
    log.info("[MediGraph] Entity type occurrence counts (x+y combined):\n%s", all_types.to_string())

    # Check for unexpected types
    all_raw_types = set(df["x_type"].dropna().unique()) | set(df["y_type"].dropna().unique())
    unknown_types = all_raw_types - set(PRIMEKG_TYPE_MAP.keys())
    if unknown_types:
        log.warning(
            "[MediGraph] Found unknown entity types not in our mapping: %s. "
            "These rows will be dropped.",
            unknown_types,
        )

    summary = {
        "total_raw_edges":          len(df),
        "x_type_counts":            x_types.to_dict(),
        "y_type_counts":            y_types.to_dict(),
        "relation_counts":          relations.to_dict(),
        "display_relation_counts":  display_relations.to_dict(),
        "all_entity_type_counts":   all_types.to_dict(),
        "unknown_types":            list(unknown_types),
    }
    return summary


# ---------------------------------------------------------------------------
# Step 3: Clean the raw dataframe
# ---------------------------------------------------------------------------

def clean_raw_primekg(
    df: pd.DataFrame,
    keep_entity_types: Optional[set[str]] = None,
    max_edges: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Clean and filter the raw PrimeKG dataframe.

    Operations performed (in order):
      1. Drop rows with NaN in any of the critical columns
      2. Strip whitespace from string columns
      3. Drop rows where x_type or y_type are not in our type map
      4. Remove self-loops (x_name == y_name AND x_type == y_type)
      5. Drop duplicate (x_name, relation, y_name) triples
      6. Optionally filter to a subset of entity types
      7. Optionally cap the number of edges (random sample, reproducible)

    Parameters
    ----------
    df : pd.DataFrame
        Raw PrimeKG dataframe.
    keep_entity_types : set of str, optional
        Set of PrimeKG raw x_type/y_type values to retain.
        If None, all known types are kept.
    max_edges : int, optional
        If provided, randomly sample this many edges (after all filtering).
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    initial_count = len(df)
    log.info("[MediGraph] Starting cleaning. Initial edges: %d", initial_count)

    # 1. Drop rows with NaN in critical columns
    critical_cols = ["x_name", "x_type", "x_id", "y_name", "y_type", "y_id", "relation"]
    df = df.dropna(subset=critical_cols).copy()
    log.info("[MediGraph] After dropping NaN in critical cols: %d rows (removed %d)",
             len(df), initial_count - len(df))

    # 2. Strip whitespace from string columns
    strip_cols = [
        "x_name", "x_type", "x_id", "x_source",
        "y_name", "y_type", "y_id", "y_source",
        "relation", "display_relation",
    ]
    df = df.copy()
    for col in strip_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()


    # 3. Drop rows where x_type or y_type are not in our mapping
    known_types = set(PRIMEKG_TYPE_MAP.keys())
    before = len(df)
    df = df[df["x_type"].isin(known_types) & df["y_type"].isin(known_types)].copy()
    log.info("[MediGraph] After dropping unknown entity types: %d rows (removed %d)",
             len(df), before - len(df))

    # 4. Remove self-loops (same entity on both sides)
    before = len(df)
    self_loop_mask = (df["x_name"] == df["y_name"]) & (df["x_type"] == df["y_type"])
    df = df[~self_loop_mask].copy()
    log.info("[MediGraph] After removing self-loops: %d rows (removed %d)",
             len(df), before - len(df))

    # 5. Drop duplicate (x_name, relation, y_name) triples
    before = len(df)
    df = df.drop_duplicates(subset=["x_name", "relation", "y_name"]).copy()
    log.info("[MediGraph] After deduplication: %d rows (removed %d)",
             len(df), before - len(df))

    # 6. Filter to specified entity types
    if keep_entity_types is not None:
        before = len(df)
        df = df[
            df["x_type"].isin(keep_entity_types) &
            df["y_type"].isin(keep_entity_types)
        ].copy()
        log.info("[MediGraph] After entity type filtering (%s): %d rows (removed %d)",
                 sorted(keep_entity_types), len(df), before - len(df))

    # 7. Cap edges via reproducible random sample
    if max_edges is not None and len(df) > max_edges:
        log.info(
            "[MediGraph] Sampling %d edges from %d (seed=%d) for manageability.",
            max_edges, len(df), seed,
        )
        df = df.sample(n=max_edges, random_state=seed).copy()
        log.info("[MediGraph] After sampling: %d rows", len(df))

    log.info(
        "[MediGraph] Cleaning complete. Final edges: %d (%.1f%% of original).",
        len(df),
        100 * len(df) / initial_count if initial_count > 0 else 0,
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Build nodes dataframe
# ---------------------------------------------------------------------------

def build_nodes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Extract unique nodes from the cleaned PrimeKG edge dataframe.

    Each unique (x_name, x_type, x_id) or (y_name, y_type, y_id) combination
    becomes one node. If the same entity name appears with different types or IDs,
    those are treated as separate nodes (edge-case that rarely occurs in PrimeKG).

    For the common case where x_name uniquely identifies an entity across the graph,
    we assign contiguous node_ids sorted by entity_name for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PrimeKG dataframe.

    Returns
    -------
    node_df : pd.DataFrame
        Columns: [node_id, entity_name, entity_type, source_id]
    node_mapping : dict
        {entity_name: node_id}
    """
    log.info("[MediGraph] Building nodes dataframe...")

    # Extract source-side entities
    x_entities = df[["x_name", "x_type", "x_id"]].rename(columns={
        "x_name": "entity_name",
        "x_type": "entity_type",
        "x_id":   "source_id",
    })

    # Extract target-side entities
    y_entities = df[["y_name", "y_type", "y_id"]].rename(columns={
        "y_name": "entity_name",
        "y_type": "entity_type",
        "y_id":   "source_id",
    })

    # Combine and deduplicate
    all_entities = pd.concat([x_entities, y_entities], ignore_index=True)

    # Map PrimeKG raw types to our internal types
    all_entities = all_entities.copy()
    all_entities["entity_type"] = all_entities["entity_type"].map(PRIMEKG_TYPE_MAP)

    # Cast source_id to string (IDs from different DBs may be int or str)
    all_entities["source_id"] = all_entities["source_id"].astype(str).str.strip()

    # Deduplicate: keep one record per (entity_name, entity_type)
    # If the same name appears with multiple source_ids, keep the first occurrence
    all_entities = (
        all_entities
        .drop_duplicates(subset=["entity_name", "entity_type"])
        .sort_values(["entity_name", "entity_type"])  # deterministic order
        .reset_index(drop=True)
    )

    # Assign contiguous node_ids (0-indexed, sorted by entity_name)
    all_entities.insert(0, "node_id", range(len(all_entities)))

    # Validate no NaN crept in
    nan_rows = all_entities[all_entities.isnull().any(axis=1)]
    if not nan_rows.empty:
        log.error(
            "[MediGraph] %d node rows still contain NaN after build. "
            "These will be dropped.",
            len(nan_rows),
        )
        all_entities = all_entities.dropna().reset_index(drop=True)
        all_entities["node_id"] = range(len(all_entities))

    # Build mapping dict: entity_name → node_id
    # NOTE: entity_name may duplicate across types (e.g., "BRCA1" as gene vs protein)
    # In PrimeKG gene/protein is a single type, so each name should be unique.
    # We use entity_name as the lookup key (required by API contract §3.1).
    # In the rare case of duplicates by name, we keep the lowest node_id.
    node_mapping: dict[str, int] = (
        all_entities
        .drop_duplicates(subset=["entity_name"], keep="first")
        .set_index("entity_name")["node_id"]
        .to_dict()
    )

    node_df = all_entities[OUTPUT_NODE_COLUMNS].copy()

    log.info(
        "[MediGraph] Node build complete: %d unique nodes, %d unique entity types.",
        len(node_df),
        node_df["entity_type"].nunique(),
    )
    type_counts = node_df["entity_type"].value_counts()
    log.info("[MediGraph] Node type distribution:\n%s", type_counts.to_string())

    return node_df, node_mapping


# ---------------------------------------------------------------------------
# Step 5: Build edges dataframe
# ---------------------------------------------------------------------------

def build_edges(
    df: pd.DataFrame,
    node_mapping: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Build the edges dataframe from cleaned PrimeKG, mapping entity names to node IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PrimeKG dataframe.
    node_mapping : dict
        {entity_name: node_id} from build_nodes().

    Returns
    -------
    edge_df : pd.DataFrame
        Columns: [source, relation, target]
    relation_mapping : dict
        {relation_string: relation_id}
    """
    log.info("[MediGraph] Building edges dataframe...")

    edge_df = pd.DataFrame({
        "source": df["x_name"].map(node_mapping),
        "relation": df["relation"].values,
        "target": df["y_name"].map(node_mapping),
    })

    # Drop edges where either endpoint is unmapped (should be rare after cleaning)
    before = len(edge_df)
    edge_df = edge_df.dropna(subset=["source", "target"]).copy()
    dropped = before - len(edge_df)
    if dropped > 0:
        log.warning(
            "[MediGraph] Dropped %d edges where source/target node not in mapping.",
            dropped,
        )

    # Cast to int (dropna converts to float)
    edge_df = edge_df.copy()
    edge_df["source"] = edge_df["source"].astype(int)
    edge_df["target"] = edge_df["target"].astype(int)


    # Build relation mapping (relation_string → int ID)
    unique_relations = sorted(edge_df["relation"].unique().tolist())
    relation_mapping: dict[str, int] = {rel: idx for idx, rel in enumerate(unique_relations)}

    edge_df = edge_df[OUTPUT_EDGE_COLUMNS].reset_index(drop=True)

    log.info(
        "[MediGraph] Edge build complete: %d edges, %d unique relation types.",
        len(edge_df),
        len(relation_mapping),
    )
    rel_counts = edge_df["relation"].value_counts()
    log.info("[MediGraph] Relation type distribution:\n%s", rel_counts.to_string())

    return edge_df, relation_mapping


# ---------------------------------------------------------------------------
# Step 6: Validation
# ---------------------------------------------------------------------------

def validate_outputs(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_mapping: dict[str, int],
    relation_mapping: dict[str, int],
) -> None:
    """
    Validate that node_df, edge_df, and mappings meet the API contract.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    errors: list[str] = []

    # --- nodes.csv checks ---
    # Required columns
    for col in OUTPUT_NODE_COLUMNS:
        if col not in node_df.columns:
            errors.append(f"nodes.csv missing column '{col}'")

    # No NaN
    if node_df.isnull().any().any():
        null_counts = node_df.isnull().sum()
        errors.append(f"nodes.csv contains NaN:\n{null_counts[null_counts > 0]}")

    # node_id is contiguous [0, N)
    expected_ids = set(range(len(node_df)))
    actual_ids   = set(node_df["node_id"].tolist())
    if expected_ids != actual_ids:
        errors.append(
            f"node_id values are not contiguous [0, {len(node_df)}). "
            f"Missing: {expected_ids - actual_ids}, Extra: {actual_ids - expected_ids}"
        )

    # No duplicate node_ids
    if node_df["node_id"].duplicated().any():
        dupes = node_df[node_df["node_id"].duplicated(keep=False)]
        errors.append(f"Duplicate node_ids found:\n{dupes}")

    # No empty entity_name
    if (node_df["entity_name"].str.strip() == "").any():
        errors.append("nodes.csv contains empty entity_name values.")

    # --- edges.csv checks ---
    for col in OUTPUT_EDGE_COLUMNS:
        if col not in edge_df.columns:
            errors.append(f"edges.csv missing column '{col}'")

    if edge_df.isnull().any().any():
        null_counts = edge_df.isnull().sum()
        errors.append(f"edges.csv contains NaN:\n{null_counts[null_counts > 0]}")

    # Referential integrity
    valid_node_ids = set(node_df["node_id"].tolist())
    invalid_sources = set(edge_df["source"].tolist()) - valid_node_ids
    invalid_targets = set(edge_df["target"].tolist()) - valid_node_ids
    if invalid_sources:
        errors.append(
            f"edges.csv source IDs not in nodes.csv: {list(invalid_sources)[:10]}..."
        )
    if invalid_targets:
        errors.append(
            f"edges.csv target IDs not in nodes.csv: {list(invalid_targets)[:10]}..."
        )

    # --- Mapping consistency checks ---
    # node_mapping keys must match entity_names in nodes.csv exactly
    node_names_in_csv = set(node_df["entity_name"].tolist())
    mapping_keys = set(node_mapping.keys())
    missing_in_mapping = node_names_in_csv - mapping_keys
    if missing_in_mapping:
        errors.append(
            f"nodes.csv has entity_name not in node_mapping.json: "
            f"{list(missing_in_mapping)[:5]}..."
        )
    extra_in_mapping = mapping_keys - node_names_in_csv
    if extra_in_mapping:
        errors.append(
            f"node_mapping.json has keys not in nodes.csv: "
            f"{list(extra_in_mapping)[:5]}..."
        )

    # relation_mapping keys must match relations in edges.csv exactly
    edge_relations = set(edge_df["relation"].tolist())
    rel_mapping_keys = set(relation_mapping.keys())
    missing_in_rel_mapping = edge_relations - rel_mapping_keys
    if missing_in_rel_mapping:
        errors.append(
            f"edges.csv has relation not in relation_mapping.json: "
            f"{list(missing_in_rel_mapping)[:5]}..."
        )
    extra_in_rel_mapping = rel_mapping_keys - edge_relations
    if extra_in_rel_mapping:
        errors.append(
            f"relation_mapping.json has keys not in edges.csv: "
            f"{list(extra_in_rel_mapping)[:5]}..."
        )


    if errors:
        error_msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(
            f"[MediGraph] Data validation failed with {len(errors)} error(s):\n{error_msg}"
        )

    log.info(
        "[MediGraph] ✓ Validation passed: %d nodes, %d edges, %d relations, %d entity types.",
        len(node_df),
        len(edge_df),
        len(relation_mapping),
        node_df["entity_type"].nunique(),
    )


# ---------------------------------------------------------------------------
# Step 7: Subset builder (for 5-day prototype manageability)
# ---------------------------------------------------------------------------

def build_focused_subset(
    df: pd.DataFrame,
    cfg: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a focused subset of PrimeKG for the 5-day prototype.

    Strategy:
      - Keep all entity types by default (configurable via config.yaml graph.max_edges)
      - If graph.max_edges is set, sample that many edges
      - Ensure all 10 PrimeKG entity types are represented in the subset
        by guaranteeing at least some edges from each relation type category

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PrimeKG dataframe (after clean_raw_primekg).
    cfg : dict
        Config dict.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Subset dataframe (same schema as input).
    """
    max_edges = cfg["graph"].get("max_edges")
    max_nodes = cfg["graph"].get("max_nodes")

    if max_edges is None and max_nodes is None:
        log.info("[MediGraph] No subset limit configured. Using all %d cleaned edges.", len(df))
        return df

    if max_edges is not None:
        log.info(
            "[MediGraph] Applying max_edges=%d limit from config (from %d edges).",
            max_edges, len(df),
        )
        # Stratified sampling: sample proportionally from each relation type
        # so no relation type is completely dropped
        relation_groups = df.groupby("relation")
        num_relations = df["relation"].nunique()
        min_per_relation = max(1, max_edges // (num_relations * 10))

        sampled_parts = []
        remaining_budget = max_edges

        # First pass: guarantee minimum per relation
        for rel, group in relation_groups:
            n = min(min_per_relation, len(group))
            sampled_parts.append(group.sample(n=n, random_state=seed))
            remaining_budget -= n

        guaranteed_df = pd.concat(sampled_parts, ignore_index=True)
        guaranteed_indices = set(guaranteed_df.index)

        # Second pass: fill remaining budget from leftover edges
        leftover = df[~df.index.isin(guaranteed_indices)]
        if remaining_budget > 0 and len(leftover) > 0:
            extra = leftover.sample(
                n=min(remaining_budget, len(leftover)),
                random_state=seed,
            )
            df = pd.concat([guaranteed_df, extra], ignore_index=True)
        else:
            df = guaranteed_df

        log.info("[MediGraph] Subset built: %d edges from %d relation types.",
                 len(df), df["relation"].nunique())

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 8: Save sample data
# ---------------------------------------------------------------------------

def save_sample_data(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    cfg: dict,
    max_nodes: int = 2000,
    max_edges: int = 10000,
    seed: int = 42,
) -> None:
    """
    Save a small deterministic sample to data/sample/ for CI and demos.

    Parameters
    ----------
    node_df, edge_df : pd.DataFrame
    cfg : dict
    max_nodes, max_edges : int
        Maximum sample sizes (defaults from API contract §1).
    seed : int
    """
    sample_dir = Path(cfg["paths"]["data_sample"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Sample edges first, then derive nodes from those edges
    sampled_edges = edge_df.sample(
        n=min(max_edges, len(edge_df)),
        random_state=seed,
    ).reset_index(drop=True)

    # Keep only nodes referenced in sampled edges
    involved_nodes = set(sampled_edges["source"].tolist()) | set(sampled_edges["target"].tolist())
    sampled_nodes = node_df[node_df["node_id"].isin(involved_nodes)].copy()

    # Cap nodes if still too large
    if len(sampled_nodes) > max_nodes:
        kept_node_ids = set(
            sampled_nodes.sample(n=max_nodes, random_state=seed)["node_id"].tolist()
        )
        sampled_edges = sampled_edges[
            sampled_edges["source"].isin(kept_node_ids) &
            sampled_edges["target"].isin(kept_node_ids)
        ].reset_index(drop=True)
        sampled_nodes = node_df[node_df["node_id"].isin(kept_node_ids)].copy()

    nodes_path = sample_dir / "nodes_sample.csv"
    edges_path = sample_dir / "edges_sample.csv"
    sampled_nodes.to_csv(nodes_path, index=False)
    sampled_edges.to_csv(edges_path, index=False)

    log.info(
        "[MediGraph] Sample data saved: %d nodes → '%s', %d edges → '%s'.",
        len(sampled_nodes), nodes_path,
        len(sampled_edges), edges_path,
    )


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def preprocess(
    cfg: dict,
    keep_entity_types: Optional[set[str]] = None,
    max_edges: Optional[int] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Full preprocessing pipeline: raw PrimeKG → processed CSVs + mappings.

    Parameters
    ----------
    cfg : dict
        Loaded configuration dict (from load_config()).
    keep_entity_types : set, optional
        PrimeKG raw type strings to retain (e.g. {"disease", "drug"}).
        Default: all 10 PrimeKG types.
    max_edges : int, optional
        Override for the number of edges (overrides cfg["graph"]["max_edges"]).
    seed : int
        Random seed (overrides cfg["project"]["seed"]).

    Returns
    -------
    node_df : pd.DataFrame
    edge_df : pd.DataFrame
    node_mapping : dict
    relation_mapping : dict
    """
    set_seed(seed)

    # Resolve paths
    raw_path         = Path(cfg["paths"]["data_raw"]) / "kg.csv"
    nodes_out_path   = Path(cfg["paths"]["nodes_csv"])
    edges_out_path   = Path(cfg["paths"]["edges_csv"])
    node_map_path    = Path(cfg["paths"]["node_mapping_json"])
    rel_map_path     = Path(cfg["paths"]["relation_mapping_json"])

    # Create output directories
    nodes_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine effective max_edges
    effective_max_edges = max_edges if max_edges is not None else cfg["graph"].get("max_edges")

    # --- STEP 1: Load ---
    df_raw = load_raw_primekg(raw_path)

    # --- STEP 2: Inspect ---
    schema_summary = inspect_raw_schema(df_raw)

    # --- STEP 3: Clean ---
    df_clean = clean_raw_primekg(
        df_raw,
        keep_entity_types=keep_entity_types,
        max_edges=effective_max_edges,
        seed=seed,
    )

    if len(df_clean) == 0:
        raise ValueError(
            "[MediGraph] After cleaning, the dataframe is empty. "
            "Check entity type filters and ensure kg.csv is valid."
        )

    # --- STEP 4: Build nodes ---
    node_df, node_mapping = build_nodes(df_clean)

    # --- STEP 5: Build edges ---
    # Re-derive edges aligned with the cleaned df
    # We need to rebuild the edge_df from df_clean since clean_raw_primekg
    # may have resampled / reordered rows.
    edge_df, relation_mapping = build_edges(df_clean, node_mapping)

    # --- STEP 6: Validate ---
    validate_outputs(node_df, edge_df, node_mapping, relation_mapping)

    # --- STEP 7: Save ---
    import json

    node_df.to_csv(nodes_out_path, index=False)
    log.info("[MediGraph] nodes.csv saved → '%s'", nodes_out_path)

    edge_df.to_csv(edges_out_path, index=False)
    log.info("[MediGraph] edges.csv saved → '%s'", edges_out_path)

    with open(node_map_path, "w", encoding="utf-8") as fh:
        json.dump(node_mapping, fh, indent=2, ensure_ascii=False)
    log.info("[MediGraph] node_mapping.json saved → '%s'", node_map_path)

    with open(rel_map_path, "w", encoding="utf-8") as fh:
        json.dump(relation_mapping, fh, indent=2, ensure_ascii=False)
    log.info("[MediGraph] relation_mapping.json saved → '%s'", rel_map_path)

    # --- STEP 8: Save sample ---
    save_sample_data(node_df, edge_df, cfg, seed=seed)

    # --- Final summary ---
    log.info("[MediGraph] === Preprocessing Complete ===")
    log.info("[MediGraph]   Nodes:          %d", len(node_df))
    log.info("[MediGraph]   Edges:          %d", len(edge_df))
    log.info("[MediGraph]   Entity types:   %d", node_df["entity_type"].nunique())
    log.info("[MediGraph]   Relation types: %d", len(relation_mapping))
    log.info("[MediGraph]   Node types:     %s",
             node_df["entity_type"].value_counts().to_dict())
    log.info("[MediGraph]   Relations:      %s", list(relation_mapping.keys()))

    return node_df, edge_df, node_mapping, relation_mapping


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess PrimeKG raw data into MediGraph AI format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preprocessing (uses config.yaml defaults)
  python src/data/preprocessing.py

  # Subset: only disease-drug-gene relations, cap at 500K edges
  python src/data/preprocessing.py --types disease drug gene/protein --max-edges 500000

  # Inspect raw schema only (no output written)
  python src/data/preprocessing.py --inspect-only
        """,
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Path to config.yaml (default: project root config.yaml)")
    p.add_argument(
        "--types", nargs="+", default=None,
        metavar="TYPE",
        help=(
            "PrimeKG entity types to keep (raw names, space-separated). "
            "Valid: disease drug 'gene/protein' 'effect/phenotype' pathway "
            "biological_process molecular_function cellular_component anatomy exposure. "
            "Default: all types."
        ),
    )
    p.add_argument("--max-edges", type=int, default=None,
                   help="Maximum number of edges to retain (random sample).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: from config.yaml project.seed).")
    p.add_argument("--inspect-only", action="store_true",
                   help="Only inspect raw schema; do not write output files.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]

    keep_types = set(args.types) if args.types else None
    if keep_types:
        # Validate provided type names
        valid_types = set(PRIMEKG_TYPE_MAP.keys())
        invalid = keep_types - valid_types
        if invalid:
            print(f"[MediGraph] Unknown entity types: {invalid}")
            print(f"Valid types: {sorted(valid_types)}")
            sys.exit(1)

    raw_path = Path(cfg["paths"]["data_raw"]) / "kg.csv"
    if not raw_path.exists():
        print(
            f"\n[MediGraph] PrimeKG not found at '{raw_path}'.\n"
            f"Run: python src/data/download.py\n"
        )
        sys.exit(1)

    if args.inspect_only:
        df_raw = load_raw_primekg(raw_path)
        inspect_raw_schema(df_raw)
        sys.exit(0)

    preprocess(cfg, keep_entity_types=keep_types, max_edges=args.max_edges, seed=seed)


if __name__ == "__main__":
    main()

