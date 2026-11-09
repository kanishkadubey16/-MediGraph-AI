"""
src/data/preprocess.py
----------------------
Raw-to-processed preprocessing pipeline for MediGraph AI.

Reads the PrimeKG edge file (``data/raw/kg.csv``), builds canonical node and
edge tables, writes four artefacts to ``data/processed/``, and logs summary
statistics.

Can be used as an **importable module**:

    from src.data.preprocess import preprocess
    from src.utils.config import load_config

    cfg = load_config()
    node_df, edge_df, node_mapping, rel_mapping = preprocess(cfg)

Or run directly from the project root:

    python src/data/preprocess.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger, configure_root_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# PrimeKG column names (source-of-truth: docs/dataset.md)
# ---------------------------------------------------------------------------
_PRIMEKG_REQUIRED_COLS: list[str] = [
    "relation",
    "display_relation",
    "x_id",
    "x_type",
    "x_name",
    "x_source",
    "y_id",
    "y_type",
    "y_name",
    "y_source",
]


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def preprocess(
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Full raw-to-processed preprocessing pipeline.

    Steps
    -----
    1. Validate that ``data/raw/kg.csv`` exists (raises ``FileNotFoundError``
       with instructions if not).
    2. Load kg.csv and check required PrimeKG columns are present.
    3. Build a **deduplicated nodes** DataFrame
       ``[node_id, entity_name, entity_type, source_id]`` by extracting all
       unique ``(x_name, x_type, x_id)`` and ``(y_name, y_type, y_id)``
       pairs.  Node IDs are assigned as a contiguous range ``[0, N)`` sorted
       by ``entity_name`` for reproducibility.
    4. Build an **edges** DataFrame ``[source, relation, target]`` by mapping
       ``x_name`` → integer source and ``y_name`` → integer target. Self-loops
       and rows with NaN are removed.
    5. Build ``node_mapping`` (``{entity_name: node_id}``) and
       ``relation_mapping`` (``{relation_string: relation_id}``).
    6. Persist all four artefacts to paths from *cfg*.
    7. Log summary statistics.

    Parameters
    ----------
    cfg:
        Configuration dictionary returned by
        :py:func:`src.utils.config.load_config`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]
        ``(node_df, edge_df, node_mapping, relation_mapping)``

    Raises
    ------
    FileNotFoundError
        If ``data/raw/kg.csv`` does not exist.
    KeyError
        If required columns are absent from kg.csv.
    ValueError
        If produced artefacts fail internal consistency checks.
    """
    paths = cfg["paths"]
    raw_dir = Path(paths["data_raw"])
    kg_path = raw_dir / "kg.csv"

    # ------------------------------------------------------------------ 1 --
    _assert_kg_exists(kg_path)

    # ------------------------------------------------------------------ 2 --
    log.info("Loading PrimeKG from '%s' …", kg_path)
    kg_df = pd.read_csv(
        kg_path,
        dtype={
            "x_id": str,
            "y_id": str,
            "x_type": str,
            "y_type": str,
            "x_name": str,
            "y_name": str,
            "x_source": str,
            "y_source": str,
            "relation": str,
            "display_relation": str,
        },
        low_memory=False,
    )
    log.info("Raw kg.csv loaded: %d rows, %d columns.", len(kg_df), len(kg_df.columns))

    _assert_primekg_columns(kg_df, kg_path)

    # ------------------------------------------------------------------ 3 --
    log.info("Building deduplicated node table …")
    node_df, node_mapping = _build_nodes(kg_df)
    log.info(
        "Node table built: %d unique nodes across %d entity type(s).",
        len(node_df),
        node_df["entity_type"].nunique(),
    )

    # ------------------------------------------------------------------ 4 --
    log.info("Building edge table …")
    edge_df = _build_edges(kg_df, node_mapping)
    log.info(
        "Edge table built: %d edges (%d unique relation types).",
        len(edge_df),
        edge_df["relation"].nunique(),
    )

    # ------------------------------------------------------------------ 5 --
    relation_mapping = _build_relation_mapping(edge_df)

    # ------------------------------------------------------------------ 6 --
    _save_artefacts(node_df, edge_df, node_mapping, relation_mapping, paths)

    # ------------------------------------------------------------------ 7 --
    _log_statistics(node_df, edge_df, node_mapping, relation_mapping)

    return node_df, edge_df, node_mapping, relation_mapping


# ---------------------------------------------------------------------------
# Internal builder helpers
# ---------------------------------------------------------------------------


def _assert_kg_exists(kg_path: Path) -> None:
    """Raise ``FileNotFoundError`` with helpful instructions if kg.csv is absent."""
    if not kg_path.exists():
        raise FileNotFoundError(
            f"[MediGraph] PrimeKG edge file not found at '{kg_path}'. \n\n"
            f"To obtain the dataset:\n"
            f"  • See docs/dataset.md for download instructions.\n"
            f"  • Visit: https://dataverse.harvard.edu/dataset.xhtml"
            f"?persistentId=doi:10.7910/DVN/IXA7BM\n"
            f"  • Download kg.csv and place it at: {kg_path}\n\n"
            f"The raw file is ~500 MB. Do NOT commit it to Git."
        )


def _assert_primekg_columns(kg_df: pd.DataFrame, kg_path: Path) -> None:
    """Raise ``KeyError`` if any required PrimeKG column is missing."""
    missing = [c for c in _PRIMEKG_REQUIRED_COLS if c not in kg_df.columns]
    if missing:
        raise KeyError(
            f"[MediGraph] kg.csv at '{kg_path}' is missing required "
            f"column(s): {missing}. "
            f"Expected PrimeKG columns: {_PRIMEKG_REQUIRED_COLS}."
        )


def _build_nodes(
    kg_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Extract unique nodes from PrimeKG and assign contiguous node IDs.

    Parameters
    ----------
    kg_df:
        Raw PrimeKG DataFrame (must have x_name/x_type/x_id/y_name/y_type/y_id).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        ``(node_df, node_mapping)`` where *node_df* has columns
        ``[node_id, entity_name, entity_type, source_id]`` and
        *node_mapping* maps entity_name → node_id.
    """
    # Extract source-side nodes
    source_nodes = (
        kg_df[["x_name", "x_type", "x_id"]]
        .rename(columns={"x_name": "entity_name", "x_type": "entity_type", "x_id": "source_id"})
        .copy()
    )

    # Extract target-side nodes
    target_nodes = (
        kg_df[["y_name", "y_type", "y_id"]]
        .rename(columns={"y_name": "entity_name", "y_type": "entity_type", "y_id": "source_id"})
        .copy()
    )

    # Combine and deduplicate on entity_name (entity_name is the canonical key)
    combined = (
        pd.concat([source_nodes, target_nodes], ignore_index=True)
        .dropna(subset=["entity_name", "entity_type"])
        .drop_duplicates(subset=["entity_name"])
        .sort_values("entity_name", kind="mergesort")  # stable sort for reproducibility
        .reset_index(drop=True)
    )

    # Ensure source_id is a string even if originally numeric
    combined["source_id"] = combined["source_id"].astype(str)

    # Assign contiguous node_id [0, N)
    combined.insert(0, "node_id", range(len(combined)))
    combined["node_id"] = combined["node_id"].astype("int64")

    # Reorder columns to match contract §2.1
    node_df = combined[["node_id", "entity_name", "entity_type", "source_id"]].copy()

    # Build mapping dict
    node_mapping: dict[str, int] = dict(
        zip(node_df["entity_name"].tolist(), node_df["node_id"].tolist())
    )

    return node_df, node_mapping


def _build_edges(
    kg_df: pd.DataFrame,
    node_mapping: dict[str, int],
) -> pd.DataFrame:
    """
    Build the edges DataFrame from raw PrimeKG.

    Maps ``x_name`` → integer source and ``y_name`` → integer target using
    *node_mapping*.  Removes self-loops and rows with any NaN.

    Parameters
    ----------
    kg_df:
        Raw PrimeKG DataFrame.
    node_mapping:
        Mapping ``{entity_name: node_id}`` produced by :py:func:`_build_nodes`.

    Returns
    -------
    pd.DataFrame
        Edges DataFrame with columns ``[source, relation, target]``,
        all non-NaN, no self-loops.
    """
    edge_df = kg_df[["x_name", "relation", "y_name"]].copy()

    # Map entity names to integer node IDs
    edge_df["source"] = edge_df["x_name"].map(node_mapping)
    edge_df["target"] = edge_df["y_name"].map(node_mapping)

    # Drop rows where mapping produced NaN (entity not in node table)
    n_before = len(edge_df)
    edge_df = edge_df.dropna(subset=["source", "relation", "target"])
    n_dropped_nan = n_before - len(edge_df)
    if n_dropped_nan > 0:
        log.warning(
            "[MediGraph] Dropped %d edge row(s) with NaN after node mapping.",
            n_dropped_nan,
        )

    # Cast to int64 after NaN removal
    edge_df["source"] = edge_df["source"].astype("int64")
    edge_df["target"] = edge_df["target"].astype("int64")

    # Remove self-loops
    n_before_sl = len(edge_df)
    edge_df = edge_df[edge_df["source"] != edge_df["target"]]
    n_self_loops = n_before_sl - len(edge_df)
    if n_self_loops > 0:
        log.info("Removed %d self-loop edge(s).", n_self_loops)

    # Keep only contract columns in the correct order
    edge_df = edge_df[["source", "relation", "target"]].reset_index(drop=True)

    return edge_df


def _build_relation_mapping(edge_df: pd.DataFrame) -> dict[str, int]:
    """
    Build a 0-indexed relation mapping from unique relation strings in *edge_df*.

    Relations are sorted alphabetically for reproducibility.

    Parameters
    ----------
    edge_df:
        Edges DataFrame with a ``relation`` column.

    Returns
    -------
    dict[str, int]
        ``{relation_string: relation_id}`` with contiguous IDs ``[0, R)``.
    """
    unique_relations = sorted(edge_df["relation"].dropna().unique().tolist())
    return {rel: idx for idx, rel in enumerate(unique_relations)}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _save_artefacts(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_mapping: dict[str, int],
    relation_mapping: dict[str, int],
    paths: dict,
) -> None:
    """
    Persist all four processed artefacts to disk.

    Parameters
    ----------
    node_df:
        Nodes DataFrame to write to ``nodes_csv`` path.
    edge_df:
        Edges DataFrame to write to ``edges_csv`` path.
    node_mapping:
        Dict to serialize to ``node_mapping_json`` path.
    relation_mapping:
        Dict to serialize to ``relation_mapping_json`` path.
    paths:
        ``cfg["paths"]`` dict with resolved absolute paths.
    """
    processed_dir = Path(paths["data_processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # nodes.csv
    nodes_out = Path(paths["nodes_csv"])
    node_df.to_csv(nodes_out, index=False)
    log.info("Saved nodes.csv → '%s' (%d rows).", nodes_out, len(node_df))

    # edges.csv
    edges_out = Path(paths["edges_csv"])
    edge_df.to_csv(edges_out, index=False)
    log.info("Saved edges.csv → '%s' (%d rows).", edges_out, len(edge_df))

    # node_mapping.json
    node_map_out = Path(paths["node_mapping_json"])
    with node_map_out.open("w", encoding="utf-8") as fh:
        json.dump(node_mapping, fh, ensure_ascii=False, indent=2)
    log.info(
        "Saved node_mapping.json → '%s' (%d entries).", node_map_out, len(node_mapping)
    )

    # relation_mapping.json
    rel_map_out = Path(paths["relation_mapping_json"])
    with rel_map_out.open("w", encoding="utf-8") as fh:
        json.dump(relation_mapping, fh, ensure_ascii=False, indent=2)
    log.info(
        "Saved relation_mapping.json → '%s' (%d entries).",
        rel_map_out,
        len(relation_mapping),
    )


# ---------------------------------------------------------------------------
# Statistics logging
# ---------------------------------------------------------------------------


def _log_statistics(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_mapping: dict[str, int],
    relation_mapping: dict[str, int],
) -> None:
    """
    Log preprocessing summary statistics at INFO level.

    Parameters
    ----------
    node_df:
        Processed nodes DataFrame.
    edge_df:
        Processed edges DataFrame.
    node_mapping:
        ``{entity_name: node_id}`` mapping.
    relation_mapping:
        ``{relation_string: relation_id}`` mapping.
    """
    num_nodes = len(node_df)
    num_edges = len(edge_df)
    num_node_types = node_df["entity_type"].nunique()
    num_relation_types = len(relation_mapping)

    log.info("=" * 60)
    log.info("MediGraph Preprocessing Statistics")
    log.info("=" * 60)
    log.info("  num_nodes          : %d", num_nodes)
    log.info("  num_edges          : %d", num_edges)
    log.info("  num_node_types     : %d", num_node_types)
    log.info("  num_relation_types : %d", num_relation_types)
    log.info("  node_mapping size  : %d", len(node_mapping))

    # Per-type node counts
    type_counts = node_df["entity_type"].value_counts().to_dict()
    for etype, count in sorted(type_counts.items()):
        log.info("    %-30s %d nodes", etype, count)

    # Per-relation edge counts
    rel_counts = edge_df["relation"].value_counts().to_dict()
    for rel, count in sorted(rel_counts.items()):
        log.info("    %-40s %d edges", rel, count)

    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description=(
            "MediGraph AI — Preprocessing pipeline. "
            "Converts data/raw/kg.csv into the four processed artefacts."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to config.yaml (default: auto-detected project-root/config.yaml)."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    configure_root_logger()

    args = _parse_args(sys.argv[1:])

    if args.config is not None:
        cfg = load_config(args.config)
    else:
        cfg = load_config()

    log.info("Starting MediGraph preprocessing pipeline …")
    try:
        node_df, edge_df, node_mapping, rel_mapping = preprocess(cfg)
        log.info("Preprocessing complete.")
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except (KeyError, ValueError) as exc:
        log.error("[MediGraph] Preprocessing failed: %s", exc)
        sys.exit(1)
