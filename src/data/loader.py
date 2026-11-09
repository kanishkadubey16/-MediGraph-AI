"""
src/data/loader.py
------------------
Data loading utilities for MediGraph AI.

Provides ``MediGraphLoader``, a class that reads the four processed data
artefacts produced by ``src/data/preprocess.py``, validates their schemas,
and returns strongly-typed Python objects ready for downstream consumers
(graph builder, training, inference, Streamlit app).

Canonical artefact paths are read from ``config.yaml`` via
``src.utils.config.load_config`` — never hardcoded here.

Usage
-----
    from src.data.loader import MediGraphLoader
    from src.utils.config import load_config

    cfg    = load_config()
    loader = MediGraphLoader()
    node_df, edge_df, node_mapping, rel_mapping = loader.load_all(cfg)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Required columns per the API contract (§2.1, §2.2)
# ---------------------------------------------------------------------------
_NODE_REQUIRED_COLS: list[str] = ["node_id", "entity_name", "entity_type", "source_id"]
_EDGE_REQUIRED_COLS: list[str] = ["source", "relation", "target"]


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def validate_edge_referential_integrity(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
) -> None:
    """
    Raise ``ValueError`` if any source or target node ID in *edge_df* does
    not exist in *node_df*.

    Parameters
    ----------
    edge_df:
        Edges DataFrame with at least columns ``source`` and ``target``
        (integer node IDs).
    node_df:
        Nodes DataFrame with at least column ``node_id`` (integer).

    Raises
    ------
    ValueError
        If any ``source`` or ``target`` value is absent from
        ``node_df["node_id"]``.
    """
    valid_ids: set[int] = set(node_df["node_id"].tolist())

    bad_sources = set(edge_df.loc[~edge_df["source"].isin(valid_ids), "source"].unique())
    if bad_sources:
        raise ValueError(
            f"[MediGraph] Referential integrity violation: {len(bad_sources)} "
            f"edge source node_id(s) not found in nodes DataFrame. "
            f"Example offending IDs: {sorted(bad_sources)[:10]}"
        )

    bad_targets = set(edge_df.loc[~edge_df["target"].isin(valid_ids), "target"].unique())
    if bad_targets:
        raise ValueError(
            f"[MediGraph] Referential integrity violation: {len(bad_targets)} "
            f"edge target node_id(s) not found in nodes DataFrame. "
            f"Example offending IDs: {sorted(bad_targets)[:10]}"
        )

    log.debug("Edge referential integrity check passed.")


# ---------------------------------------------------------------------------
# MediGraphLoader
# ---------------------------------------------------------------------------


class MediGraphLoader:
    """
    Loads and validates the four processed MediGraph AI artefacts:

    * ``nodes.csv``             — node metadata table
    * ``edges.csv``             — edge list (integer source/target IDs)
    * ``node_mapping.json``     — ``{entity_name: node_id}``
    * ``relation_mapping.json`` — ``{relation_string: relation_id}``

    All validation logic mirrors the schemas defined in
    ``docs/API_CONTRACT.md`` §2 and §3. Errors are raised with a
    ``[MediGraph]`` prefix for easy log filtering.

    Parameters
    ----------
    check_referential_integrity:
        When ``True`` (default), :py:meth:`load_edges` will verify that every
        ``source`` and ``target`` node ID exists in the nodes DataFrame that
        was passed as *node_df*.  Set to ``False`` to skip this potentially
        expensive check on very large graphs.

    Example
    -------
    ::

        from src.data.loader import MediGraphLoader
        from src.utils.config import load_config

        cfg    = load_config()
        loader = MediGraphLoader()
        node_df, edge_df, node_map, rel_map = loader.load_all(cfg)
    """

    def __init__(self, check_referential_integrity: bool = True) -> None:
        self._check_ri = check_referential_integrity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_nodes(self, path: str) -> pd.DataFrame:
        """
        Load and validate ``nodes.csv``.

        Validation rules (per API contract §2.1):

        * Required columns: ``node_id``, ``entity_name``, ``entity_type``,
          ``source_id``.
        * ``node_id`` must form a contiguous integer range ``[0, N)``.
        * No ``NaN`` values in any required column.
        * No duplicate ``node_id`` values.

        Parameters
        ----------
        path:
            Absolute or relative path to ``nodes.csv``.

        Returns
        -------
        pd.DataFrame
            Validated nodes DataFrame with ``node_id`` as ``int64``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at *path*.
        KeyError
            If any required column is absent.
        ValueError
            If ``node_id`` is not a contiguous range, contains duplicates,
            or any required column contains ``NaN``.
        """
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"[MediGraph] nodes.csv not found at '{resolved}'. "
                f"Run `python src/data/preprocess.py` to generate it."
            )

        log.info("Loading nodes from '%s' …", resolved)
        df = pd.read_csv(resolved, dtype={"node_id": "int64", "source_id": str})

        # --- column presence ---
        self._assert_required_columns(df, _NODE_REQUIRED_COLS, str(resolved))

        # --- NaN checks ---
        for col in _NODE_REQUIRED_COLS:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                raise ValueError(
                    f"[MediGraph] nodes.csv column '{col}' contains {n_null} "
                    f"NaN value(s). All required columns must be fully populated."
                )

        # --- duplicate node_id ---
        n_dupes = int(df["node_id"].duplicated().sum())
        if n_dupes > 0:
            raise ValueError(
                f"[MediGraph] nodes.csv has {n_dupes} duplicate node_id value(s). "
                f"node_id must be unique across all rows."
            )

        # --- contiguous integer range [0, N) ---
        self._assert_contiguous_node_ids(df)

        log.info("Loaded %d nodes (types: %s).", len(df), df["entity_type"].nunique())
        return df

    def load_edges(
        self,
        path: str,
        node_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Load and validate ``edges.csv``.

        Validation rules (per API contract §2.2):

        * Required columns: ``source``, ``relation``, ``target``.
        * No ``NaN`` values in any required column.
        * If *node_df* is provided (and ``check_referential_integrity`` is
          ``True`` on the loader), referential integrity is verified — every
          ``source`` and ``target`` must appear in ``node_df["node_id"]``.

        Parameters
        ----------
        path:
            Absolute or relative path to ``edges.csv``.
        node_df:
            Optional nodes DataFrame (output of :py:meth:`load_nodes`).
            When provided, referential integrity is validated.

        Returns
        -------
        pd.DataFrame
            Validated edges DataFrame with ``source`` and ``target`` as
            ``int64``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at *path*.
        KeyError
            If any required column is absent.
        ValueError
            If any required column contains ``NaN``, or if referential
            integrity is violated.
        """
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"[MediGraph] edges.csv not found at '{resolved}'. "
                f"Run `python src/data/preprocess.py` to generate it."
            )

        log.info("Loading edges from '%s' …", resolved)
        df = pd.read_csv(
            resolved,
            dtype={"source": "int64", "target": "int64", "relation": str},
        )

        # --- column presence ---
        self._assert_required_columns(df, _EDGE_REQUIRED_COLS, str(resolved))

        # --- NaN checks ---
        for col in _EDGE_REQUIRED_COLS:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                raise ValueError(
                    f"[MediGraph] edges.csv column '{col}' contains {n_null} "
                    f"NaN value(s). All required columns must be fully populated."
                )

        # --- referential integrity (optional) ---
        if node_df is not None and self._check_ri:
            validate_edge_referential_integrity(df, node_df)

        log.info(
            "Loaded %d edges (%d unique relation types).",
            len(df),
            df["relation"].nunique(),
        )
        return df

    def load_node_mapping(self, path: str) -> dict[str, int]:
        """
        Load ``node_mapping.json`` (``{entity_name: node_id}``).

        Parameters
        ----------
        path:
            Absolute or relative path to ``node_mapping.json``.

        Returns
        -------
        dict[str, int]
            Mapping from entity name string to integer node ID.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at *path*.
        ValueError
            If the JSON does not decode to a ``dict``, or if any value is
            not an integer.
        """
        return self._load_mapping_json(path, label="node_mapping")

    def load_relation_mapping(self, path: str) -> dict[str, int]:
        """
        Load ``relation_mapping.json`` (``{relation_string: relation_id}``).

        Parameters
        ----------
        path:
            Absolute or relative path to ``relation_mapping.json``.

        Returns
        -------
        dict[str, int]
            Mapping from relation type string to integer relation ID.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at *path*.
        ValueError
            If the JSON does not decode to a ``dict``, or if any value is
            not an integer.
        """
        return self._load_mapping_json(path, label="relation_mapping")

    def load_all(
        self,
        cfg: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
        """
        Convenience method: load all four artefacts using paths from *cfg*.

        Paths are read from ``cfg["paths"]`` (resolved absolute strings as
        produced by :py:func:`src.utils.config.load_config`):

        * ``cfg["paths"]["nodes_csv"]``
        * ``cfg["paths"]["edges_csv"]``
        * ``cfg["paths"]["node_mapping_json"]``
        * ``cfg["paths"]["relation_mapping_json"]``

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
        KeyError
            If any required path key is absent from ``cfg["paths"]``.
        FileNotFoundError / ValueError
            Propagated from the individual ``load_*`` methods.
        """
        try:
            paths = cfg["paths"]
        except KeyError as exc:
            raise KeyError(
                f"[MediGraph] 'paths' key missing from configuration dict. "
                f"Ensure load_config() was called correctly."
            ) from exc

        required_keys = [
            "nodes_csv",
            "edges_csv",
            "node_mapping_json",
            "relation_mapping_json",
        ]
        for key in required_keys:
            if key not in paths:
                raise KeyError(
                    f"[MediGraph] Required path key '{key}' not found in "
                    f"cfg['paths']. Check config.yaml."
                )

        log.info("Loading all MediGraph artefacts …")
        node_df = self.load_nodes(paths["nodes_csv"])
        edge_df = self.load_edges(paths["edges_csv"], node_df=node_df)
        node_mapping = self.load_node_mapping(paths["node_mapping_json"])
        rel_mapping = self.load_relation_mapping(paths["relation_mapping_json"])

        log.info(
            "All artefacts loaded — %d nodes, %d edges, %d relations.",
            len(node_df),
            len(edge_df),
            len(rel_mapping),
        )
        return node_df, edge_df, node_mapping, rel_mapping

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_required_columns(
        df: pd.DataFrame,
        required: list[str],
        file_path: str,
    ) -> None:
        """Raise ``KeyError`` if any column in *required* is absent from *df*."""
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"[MediGraph] File '{file_path}' is missing required "
                f"column(s): {missing}. "
                f"Expected columns: {required}."
            )

    @staticmethod
    def _assert_contiguous_node_ids(df: pd.DataFrame) -> None:
        """
        Raise ``ValueError`` if ``df['node_id']`` is not the contiguous range
        ``[0, N)`` where N = len(df).
        """
        sorted_ids = df["node_id"].sort_values().reset_index(drop=True)
        expected = pd.RangeIndex(start=0, stop=len(df), step=1)
        if not sorted_ids.equals(pd.Series(expected, dtype="int64")):
            actual_min = int(df["node_id"].min())
            actual_max = int(df["node_id"].max())
            raise ValueError(
                f"[MediGraph] node_id values are not a contiguous range [0, {len(df)}). "
                f"Found range [{actual_min}, {actual_max}] with {len(df)} rows. "
                f"Re-run `python src/data/preprocess.py` to regenerate the processed data."
            )

    @staticmethod
    def _load_mapping_json(path: str, label: str) -> dict[str, int]:
        """
        Load a JSON file and return it as a ``dict[str, int]``.

        Parameters
        ----------
        path:
            Path to the JSON file.
        label:
            Human-readable label used in error messages (e.g. ``"node_mapping"``).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the JSON does not contain a ``dict``, or values are not ints.
        """
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"[MediGraph] {label}.json not found at '{resolved}'. "
                f"Run `python src/data/preprocess.py` to generate it."
            )

        log.info("Loading %s from '%s' …", label, resolved)
        with resolved.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            raise ValueError(
                f"[MediGraph] {label}.json must be a JSON object (dict), "
                f"got {type(data).__name__} instead."
            )

        # Coerce values to int and validate
        coerced: dict[str, int] = {}
        for k, v in data.items():
            try:
                coerced[str(k)] = int(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[MediGraph] {label}.json has non-integer value for key "
                    f"'{k}': {v!r}. All values must be integer node/relation IDs."
                ) from exc

        log.info("Loaded %s with %d entries.", label, len(coerced))
        return coerced
