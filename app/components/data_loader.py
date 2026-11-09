"""
app/components/data_loader.py
-----------------------------
Streamlit data-loading helpers for MediGraph AI.

Populates ``st.session_state`` with the keys defined in API_CONTRACT.md §9.1.
All public functions are safe to call multiple times; they short-circuit when
the requested artefact is already in session state.

Usage::

    from app.components.data_loader import load_project_data, load_model_checkpoint, get_data_status

    cfg = load_config()
    if load_project_data(cfg):
        st.success("Data ready!")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_project_data(cfg: dict) -> bool:
    """Load graph data files into ``st.session_state``.

    Reads the four processed artefacts defined in API_CONTRACT.md §1:

    * ``data/processed/nodes.csv``  → ``st.session_state.node_df``
    * ``data/processed/edges.csv``  → builds ``st.session_state.edge_index``
      and ``st.session_state.edge_type``
    * ``data/processed/node_mapping.json``    → ``st.session_state.node_mapping``
    * ``data/processed/relation_mapping.json`` → ``st.session_state.rel_mapping``

    If any file is missing a :class:`st.warning` is displayed and ``False``
    is returned so the caller can gate downstream UI.

    Parameters
    ----------
    cfg : dict
        Config dict returned by :func:`src.utils.config.load_config`.

    Returns
    -------
    bool
        ``True`` if all four artefacts were loaded successfully, ``False``
        if any file was absent or could not be parsed.
    """
    paths = cfg.get("paths", {})

    nodes_path = Path(paths.get("nodes_csv", "data/processed/nodes.csv"))
    edges_path = Path(paths.get("edges_csv", "data/processed/edges.csv"))
    node_map_path = Path(
        paths.get("node_mapping_json", "data/processed/node_mapping.json")
    )
    rel_map_path = Path(
        paths.get(
            "relation_mapping_json", "data/processed/relation_mapping.json"
        )
    )

    missing = [
        str(p)
        for p in (nodes_path, edges_path, node_map_path, rel_map_path)
        if not p.exists()
    ]
    if missing:
        st.warning(
            "⚠️ **[MediGraph] Data files not found.** "
            "Run `python src/data/preprocess.py` first.\n\n"
            "Missing:\n" + "\n".join(f"- `{m}`" for m in missing),
            icon="⚠️",
        )
        log.warning(
            "[MediGraph] load_project_data: missing files: %s", missing
        )
        return False

    try:
        # --- nodes.csv ---------------------------------------------------
        node_df: pd.DataFrame = pd.read_csv(nodes_path)
        _validate_node_df(node_df, nodes_path)

        # --- edges.csv ---------------------------------------------------
        edge_df: pd.DataFrame = pd.read_csv(edges_path)
        _validate_edge_df(edge_df, edges_path)

        # Build tensors (API_CONTRACT.md §4)
        edge_index = torch.tensor(
            edge_df[["source", "target"]].values.T, dtype=torch.long
        )  # shape (2, E)
        edge_type_arr: torch.Tensor

        # --- node_mapping.json -------------------------------------------
        with open(node_map_path, "r", encoding="utf-8") as fh:
            node_mapping: dict[str, int] = json.load(fh)

        # --- relation_mapping.json ----------------------------------------
        with open(rel_map_path, "r", encoding="utf-8") as fh:
            rel_mapping: dict[str, int] = json.load(fh)

        # Map relation strings to integer IDs for edge_type tensor
        edge_type_arr = torch.tensor(
            edge_df["relation"].map(rel_mapping).values, dtype=torch.long
        )  # shape (E,)

        # --- Write to session state (§9.1) --------------------------------
        st.session_state.node_df = node_df
        st.session_state.edge_index = edge_index
        st.session_state.edge_type = edge_type_arr
        st.session_state.node_mapping = node_mapping
        st.session_state.rel_mapping = rel_mapping

        log.info(
            "[MediGraph] load_project_data: loaded %d nodes, %d edges, "
            "%d relation types.",
            len(node_df),
            edge_index.shape[1],
            len(rel_mapping),
        )
        return True

    except Exception as exc:  # noqa: BLE001
        st.error(
            f"❌ **[MediGraph] Failed to load project data:** {exc}",
            icon="❌",
        )
        log.error("[MediGraph] load_project_data error: %s", exc, exc_info=True)
        return False


def load_model_checkpoint(checkpoint_path: str, cfg: dict) -> bool:
    """Load a MediGraphModel from a ``.pt`` checkpoint into session state.

    Instantiates :class:`src.models.rgcn.RGCNEncoder` and
    :class:`src.models.decoder.DistMultDecoder` from the architecture config
    embedded in the checkpoint (API_CONTRACT.md §6), wraps them in a
    :class:`src.models.rgcn.MediGraphModel`, loads the saved weights, and
    sets the model to ``eval()`` mode before writing to
    ``st.session_state.model``.

    Parameters
    ----------
    checkpoint_path : str
        Absolute or relative path to the ``.pt`` checkpoint file.
    cfg : dict
        Config dict (unused today, reserved for future override logic).

    Returns
    -------
    bool
        ``True`` on success, ``False`` if loading failed.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        st.error(
            f"❌ **[MediGraph] Checkpoint not found:** `{ckpt_path}`.  "
            "Train the model first with `python src/training/train.py`.",
            icon="❌",
        )
        log.error(
            "[MediGraph] load_model_checkpoint: file not found: %s", ckpt_path
        )
        return False

    try:
        from src.models.rgcn import MediGraphModel, RGCNEncoder
        from src.models.decoder import DistMultDecoder

        checkpoint: dict[str, Any] = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )

        model_cfg: dict = checkpoint["model_config"]

        encoder = RGCNEncoder(
            num_nodes=model_cfg["num_nodes"],
            num_relations=model_cfg["num_relations"],
            in_channels=model_cfg["in_channels"],
            hidden_channels=model_cfg["hidden_channels"],
            out_channels=model_cfg["out_channels"],
            num_layers=model_cfg["num_layers"],
            num_bases=model_cfg["num_bases"],
            dropout=model_cfg["dropout"],
        )
        decoder = DistMultDecoder(
            num_relations=model_cfg["num_relations"],
            embedding_dim=model_cfg["out_channels"],
        )
        model = MediGraphModel(encoder=encoder, decoder=decoder)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Persist model and checkpoint metadata
        st.session_state.model = model
        st.session_state["_checkpoint_meta"] = {
            "epoch": checkpoint.get("epoch"),
            "val_mrr": checkpoint.get("val_mrr"),
            "val_hits_at_10": checkpoint.get("val_hits_at_10"),
            "train_loss": checkpoint.get("train_loss"),
            "val_loss": checkpoint.get("val_loss"),
            "checkpoint_path": str(ckpt_path),
        }

        log.info(
            "[MediGraph] load_model_checkpoint: loaded epoch=%s, "
            "val_mrr=%.4f from '%s'.",
            checkpoint.get("epoch"),
            checkpoint.get("val_mrr", 0.0),
            ckpt_path,
        )
        return True

    except Exception as exc:  # noqa: BLE001
        st.error(
            f"❌ **[MediGraph] Failed to load checkpoint:** {exc}",
            icon="❌",
        )
        log.error(
            "[MediGraph] load_model_checkpoint error: %s", exc, exc_info=True
        )
        return False


def get_data_status() -> dict[str, Any]:
    """Return a snapshot of the current session-state data status.

    Returns
    -------
    dict
        A dict with the following keys:

        * ``data_loaded`` (bool): True if ``node_df`` and ``edge_index`` are
          both present in session state.
        * ``model_loaded`` (bool): True if ``model`` is present.
        * ``num_nodes`` (int): Number of rows in ``node_df`` (0 if not loaded).
        * ``num_edges`` (int): Number of edges in ``edge_index`` (0 if not
          loaded).
    """
    node_df: pd.DataFrame | None = st.session_state.get("node_df")
    edge_index: torch.Tensor | None = st.session_state.get("edge_index")
    model = st.session_state.get("model")

    data_loaded = node_df is not None and edge_index is not None
    model_loaded = model is not None

    return {
        "data_loaded": data_loaded,
        "model_loaded": model_loaded,
        "num_nodes": int(len(node_df)) if data_loaded and node_df is not None else 0,
        "num_edges": int(edge_index.shape[1]) if data_loaded and edge_index is not None else 0,
    }


# ---------------------------------------------------------------------------
# Private validation helpers
# ---------------------------------------------------------------------------


def _validate_node_df(df: pd.DataFrame, path: Path) -> None:
    """Raise :class:`KeyError` if required columns are absent (§2.1)."""
    required = {"node_id", "entity_name", "entity_type", "source_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"[MediGraph] nodes.csv at '{path}' is missing columns: {missing}. "
            "Check API_CONTRACT.md §2.1."
        )


def _validate_edge_df(df: pd.DataFrame, path: Path) -> None:
    """Raise :class:`KeyError` if required columns are absent (§2.2)."""
    required = {"source", "relation", "target"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"[MediGraph] edges.csv at '{path}' is missing columns: {missing}. "
            "Check API_CONTRACT.md §2.2."
        )
