"""
src/training/train.py
---------------------
Full training loop for MediGraph AI (R-GCN + DistMult).

This module is both importable as a library function and runnable as a CLI
script::

    # Importable
    from src.training.train import train
    cfg = load_config()
    train(cfg)

    # CLI
    python src/training/train.py [--config config.yaml] [--epochs N] [--device cpu|cuda]

Training procedure
------------------
1. Seed everything for reproducibility.
2. Load ``nodes.csv`` and ``edges.csv`` via ``MediGraphLoader``.
3. Build edge tensors via ``GraphBuilder``.
4. Split edges into train / val / test with ``split_edges()``.
5. Instantiate ``MediGraphModel(RGCNEncoder(...), DistMultDecoder(...))``.
6. Adam optimiser with LR and weight-decay from config.
7. For each epoch:
   a. Sample negatives for the training partition.
   b. Encode on training edges only.
   c. Decode positive + negative edges → BCE loss.
   d. Backward + step.
   e. Every ``val_interval`` epochs: evaluate on val set, save checkpoint.
8. Final test-set evaluation.
9. Write ``artifacts/metrics/metrics_{timestamp}.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running as __main__
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# MediGraph imports (deferred so the path patch above takes effect first)
# ---------------------------------------------------------------------------
from src.utils.config import load_config                     # noqa: E402
from src.utils.logger import configure_root_logger, get_logger  # noqa: E402
from src.utils.seed import set_seed                          # noqa: E402
from src.training.splitter import split_edges, sample_negatives  # noqa: E402
from src.training.evaluator import LinkPredictionEvaluator   # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports for optional heavy dependencies (data/model layers written by
# Member 1 and Member 2).  We import them inside train() so that the module
# can still be imported in test environments where those modules are stubbed.
# ---------------------------------------------------------------------------

def _import_data_layer() -> tuple[Any, Any]:
    """Import MediGraphLoader and GraphBuilder (Member 1 deliverables)."""
    try:
        from src.data.loader import MediGraphLoader        # type: ignore
        from src.data.graph_builder import GraphBuilder    # type: ignore
        return MediGraphLoader, GraphBuilder
    except ImportError as exc:
        raise ImportError(
            f"[MediGraph] train: could not import data layer modules. "
            f"Ensure src/data/loader.py and src/data/graph_builder.py exist. "
            f"Original error: {exc}"
        ) from exc


def _import_model_layer() -> tuple[Any, Any, Any]:
    """Import RGCNEncoder, DistMultDecoder, MediGraphModel (Member 2)."""
    try:
        from src.models.rgcn import RGCNEncoder, MediGraphModel   # type: ignore
        from src.models.decoder import DistMultDecoder             # type: ignore
        return RGCNEncoder, DistMultDecoder, MediGraphModel
    except ImportError as exc:
        raise ImportError(
            f"[MediGraph] train: could not import model layer modules. "
            f"Ensure src/models/rgcn.py and src/models/decoder.py exist. "
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------


def train(cfg: dict[str, Any]) -> None:
    """
    Execute the full MediGraph AI training pipeline.

    This function orchestrates data loading, model construction, the training
    loop, periodic validation, checkpointing, and final test evaluation.  It
    writes one metrics JSON file to ``artifacts/metrics/``.

    Parameters
    ----------
    cfg : dict
        Fully-resolved configuration dictionary as returned by
        ``src.utils.config.load_config()``.  Must contain at minimum the
        sub-dicts ``"project"``, ``"paths"``, ``"model"``, and ``"training"``.

    Raises
    ------
    FileNotFoundError
        If ``nodes.csv`` or ``edges.csv`` do not exist at the configured
        paths.
    ImportError
        If the data or model layer modules (Member 1 / Member 2 work) are not
        yet importable.

    Notes
    -----
    - Only the **training** partition of edges is passed to
      ``RGCNEncoder.forward()`` — validation and test edges are withheld from
      message passing per §7.1 of the API contract.
    - The best checkpoint (by validation MRR) is saved to
      ``artifacts/checkpoints/best_model.pt``.
    - Periodic checkpoints are saved every ``val_interval`` epochs to
      ``artifacts/checkpoints/rgcn_epoch_{N}.pt``.
    """
    # ------------------------------------------------------------------ #
    # 0. Configure logging and device
    # ------------------------------------------------------------------ #
    log_level_str: str = cfg.get("logging", {}).get("level", "INFO").upper()
    import logging as _logging
    configure_root_logger(level=getattr(_logging, log_level_str, _logging.INFO))

    device_str: str = cfg.get("_device_override", "cpu")  # injected by CLI
    device = torch.device(device_str)
    log.info("Training device: %s", device)

    # ------------------------------------------------------------------ #
    # 1. Seed
    # ------------------------------------------------------------------ #
    seed: int = int(cfg["project"]["seed"])
    set_seed(seed)
    log.info("Random seed set to %d.", seed)

    # ------------------------------------------------------------------ #
    # 2. Load data
    # ------------------------------------------------------------------ #
    MediGraphLoader, GraphBuilder = _import_data_layer()

    nodes_path    = cfg["paths"]["nodes_csv"]
    edges_path    = cfg["paths"]["edges_csv"]
    nm_path       = cfg["paths"]["node_mapping_json"]
    rm_path       = cfg["paths"]["relation_mapping_json"]

    for label, fpath in [
        ("nodes_csv",             nodes_path),
        ("edges_csv",             edges_path),
        ("node_mapping_json",     nm_path),
        ("relation_mapping_json", rm_path),
    ]:
        if not Path(fpath).exists():
            raise FileNotFoundError(
                f"[MediGraph] train: required file '{label}' not found at "
                f"'{fpath}'.  Run `python src/data/preprocess.py` first."
            )

    loader = MediGraphLoader(
        nodes_path=nodes_path,
        edges_path=edges_path,
        node_mapping_path=nm_path,
        relation_mapping_path=rm_path,
    )
    node_df, edge_df, node_mapping, rel_mapping = loader.load()
    log.info("Loaded %d nodes and %d edges.", len(node_df), len(edge_df))

    num_nodes     = len(node_df)
    num_relations = len(rel_mapping)

    # ------------------------------------------------------------------ #
    # 3. Build edge tensors
    # ------------------------------------------------------------------ #
    builder    = GraphBuilder(node_df, edge_df, node_mapping, rel_mapping)
    edge_index, edge_type = builder.build_edge_tensors()
    # edge_index: (2, E)  edge_type: (E,)
    log.info("Edge tensors built: edge_index=%s, edge_type=%s.",
             tuple(edge_index.shape), tuple(edge_type.shape))

    # ------------------------------------------------------------------ #
    # 4. Split edges
    # ------------------------------------------------------------------ #
    train_cfg = cfg["training"]
    splits = split_edges(
        edge_index=edge_index,
        edge_type=edge_type,
        train_ratio=float(train_cfg["train_ratio"]),
        val_ratio=float(train_cfg["val_ratio"]),
        test_ratio=float(train_cfg["test_ratio"]),
        seed=seed,
    )
    train_ei   = splits["train"]["edge_index"].to(device)
    train_et   = splits["train"]["edge_type"].to(device)
    val_ei     = splits["val"]["edge_index"].to(device)
    val_et     = splits["val"]["edge_type"].to(device)
    test_ei    = splits["test"]["edge_index"].to(device)
    test_et    = splits["test"]["edge_type"].to(device)

    # Pre-compute the set of known positive (src, tgt) pairs for negative sampling
    known_edges: set[tuple[int, int]] = {
        (int(s), int(t))
        for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist())
    }

    # ------------------------------------------------------------------ #
    # 5. Build model
    # ------------------------------------------------------------------ #
    RGCNEncoder, DistMultDecoder, MediGraphModel = _import_model_layer()

    model_cfg = cfg["model"]
    encoder = RGCNEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        in_channels=int(model_cfg["in_channels"]),
        hidden_channels=int(model_cfg["hidden_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        num_layers=int(model_cfg["num_layers"]),
        num_bases=int(model_cfg["num_bases"]),
        dropout=float(model_cfg["dropout"]),
    )
    decoder = DistMultDecoder(
        num_relations=num_relations,
        embedding_dim=int(model_cfg["out_channels"]),
    )
    model = MediGraphModel(encoder=encoder, decoder=decoder).to(device)
    log.info(
        "Model instantiated: %s — %d parameters.",
        model.__class__.__name__,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ------------------------------------------------------------------ #
    # 6. Optimiser
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # ------------------------------------------------------------------ #
    # 7. Training loop
    # ------------------------------------------------------------------ #
    # Import tqdm here to keep it optional in library use
    try:
        from tqdm import tqdm  # type: ignore
        _use_tqdm = True
    except ImportError:
        log.warning("tqdm not installed; progress bars disabled.")
        _use_tqdm = False

    epochs        = int(train_cfg.get("epochs", 200))
    val_interval  = int(train_cfg.get("val_interval", 10))
    neg_ratio     = int(train_cfg.get("negative_sampling_ratio", 1))
    batch_size    = int(train_cfg.get("batch_size", 1024))

    checkpoints_dir = Path(cfg["paths"]["checkpoints"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    evaluator = LinkPredictionEvaluator(num_nodes=num_nodes, device=device_str)

    best_val_mrr   = -1.0
    best_epoch     = 0
    per_epoch_log: list[dict[str, Any]] = []

    epoch_iter = range(1, epochs + 1)
    if _use_tqdm:
        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    for epoch in epoch_iter:
        model.train()

        # ---- Sample negatives for this epoch's training batch ----
        # We sample from the full training partition; the caller can tune
        # batch_size to control memory.  Here we process all train edges per
        # epoch (standard for small-to-medium graphs).
        neg_ei, neg_et = sample_negatives(
            pos_edge_index=train_ei,
            pos_edge_type=train_et,
            num_nodes=num_nodes,
            num_neg_per_pos=neg_ratio,
            known_edges=known_edges,
        )

        # ---- Forward pass ----
        z = model.encode(train_ei, train_et)                     # (N, D)
        pos_scores = model.decode(z, train_ei, train_et)         # (E_train,)
        neg_scores = model.decode(z, neg_ei,   neg_et)           # (E_train * neg_ratio,)

        # Binary cross-entropy loss:  positives → 1,  negatives → 0
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([pos_labels, neg_labels]),
        )

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = float(loss.item())

        if _use_tqdm:
            epoch_iter.set_postfix(train_loss=f"{train_loss:.4f}")  # type: ignore[union-attr]

        # ---- Validation step ----
        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                # Val loss
                z_val     = model.encode(train_ei, train_et)
                vp_scores = model.decode(z_val, val_ei, val_et)
                vn_ei, vn_et = sample_negatives(
                    pos_edge_index=val_ei,
                    pos_edge_type=val_et,
                    num_nodes=num_nodes,
                    num_neg_per_pos=neg_ratio,
                    known_edges=known_edges,
                )
                vn_scores = model.decode(z_val, vn_ei, vn_et)
                val_loss = float(F.binary_cross_entropy_with_logits(
                    torch.cat([vp_scores, vn_scores]),
                    torch.cat([torch.ones_like(vp_scores), torch.zeros_like(vn_scores)]),
                ).item())

            # Filtered MRR / Hits@K on val set
            val_metrics = evaluator.evaluate_mrr_hits(
                model=model,
                eval_edge_index=val_ei,
                eval_edge_type=val_et,
                mp_edge_index=train_ei,
                mp_edge_type=train_et,
                batch_size=batch_size,
                hits_k=[1, 10, 50],
            )
            val_mrr     = val_metrics["mrr"]
            val_hits_1  = val_metrics.get("hits@1",  0.0)
            val_hits_10 = val_metrics.get("hits@10", 0.0)
            val_hits_50 = val_metrics.get("hits@50", 0.0)

            log.info(
                "Epoch %4d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_mrr=%.4f | val_hits@10=%.4f",
                epoch, epochs, train_loss, val_loss, val_mrr, val_hits_10,
            )

            per_epoch_log.append({
                "epoch":         epoch,
                "train_loss":    round(train_loss, 6),
                "val_loss":      round(val_loss,   6),
                "val_mrr":       round(val_mrr,    6),
                "val_hits_at_1": round(val_hits_1,  6),
                "val_hits_at_10":round(val_hits_10, 6),
                "val_hits_at_50":round(val_hits_50, 6),
            })

            # ---- Save periodic checkpoint ----
            periodic_path = checkpoints_dir / f"rgcn_epoch_{epoch}.pt"
            _save_checkpoint(
                path=periodic_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mrr=val_mrr,
                val_hits_at_10=val_hits_10,
                model_cfg=model_cfg,
                num_nodes=num_nodes,
                num_relations=num_relations,
                seed=seed,
            )
            log.info("Periodic checkpoint saved to '%s'.", periodic_path)

            # ---- Save best checkpoint ----
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_epoch   = epoch
                best_path    = checkpoints_dir / "best_model.pt"
                _save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_mrr=val_mrr,
                    val_hits_at_10=val_hits_10,
                    model_cfg=model_cfg,
                    num_nodes=num_nodes,
                    num_relations=num_relations,
                    seed=seed,
                )
                log.info(
                    "New best val_mrr=%.4f at epoch %d. Checkpoint saved to '%s'.",
                    best_val_mrr, best_epoch, best_path,
                )

    # ------------------------------------------------------------------ #
    # 8. Final test evaluation
    # ------------------------------------------------------------------ #
    log.info("Loading best model from '%s' for test evaluation.", best_path)
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_metrics = evaluator.evaluate_mrr_hits(
        model=model,
        eval_edge_index=test_ei,
        eval_edge_type=test_et,
        mp_edge_index=train_ei,
        mp_edge_type=train_et,
        batch_size=batch_size,
        hits_k=[1, 10, 50],
    )

    # Test AUC-ROC
    with torch.no_grad():
        z_test = model.encode(train_ei, train_et)
        pos_sc = model.decode(z_test, test_ei, test_et)
        tn_ei, tn_et = sample_negatives(
            pos_edge_index=test_ei,
            pos_edge_type=test_et,
            num_nodes=num_nodes,
            num_neg_per_pos=1,
            known_edges=known_edges,
        )
        neg_sc = model.decode(z_test, tn_ei, tn_et)

    test_auc = evaluator.compute_auc_roc(pos_sc, neg_sc)

    log.info(
        "Test results — MRR=%.4f | Hits@1=%.4f | Hits@10=%.4f | "
        "Hits@50=%.4f | AUC-ROC=%.4f",
        test_metrics["mrr"],
        test_metrics.get("hits@1",  0.0),
        test_metrics.get("hits@10", 0.0),
        test_metrics.get("hits@50", 0.0),
        test_auc,
    )

    # ------------------------------------------------------------------ #
    # 9. Save metrics JSON
    # ------------------------------------------------------------------ #
    run_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    metrics_payload: dict[str, Any] = {
        "run_id": run_id,
        "config": {
            "project": cfg.get("project", {}),
            "model":   cfg.get("model",   {}),
            "training":cfg.get("training",{}),
        },
        "per_epoch": per_epoch_log,
        "best": {
            "epoch":           best_epoch,
            "val_mrr":         round(best_val_mrr,                       6),
            "test_mrr":        round(test_metrics["mrr"],                 6),
            "test_hits_at_1":  round(test_metrics.get("hits@1",  0.0),   6),
            "test_hits_at_10": round(test_metrics.get("hits@10", 0.0),   6),
            "test_hits_at_50": round(test_metrics.get("hits@50", 0.0),   6),
            "test_auc_roc":    round(test_auc,                            6),
        },
    }

    metrics_dir = Path(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_path = metrics_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)
    log.info("Metrics saved to '%s'.", metrics_path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_mrr: float,
    val_hits_at_10: float,
    model_cfg: dict[str, Any],
    num_nodes: int,
    num_relations: int,
    seed: int,
) -> None:
    """
    Serialise a training checkpoint to disk in the format required by §6 of
    the API contract.

    Parameters
    ----------
    path : Path
        Destination file path.
    model : torch.nn.Module
        The model whose ``state_dict()`` will be saved.
    optimizer : torch.optim.Optimizer
        The optimiser whose ``state_dict()`` will be saved.
    epoch : int
        Current epoch number (1-indexed).
    train_loss : float
        Training loss at this epoch.
    val_loss : float
        Validation loss at this epoch.
    val_mrr : float
        Filtered MRR on the validation set.
    val_hits_at_10 : float
        Hits@10 on the validation set.
    model_cfg : dict
        Model hyperparameters from ``cfg["model"]``.
    num_nodes : int
        Total number of nodes (required to reconstruct the encoder).
    num_relations : int
        Total number of relation types.
    seed : int
        Random seed used during training.
    """
    try:
        import torch_geometric  # type: ignore
        pyg_version = torch_geometric.__version__
    except ImportError:
        pyg_version = "unknown"

    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        "train_loss":           train_loss,
        "val_loss":             val_loss,
        "val_mrr":              val_mrr,
        "val_hits_at_10":       val_hits_at_10,
        "model_config": {
            "num_nodes":       num_nodes,
            "num_relations":   num_relations,
            "in_channels":     int(model_cfg["in_channels"]),
            "hidden_channels": int(model_cfg["hidden_channels"]),
            "out_channels":    int(model_cfg["out_channels"]),
            "num_layers":      int(model_cfg["num_layers"]),
            "num_bases":       int(model_cfg["num_bases"]),
            "dropout":         float(model_cfg["dropout"]),
        },
        "random_seed":  seed,
        "torch_version": torch.__version__,
        "pyg_version":   pyg_version,
    }
    torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        prog="train",
        description="MediGraph AI — R-GCN training entry point.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (default: <project_root>/config.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override cfg['training']['epochs'].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"] + [f"cuda:{i}" for i in range(8)],
        help="PyTorch device (default: cpu).",
    )
    return parser


def main() -> None:
    """
    CLI entry point for ``python src/training/train.py``.

    Parses command-line arguments, loads configuration, applies overrides,
    and delegates to :func:`train`.
    """
    parser = _build_parser()
    args   = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
        log.info("CLI override: epochs=%d", args.epochs)

    # Inject device into cfg so train() can read it
    cfg["_device_override"] = args.device

    train(cfg)


if __name__ == "__main__":
    main()
