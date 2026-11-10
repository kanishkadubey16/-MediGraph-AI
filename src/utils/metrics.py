"""
src/utils/metrics.py
--------------------
Link prediction and classification metric utilities for MediGraph AI.

Provides:
  - compute_mrr: Mean Reciprocal Rank
  - compute_hits_at_k: Hits at rank K (e.g. Hits@1, Hits@3, Hits@10)
  - compute_roc_auc: Area Under the ROC Curve
  - compute_classification_metrics: Precision, Recall, F1-score, Accuracy, ROC-AUC
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def compute_mrr(ranks: Tensor | list[float] | np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = (1 / |Q|) * sum_{q in Q} (1 / rank_q)
    where rank_q is the 1-based rank of the true entity.
    """
    if isinstance(ranks, (list, tuple)):
        ranks = np.array(ranks, dtype=np.float64)
    elif isinstance(ranks, Tensor):
        ranks = ranks.detach().cpu().numpy().astype(np.float64)

    if len(ranks) == 0:
        return 0.0

    reciprocal_ranks = 1.0 / ranks
    return float(np.mean(reciprocal_ranks))


def compute_hits_at_k(ranks: Tensor | list[float] | np.ndarray, k: int) -> float:
    """
    Compute Hits@K: proportion of true entities ranked at or above position K.

    Parameters
    ----------
    ranks : Tensor | list[float] | np.ndarray
        1-based integer ranks of true entities.
    k : int
        Cutoff threshold (e.g., 1, 3, 10).
    """
    if isinstance(ranks, (list, tuple)):
        ranks = np.array(ranks, dtype=np.float64)
    elif isinstance(ranks, Tensor):
        ranks = ranks.detach().cpu().numpy().astype(np.float64)

    if len(ranks) == 0:
        return 0.0

    return float(np.mean(ranks <= k))


def compute_roc_auc(pos_scores: Tensor | np.ndarray, neg_scores: Tensor | np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic (ROC-AUC).

    Parameters
    ----------
    pos_scores : Tensor | np.ndarray
        Raw model scores (logits) for true positive edges.
    neg_scores : Tensor | np.ndarray
        Raw model scores (logits) for corrupted negative edges.
    """
    if isinstance(pos_scores, Tensor):
        pos_scores = pos_scores.detach().cpu().numpy()
    if isinstance(neg_scores, Tensor):
        neg_scores = neg_scores.detach().cpu().numpy()

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_scores = np.concatenate([pos_scores, neg_scores])

    # If all scores are identical, roc_auc_score raises ValueError
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return 0.5


def compute_classification_metrics(
    pos_scores: Tensor | np.ndarray,
    neg_scores: Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute Precision, Recall, F1, Accuracy, and ROC-AUC on binary classification setup.

    Raw scores are transformed to probabilities via sigmoid: p = 1 / (1 + exp(-score)).
    Predictions are positive if p >= threshold.

    Parameters
    ----------
    pos_scores : Tensor | np.ndarray
        Logits for positive edges.
    neg_scores : Tensor | np.ndarray
        Logits for negative edges.
    threshold : float
        Decision threshold in [0, 1] on sigmoid probabilities. Default: 0.5 (logit 0.0).

    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'precision', 'recall', 'f1', 'accuracy', 'roc_auc'.
    """
    if isinstance(pos_scores, Tensor):
        pos_scores = pos_scores.detach().cpu().numpy()
    if isinstance(neg_scores, Tensor):
        neg_scores = neg_scores.detach().cpu().numpy()

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "roc_auc": 0.5,
        }

    y_true = np.concatenate([np.ones(len(pos_scores), dtype=int), np.zeros(len(neg_scores), dtype=int)])
    raw_scores = np.concatenate([pos_scores, neg_scores])

    # Numerical stable sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -50.0, 50.0)))
    y_pred = (probs >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = 0.5

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4),
    }
