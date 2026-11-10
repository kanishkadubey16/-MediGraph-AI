"""
src.training — Training loop, evaluation, negative sampling, and baselines.
Member 3 responsibility.
"""

from src.training.train import train
from src.training.negative_sampling import TypedNegativeSampler, corrupt_triples, sample_negatives
from src.training.splitter import split_edges
from src.training.evaluate import evaluate_link_prediction, evaluate_model
from src.training.evaluator import LinkPredictionEvaluator
from src.training.baseline import DistMultBaseline, train_baseline, compare_rgcn_and_baseline

__all__ = [
    "train",
    "TypedNegativeSampler",
    "corrupt_triples",
    "sample_negatives",
    "split_edges",
    "evaluate_link_prediction",
    "evaluate_model",
    "LinkPredictionEvaluator",
    "DistMultBaseline",
    "train_baseline",
    "compare_rgcn_and_baseline",
]
