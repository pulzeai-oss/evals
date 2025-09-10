"""
Evaluators module for benchmark evaluations.
"""

from .base_evaluator import BaseEvaluator
from .financebench_evaluator import FinanceBenchEvaluator
from .marketing_evaluator import MarketingEvaluator
from .mmlu_evaluator import MMLUEvaluator
from .pulze_evaluator import PulzeEvaluator

__all__ = [
    "BaseEvaluator",
    "FinanceBenchEvaluator",
    "MMLUEvaluator",
    "PulzeEvaluator",
    "MarketingEvaluator",
    "get_evaluator",
]


def get_evaluator(benchmark: str, config: dict) -> BaseEvaluator:
    """
    Factory function to get the appropriate evaluator for a benchmark.

    Args:
        benchmark: Name of the benchmark
        config: Configuration dictionary

    Returns:
        Evaluator instance

    Raises:
        ValueError: If benchmark is not supported
    """
    evaluators = {
        "financebench": FinanceBenchEvaluator,
        "mmlu": MMLUEvaluator,
        "pulze": PulzeEvaluator,
        "marketing": MarketingEvaluator,
    }

    if benchmark not in evaluators:
        available = ", ".join(evaluators.keys())
        raise ValueError(f"Unsupported benchmark: {benchmark}. Available: {available}")

    return evaluators[benchmark](config)
