"""Evaluator registry and factory."""
from __future__ import annotations

from ragcheck.core.evaluators.answer_relevance import AnswerRelevanceEvaluator
from ragcheck.core.evaluators.base import BaseEvaluator
from ragcheck.core.evaluators.chunk_utilization import ChunkUtilizationEvaluator
from ragcheck.core.evaluators.context_recall import ContextRecallEvaluator
from ragcheck.core.evaluators.context_relevance import ContextRelevanceEvaluator
from ragcheck.core.evaluators.faithfulness import FaithfulnessEvaluator
from ragcheck.core.evaluators.noise_sensitivity import NoiseSensitivityEvaluator
from ragcheck.core.judges.base import BaseJudge
from ragcheck.core.schema import MetricName

_REGISTRY: dict[MetricName, type[BaseEvaluator]] = {
    MetricName.CONTEXT_RELEVANCE: ContextRelevanceEvaluator,
    MetricName.FAITHFULNESS: FaithfulnessEvaluator,
    MetricName.ANSWER_RELEVANCE: AnswerRelevanceEvaluator,
    MetricName.CONTEXT_RECALL: ContextRecallEvaluator,
    MetricName.NOISE_SENSITIVITY: NoiseSensitivityEvaluator,
    MetricName.CHUNK_UTILIZATION: ChunkUtilizationEvaluator,
}


def build_evaluators(metrics: list[MetricName], judge: BaseJudge) -> list[BaseEvaluator]:
    """Instantiate evaluators for the requested metrics."""
    return [_REGISTRY[m](judge) for m in metrics]


__all__ = [
    "BaseEvaluator",
    "ContextRelevanceEvaluator",
    "FaithfulnessEvaluator",
    "AnswerRelevanceEvaluator",
    "ContextRecallEvaluator",
    "NoiseSensitivityEvaluator",
    "ChunkUtilizationEvaluator",
    "build_evaluators",
]
