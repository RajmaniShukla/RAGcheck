"""Abstract base class for all evaluators."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ragcheck.core.judges.base import BaseJudge
from ragcheck.core.schema import EvalSample, MetricName, MetricScore


class BaseEvaluator(ABC):
    """All evaluators implement this interface."""

    metric: MetricName

    def __init__(self, judge: BaseJudge) -> None:
        self.judge = judge

    @abstractmethod
    async def evaluate(self, sample: EvalSample) -> MetricScore:
        """Score a single sample. Never raises — errors are captured in MetricScore.error."""

    def _error_score(self, error: str) -> MetricScore:
        return MetricScore(metric=self.metric, score=0.0, reasoning="", error=error)

    def _fmt_contexts(self, contexts: list[str]) -> str:
        return "\n\n".join(
            f"[Chunk {i + 1}]\n{c.strip()}" for i, c in enumerate(contexts)
        )
