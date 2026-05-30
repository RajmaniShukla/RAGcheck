"""
RAGcheck Pipeline
-----------------
Orchestrates all evaluators across a dataset with concurrency control,
progress reporting, and aggregate statistics.
"""
from __future__ import annotations

import asyncio
import logging
import math
import statistics
from typing import Callable

from ragcheck.core.evaluators import build_evaluators
from ragcheck.core.judges import build_judge
from ragcheck.core.schema import (
    AggregateStats,
    EvalConfig,
    EvalDataset,
    EvalReport,
    EvalSample,
    MetricName,
    MetricScore,
    SampleResult,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main evaluation pipeline.

    Usage:
        pipeline = Pipeline(config)
        report = await pipeline.run(dataset)
    """

    def __init__(
        self,
        config: EvalConfig,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self._progress_callback = progress_callback
        self._judge = build_judge(config.judge)
        self._evaluators = build_evaluators(config.metrics, self._judge)

    async def run(self, dataset: EvalDataset) -> EvalReport:
        """Run all configured metrics across all samples."""
        logger.info(
            "Starting evaluation: %d samples × %d metrics (concurrency=%d)",
            len(dataset.samples),
            len(self._evaluators),
            self.config.concurrency,
        )

        semaphore = asyncio.Semaphore(self.config.concurrency)
        completed = 0

        async def eval_sample(sample: EvalSample) -> SampleResult:
            nonlocal completed
            async with semaphore:
                scores = await self._eval_one(sample)
                completed += 1
                if self._progress_callback:
                    self._progress_callback(completed, len(dataset.samples))
                logger.debug(
                    "Sample %d/%d done — aggregate=%.3f",
                    completed,
                    len(dataset.samples),
                    _aggregate(scores),
                )
                return SampleResult(sample=sample, scores=scores)

        results = await asyncio.gather(*[eval_sample(s) for s in dataset.samples])

        aggregate_stats = _compute_aggregate_stats(list(results), self.config.metrics)
        overall = (
            statistics.mean(s.mean for s in aggregate_stats) if aggregate_stats else 0.0
        )
        overall = round(overall, 4)

        passed: bool | None = None
        if self.config.fail_threshold is not None:
            passed = overall >= self.config.fail_threshold

        return EvalReport(
            dataset_name=dataset.name,
            config=self.config,
            results=list(results),
            aggregate_stats=aggregate_stats,
            overall_score=overall,
            passed=passed,
        )

    async def _eval_one(self, sample: EvalSample) -> list[MetricScore]:
        """Run all evaluators on a single sample concurrently."""
        tasks = [ev.evaluate(sample) for ev in self._evaluators]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _aggregate(scores: list[MetricScore]) -> float:
    valid = [s.score for s in scores if s.error is None]
    return round(sum(valid) / len(valid), 4) if valid else 0.0


def _compute_aggregate_stats(
    results: list[SampleResult], metrics: list[MetricName]
) -> list[AggregateStats]:
    stats = []
    for metric in metrics:
        values = []
        failed = 0
        for r in results:
            ms = r.score_for(metric)
            if ms is None:
                continue
            if ms.error:
                failed += 1
            else:
                values.append(ms.score)

        if not values:
            stats.append(
                AggregateStats(
                    metric=metric, mean=0.0, min=0.0, max=0.0, std=0.0, failed_samples=failed
                )
            )
            continue

        mean = round(statistics.mean(values), 4)
        std = round(statistics.stdev(values) if len(values) > 1 else 0.0, 4)
        stats.append(
            AggregateStats(
                metric=metric,
                mean=mean,
                min=round(min(values), 4),
                max=round(max(values), 4),
                std=std,
                failed_samples=failed,
            )
        )
    return stats
