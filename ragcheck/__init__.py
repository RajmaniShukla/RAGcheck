"""
ragcheck — An open-source toolkit to measure, debug, and improve your RAG pipeline quality.

Quick start:
    from ragcheck import evaluate

    results = evaluate(
        questions=["What is RAG?"],
        contexts=[["RAG stands for...", "It was introduced by..."]],
        answers=["RAG is a technique that..."],
        metrics=["faithfulness", "context_relevance", "answer_relevance"],
    )
    print(results.summary())
"""
from __future__ import annotations

import asyncio
from typing import Any

from ragcheck.core.pipeline import Pipeline
from ragcheck.core.schema import (
    ALL_METRICS,
    EvalConfig,
    EvalDataset,
    EvalReport,
    EvalSample,
    JudgeConfig,
    JudgeProvider,
    MetricName,
)
from ragcheck.connectors.custom import from_dicts, from_json, from_csv, load as load_dataset

__version__ = "0.1.0"
__all__ = [
    "evaluate",
    "evaluate_dataset",
    "EvalConfig",
    "EvalDataset",
    "EvalReport",
    "EvalSample",
    "JudgeConfig",
    "JudgeProvider",
    "MetricName",
    "ALL_METRICS",
    "from_dicts",
    "from_json",
    "from_csv",
    "load_dataset",
]


def evaluate(
    questions: list[str],
    contexts: list[list[str]],
    answers: list[str],
    ground_truths: list[str | None] | None = None,
    metrics: list[str] | list[MetricName] | None = None,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "litellm",
    api_key: str | None = None,
    api_base: str | None = None,
    concurrency: int = 4,
    fail_threshold: float | None = None,
) -> EvalReport:
    """
    Evaluate a RAG pipeline. Synchronous convenience wrapper around Pipeline.

    Args:
        questions:       List of user questions.
        contexts:        List of retrieved context chunks per question (list of lists).
        answers:         List of generated answers.
        ground_truths:   Optional list of reference answers (needed for context_recall).
        metrics:         Metrics to evaluate. Defaults to [faithfulness, context_relevance, answer_relevance].
                         Use "all" as a single-element list for all 6 metrics.
        judge_model:     Model string (e.g. "gpt-4o-mini", "anthropic/claude-3-haiku-20240307", "ollama/llama3").
        judge_provider:  Provider type: "litellm" | "openai" | "anthropic" | "local".
        api_key:         API key (falls back to environment variable).
        api_base:        Custom API base URL for local deployments.
        concurrency:     Max concurrent judge calls.
        fail_threshold:  If set, report.passed=False when overall_score < threshold.

    Returns:
        EvalReport with per-sample and aggregate scores.

    Example:
        results = evaluate(
            questions=["What is RAG?"],
            contexts=[["RAG stands for Retrieval-Augmented Generation..."]],
            answers=["RAG combines retrieval with generation to improve accuracy."],
            judge_model="gpt-4o-mini",
        )
        print(results.summary())
    """
    if not (len(questions) == len(contexts) == len(answers)):
        raise ValueError("questions, contexts, and answers must all have the same length")

    # Resolve metrics
    resolved_metrics = _resolve_metrics(metrics)

    # Build samples
    samples = [
        EvalSample(
            question=q,
            contexts=c,
            answer=a,
            ground_truth=(ground_truths[i] if ground_truths else None),
        )
        for i, (q, c, a) in enumerate(zip(questions, contexts, answers))
    ]
    dataset = EvalDataset(samples=samples)

    config = EvalConfig(
        metrics=resolved_metrics,
        judge=JudgeConfig(
            provider=JudgeProvider(judge_provider),
            model=judge_model,
            api_key=api_key,
            api_base=api_base,
        ),
        concurrency=concurrency,
        fail_threshold=fail_threshold,
    )

    return asyncio.run(evaluate_dataset(dataset, config))


async def evaluate_dataset(dataset: EvalDataset, config: EvalConfig) -> EvalReport:
    """
    Async evaluation of a full EvalDataset.
    Use this when you're already in an async context.
    """
    pipeline = Pipeline(config)
    return await pipeline.run(dataset)


def _resolve_metrics(metrics: list[str] | list[MetricName] | None) -> list[MetricName]:
    if metrics is None:
        return [MetricName.CONTEXT_RELEVANCE, MetricName.FAITHFULNESS, MetricName.ANSWER_RELEVANCE]
    if metrics == ["all"]:
        return ALL_METRICS
    resolved = []
    for m in metrics:
        if isinstance(m, MetricName):
            resolved.append(m)
        else:
            resolved.append(MetricName(m))
    return resolved
