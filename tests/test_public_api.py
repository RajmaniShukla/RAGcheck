"""
Tests for the top-level public API (ragcheck.evaluate / evaluate_dataset).
Uses mock judges so no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import ragcheck
from ragcheck import evaluate, evaluate_dataset
from ragcheck.core.schema import (
    EvalConfig,
    EvalDataset,
    EvalSample,
    JudgeConfig,
    JudgeProvider,
    MetricName,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_inputs():
    return {
        "questions": ["What is RAG?"],
        "contexts": [["RAG stands for Retrieval-Augmented Generation."]],
        "answers": ["RAG combines retrieval with LLM generation."],
    }


@pytest.fixture
def multi_inputs():
    return {
        "questions": ["What is RAG?", "What is a vector DB?"],
        "contexts": [
            ["RAG stands for Retrieval-Augmented Generation."],
            ["Vector databases store embeddings for similarity search."],
        ],
        "answers": [
            "RAG combines retrieval with LLM generation.",
            "A vector DB stores and queries vector embeddings.",
        ],
    }


# ---------------------------------------------------------------------------
# evaluate() — synchronous public API
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_basic_call_returns_report(self, minimal_inputs, mock_judge):
        with patch("ragcheck.Pipeline") as MockPipeline:
            from ragcheck.core.schema import (
                AggregateStats,
                EvalReport,
                EvalSample,
                MetricName,
                MetricScore,
                SampleResult,
            )
            fake_report = EvalReport(
                config=EvalConfig(),
                results=[
                    SampleResult(
                        sample=EvalSample(
                            question="What is RAG?",
                            contexts=["RAG stands for Retrieval-Augmented Generation."],
                            answer="RAG combines retrieval with LLM generation.",
                        ),
                        scores=[MetricScore(metric=MetricName.FAITHFULNESS, score=0.9)],
                    )
                ],
                aggregate_stats=[
                    AggregateStats(
                        metric=MetricName.FAITHFULNESS, mean=0.9, min=0.9, max=0.9, std=0.0
                    )
                ],
                overall_score=0.9,
            )
            import asyncio

            async def fake_run(dataset):
                return fake_report

            mock_instance = MockPipeline.return_value
            mock_instance.run.side_effect = fake_run

            report = evaluate(**minimal_inputs)

        assert report.overall_score == pytest.approx(0.9)
        assert len(report.results) == 1

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate(
                questions=["Q1", "Q2"],
                contexts=[["C1"]],  # length 1, not 2
                answers=["A1", "A2"],
            )

    def test_metrics_none_uses_defaults(self, minimal_inputs, mock_judge):
        with patch("ragcheck.Pipeline") as MockPipeline:
            from ragcheck.core.schema import EvalConfig, EvalReport

            async def fake_run(dataset):
                return EvalReport(config=EvalConfig(), results=[], overall_score=0.8)

            MockPipeline.return_value.run.side_effect = fake_run

            # Capture the config passed to Pipeline
            evaluate(**minimal_inputs, metrics=None)
            _, kwargs = MockPipeline.call_args
            config: EvalConfig = MockPipeline.call_args[0][0]

        assert MetricName.FAITHFULNESS in config.metrics
        assert MetricName.CONTEXT_RELEVANCE in config.metrics
        assert MetricName.ANSWER_RELEVANCE in config.metrics

    def test_metrics_all_uses_all_six(self, minimal_inputs):
        with patch("ragcheck.Pipeline") as MockPipeline:
            from ragcheck.core.schema import EvalConfig, EvalReport

            async def fake_run(dataset):
                return EvalReport(config=EvalConfig(), results=[], overall_score=0.5)

            MockPipeline.return_value.run.side_effect = fake_run
            evaluate(**minimal_inputs, metrics=["all"])
            config: EvalConfig = MockPipeline.call_args[0][0]

        assert len(config.metrics) == 6

    def test_invalid_metric_raises(self, minimal_inputs):
        with pytest.raises(ValueError):
            evaluate(**minimal_inputs, metrics=["not_a_real_metric"])

    def test_ground_truths_passed_through(self, minimal_inputs, mock_judge):
        with patch("ragcheck.Pipeline") as MockPipeline:
            from ragcheck.core.schema import EvalConfig, EvalReport

            captured_dataset = None

            async def fake_run(dataset):
                nonlocal captured_dataset
                captured_dataset = dataset
                return EvalReport(config=EvalConfig(), results=[], overall_score=0.8)

            MockPipeline.return_value.run.side_effect = fake_run
            evaluate(**minimal_inputs, ground_truths=["GT answer"])

        assert captured_dataset is not None
        assert captured_dataset.samples[0].ground_truth == "GT answer"


# ---------------------------------------------------------------------------
# evaluate_dataset() — async public API
# ---------------------------------------------------------------------------

class TestEvaluateDataset:
    async def test_async_entry_point(self, mock_judge):
        dataset = EvalDataset(
            samples=[
                EvalSample(
                    question="What is RAG?",
                    contexts=["RAG stands for Retrieval-Augmented Generation."],
                    answer="RAG combines retrieval with LLM generation.",
                )
            ]
        )
        config = EvalConfig(
            metrics=[MetricName.FAITHFULNESS],
            judge=JudgeConfig(model="mock-model"),
        )
        with patch("ragcheck.Pipeline") as MockPipeline:
            from ragcheck.core.schema import EvalReport

            async def fake_run(ds):
                return EvalReport(config=config, results=[], overall_score=0.85)

            MockPipeline.return_value.run.side_effect = fake_run
            report = await evaluate_dataset(dataset, config)

        assert report.overall_score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# _resolve_metrics() helper
# ---------------------------------------------------------------------------

class TestResolveMetrics:
    def test_none_returns_default_three(self):
        result = ragcheck._resolve_metrics(None)
        assert set(result) == {
            MetricName.CONTEXT_RELEVANCE,
            MetricName.FAITHFULNESS,
            MetricName.ANSWER_RELEVANCE,
        }

    def test_all_returns_all_six(self):
        result = ragcheck._resolve_metrics(["all"])
        assert len(result) == 6

    def test_string_metric_resolved(self):
        result = ragcheck._resolve_metrics(["faithfulness"])
        assert result == [MetricName.FAITHFULNESS]

    def test_mixed_strings_and_enum(self):
        result = ragcheck._resolve_metrics([MetricName.FAITHFULNESS, "context_relevance"])
        assert MetricName.FAITHFULNESS in result
        assert MetricName.CONTEXT_RELEVANCE in result

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            ragcheck._resolve_metrics(["invalid_metric_name"])
