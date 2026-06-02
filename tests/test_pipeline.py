"""Tests for the Pipeline orchestrator."""
from unittest.mock import patch

import pytest

from ragcheck.core.pipeline import Pipeline
from ragcheck.core.schema import (
    EvalConfig,
    EvalDataset,
    EvalSample,
    JudgeConfig,
    MetricName,
)


@pytest.fixture
def simple_dataset():
    return EvalDataset(
        name="test_dataset",
        samples=[
            EvalSample(
                question="What is RAG?",
                contexts=["RAG stands for Retrieval-Augmented Generation."],
                answer="RAG is a retrieval-augmented generation technique.",
                ground_truth="RAG combines retrieval with generation.",
            ),
            EvalSample(
                question="What is a vector database?",
                contexts=["A vector database stores embeddings for similarity search."],
                answer="A vector database stores and queries vector embeddings.",
            ),
        ],
    )


@pytest.fixture
def config_with_mock():
    return EvalConfig(
        metrics=[MetricName.CONTEXT_RELEVANCE, MetricName.FAITHFULNESS],
        judge=JudgeConfig(model="mock-model"),
        concurrency=2,
    )


class TestPipeline:
    async def test_run_returns_report(self, simple_dataset, config_with_mock, mock_judge):
        with patch("ragcheck.core.pipeline.build_judge", return_value=mock_judge):
            pipeline = Pipeline(config_with_mock)
            report = await pipeline.run(simple_dataset)

        assert report.dataset_name == "test_dataset"
        assert len(report.results) == 2
        assert len(report.aggregate_stats) == 2
        assert 0.0 <= report.overall_score <= 1.0

    async def test_progress_callback_called(self, simple_dataset, config_with_mock, mock_judge):
        progress_calls = []

        def on_progress(done, total):
            progress_calls.append((done, total))

        with patch("ragcheck.core.pipeline.build_judge", return_value=mock_judge):
            pipeline = Pipeline(config_with_mock, progress_callback=on_progress)
            await pipeline.run(simple_dataset)

        assert len(progress_calls) == len(simple_dataset.samples)
        assert progress_calls[-1][0] == len(simple_dataset.samples)

    async def test_fail_threshold_passed(self, simple_dataset, config_with_mock, mock_judge):
        config_with_mock.fail_threshold = 0.5  # mock returns 0.85, should pass
        with patch("ragcheck.core.pipeline.build_judge", return_value=mock_judge):
            pipeline = Pipeline(config_with_mock)
            report = await pipeline.run(simple_dataset)
        assert report.passed is True

    async def test_fail_threshold_failed(self, simple_dataset, config_with_mock, mock_judge_low):
        config_with_mock.fail_threshold = 0.9  # mock_low returns 0.2, should fail
        with patch("ragcheck.core.pipeline.build_judge", return_value=mock_judge_low):
            pipeline = Pipeline(config_with_mock)
            report = await pipeline.run(simple_dataset)
        assert report.passed is False
