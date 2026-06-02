"""Tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError

from ragcheck.core.schema import (
    AggregateStats,
    EvalConfig,
    EvalReport,
    EvalSample,
    MetricName,
    MetricScore,
    SampleResult,
)


class TestEvalSample:
    def test_valid_sample(self):
        s = EvalSample(
            question="What is RAG?",
            contexts=["Context chunk 1", "Context chunk 2"],
            answer="RAG is...",
        )
        assert s.question == "What is RAG?"
        assert len(s.contexts) == 2
        assert s.ground_truth is None

    def test_sample_requires_contexts(self):
        with pytest.raises(ValidationError):
            EvalSample(question="Q", contexts=[], answer="A")

    def test_sample_with_ground_truth(self):
        s = EvalSample(
            question="Q", contexts=["C"], answer="A", ground_truth="GT"
        )
        assert s.ground_truth == "GT"


class TestMetricScore:
    def test_score_clamped(self):
        # Scores are validated in the judge, but schema accepts 0-1
        s = MetricScore(metric=MetricName.FAITHFULNESS, score=0.85)
        assert s.score == 0.85

    def test_error_score(self):
        s = MetricScore(
            metric=MetricName.CONTEXT_RELEVANCE, score=0.0, error="judge timeout"
        )
        assert s.error == "judge timeout"


class TestSampleResult:
    def test_aggregate(self, sample_rag, mock_judge):
        scores = [
            MetricScore(metric=MetricName.FAITHFULNESS, score=0.8),
            MetricScore(metric=MetricName.CONTEXT_RELEVANCE, score=0.9),
        ]
        result = SampleResult(sample=sample_rag, scores=scores)
        assert result.aggregate == pytest.approx(0.85, abs=0.001)

    def test_aggregate_excludes_errors(self, sample_rag):
        scores = [
            MetricScore(metric=MetricName.FAITHFULNESS, score=0.8),
            MetricScore(metric=MetricName.CONTEXT_RELEVANCE, score=0.0, error="failed"),
        ]
        result = SampleResult(sample=sample_rag, scores=scores)
        assert result.aggregate == 0.8  # only non-error scores


class TestEvalConfig:
    def test_default_metrics(self):
        config = EvalConfig()
        assert MetricName.FAITHFULNESS in config.metrics
        assert MetricName.CONTEXT_RELEVANCE in config.metrics

    def test_all_metrics(self):
        config = EvalConfig(metrics=list(MetricName))
        assert len(config.metrics) == 6


class TestEvalReport:
    def test_summary(self):
        from ragcheck.core.schema import EvalConfig
        config = EvalConfig()
        report = EvalReport(
            dataset_name="test",
            config=config,
            results=[],
            aggregate_stats=[
                AggregateStats(
                    metric=MetricName.FAITHFULNESS,
                    mean=0.85,
                    min=0.7,
                    max=1.0,
                    std=0.1,
                )
            ],
            overall_score=0.85,
            passed=True,
        )
        summary = report.summary()
        assert "0.850" in summary
        assert "PASSED" in summary
