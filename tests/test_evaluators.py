"""Tests for all 6 evaluators using mock judges."""
import pytest

from ragcheck.core.evaluators import (
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevanceEvaluator,
    ContextRecallEvaluator,
    NoiseSensitivityEvaluator,
    ChunkUtilizationEvaluator,
)
from ragcheck.core.schema import MetricName


class TestContextRelevanceEvaluator:
    async def test_evaluate_returns_score(self, sample_rag, mock_judge):
        ev = ContextRelevanceEvaluator(mock_judge)
        result = await ev.evaluate(sample_rag)
        assert result.metric == MetricName.CONTEXT_RELEVANCE
        assert 0.0 <= result.score <= 1.0
        assert result.error is None

    async def test_evaluate_low_score(self, sample_rag, mock_judge_low):
        ev = ContextRelevanceEvaluator(mock_judge_low)
        result = await ev.evaluate(sample_rag)
        assert result.score < 0.5


class TestFaithfulnessEvaluator:
    async def test_evaluate_faithful_answer(self, sample_rag, mock_judge):
        ev = FaithfulnessEvaluator(mock_judge)
        result = await ev.evaluate(sample_rag)
        assert result.metric == MetricName.FAITHFULNESS
        assert result.score >= 0.8

    async def test_evaluate_hallucinated_answer(self, sample_with_hallucination, mock_judge_low):
        ev = FaithfulnessEvaluator(mock_judge_low)
        result = await ev.evaluate(sample_with_hallucination)
        assert result.score < 0.5
        assert "hallucinated claim" in result.details.get("unsupported_claims", [])


class TestAnswerRelevanceEvaluator:
    async def test_evaluate_returns_score(self, sample_rag, mock_judge):
        ev = AnswerRelevanceEvaluator(mock_judge)
        result = await ev.evaluate(sample_rag)
        assert result.metric == MetricName.ANSWER_RELEVANCE
        assert result.error is None


class TestContextRecallEvaluator:
    async def test_skips_without_ground_truth(self, mock_judge):
        from ragcheck.core.schema import EvalSample
        sample = EvalSample(
            question="Q", contexts=["C"], answer="A"
        )
        ev = ContextRecallEvaluator(mock_judge)
        result = await ev.evaluate(sample)
        assert result.error is not None
        assert "ground_truth" in result.error.lower()

    async def test_evaluates_with_ground_truth(self, sample_rag, mock_judge):
        ev = ContextRecallEvaluator(mock_judge)
        result = await ev.evaluate(sample_rag)
        assert result.metric == MetricName.CONTEXT_RECALL
        assert result.error is None


class TestNoiseSensitivityEvaluator:
    async def test_evaluate_returns_robustness_score(self, sample_rag, mock_judge):
        ev = NoiseSensitivityEvaluator(mock_judge)
        result = await ev.evaluate(sample_rag)
        assert result.metric == MetricName.NOISE_SENSITIVITY
        assert 0.0 <= result.score <= 1.0
        assert "baseline_faithfulness" in result.details
        assert "noisy_faithfulness" in result.details


class TestChunkUtilizationEvaluator:
    async def test_evaluate_returns_utilization(self, sample_rag, mock_judge):
        ev = ChunkUtilizationEvaluator(mock_judge)
        result = await ev.evaluate(sample_rag)
        assert result.metric == MetricName.CHUNK_UTILIZATION
        assert 0.0 <= result.score <= 1.0
        assert "utilized_chunks" in result.details
