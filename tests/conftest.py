"""Shared pytest fixtures for ragcheck tests."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragcheck.core.schema import (
    EvalConfig,
    EvalSample,
    JudgeConfig,
    MetricName,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rag() -> EvalSample:
    return EvalSample(
        question="What is RAG?",
        contexts=[
            "RAG stands for Retrieval-Augmented Generation.",
            "It was introduced by Facebook AI Research in 2020.",
        ],
        answer="RAG is a technique that combines retrieval with generation to improve accuracy.",
        ground_truth="RAG (Retrieval-Augmented Generation) combines document retrieval with LLM generation.",
    )


@pytest.fixture
def sample_with_hallucination() -> EvalSample:
    return EvalSample(
        question="What is the capital of France?",
        contexts=["France is a country in Western Europe. Its capital is Paris."],
        answer="The capital of France is Berlin, which is located in central Europe.",
    )


@pytest.fixture
def eval_config_minimal() -> EvalConfig:
    return EvalConfig(
        metrics=[MetricName.FAITHFULNESS, MetricName.CONTEXT_RELEVANCE],
        judge=JudgeConfig(model="mock-model"),
        concurrency=2,
    )


# ---------------------------------------------------------------------------
# Mock judge fixture
# ---------------------------------------------------------------------------

def make_mock_judge(scores: dict[str, float] | None = None) -> Any:
    """Create a mock judge that returns predictable scores."""
    default_scores = scores or {
        "score": 0.85,
        "reasoning": "Mock evaluation reasoning.",
        "relevant_chunks": [1, 2],
        "irrelevant_chunks": [],
        "supported_claims": ["claim1"],
        "unsupported_claims": [],
        "covered_claims": ["fact1"],
        "missing_claims": [],
        "utilized_chunks": [1],
        "ignored_chunks": [],
        "text": "This is a mock noise chunk.",
    }

    mock = MagicMock()
    mock.config = JudgeConfig(model="mock-model")

    async def async_judge(prompt: str) -> dict:
        return {**default_scores}

    mock.judge = async_judge
    return mock


@pytest.fixture
def mock_judge():
    return make_mock_judge()


@pytest.fixture
def mock_judge_low():
    """Mock judge that returns low scores (simulates bad pipeline)."""
    return make_mock_judge({
        "score": 0.2,
        "reasoning": "Low quality response.",
        "relevant_chunks": [],
        "irrelevant_chunks": [1, 2],
        "supported_claims": [],
        "unsupported_claims": ["hallucinated claim"],
        "covered_claims": [],
        "missing_claims": ["key fact"],
        "utilized_chunks": [],
        "ignored_chunks": [1, 2],
        "text": "Unrelated noise text.",
    })
