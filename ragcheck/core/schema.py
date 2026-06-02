"""
Core Pydantic v2 schemas for RAGcheck input/output contracts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MetricName(str, Enum):
    CONTEXT_RELEVANCE = "context_relevance"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RECALL = "context_recall"
    NOISE_SENSITIVITY = "noise_sensitivity"
    CHUNK_UTILIZATION = "chunk_utilization"


ALL_METRICS: list[MetricName] = list(MetricName)


class JudgeProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"   # Ollama / vLLM
    LITELLM = "litellm"  # any LiteLLM-supported model


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    """A single RAG evaluation sample."""

    question: str = Field(..., description="The user question / query.")
    contexts: list[str] = Field(
        ...,
        description="List of retrieved context chunks passed to the LLM.",
        min_length=1,
    )
    answer: str = Field(..., description="The generated answer from the RAG pipeline.")
    ground_truth: str | None = Field(
        None,
        description=(
            "Reference / gold answer. Required for context_recall metric. "
            "Optional for other metrics."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (e.g. document IDs, retriever config).",
    )


class EvalDataset(BaseModel):
    """A collection of evaluation samples."""

    samples: list[EvalSample] = Field(..., min_length=1)
    name: str | None = Field(None, description="Optional dataset / experiment name.")



class JudgeConfig(BaseModel):
    """Configuration for the LLM judge backend."""

    provider: JudgeProvider = JudgeProvider.LITELLM
    model: str = Field(
        "gpt-4o-mini",
        description=(
            "Model identifier. For LiteLLM, use provider/model format e.g. "
            "'openai/gpt-4o', 'anthropic/claude-haiku-4-5', "
            "'ollama/llama3'."
        ),
    )
    api_key: str | None = Field(None, description="API key (falls back to env var).")
    api_base: str | None = Field(
        None,
        description="Custom API base URL (for local Ollama/vLLM deployments).",
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=64, le=8192)
    timeout: float = Field(60.0, description="Request timeout in seconds.")
    max_retries: int = Field(3, ge=0, le=10)


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    metrics: list[MetricName] = Field(
        default_factory=lambda: [
            MetricName.CONTEXT_RELEVANCE,
            MetricName.FAITHFULNESS,
            MetricName.ANSWER_RELEVANCE,
        ]
    )
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    concurrency: int = Field(
        4,
        ge=1,
        le=20,
        description="Max concurrent judge calls per evaluator.",
    )
    fail_threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="If set, aggregate score below this triggers a non-zero exit code.",
    )


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class MetricScore(BaseModel):
    """Score for a single metric on a single sample."""

    metric: MetricName
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None  # populated if judge call failed


class SampleResult(BaseModel):
    """Evaluation results for a single sample."""

    sample: EvalSample
    scores: list[MetricScore]

    def score_for(self, metric: MetricName) -> MetricScore | None:
        return next((s for s in self.scores if s.metric == metric), None)

    @property
    def aggregate(self) -> float:
        valid = [s.score for s in self.scores if s.error is None]
        return round(sum(valid) / len(valid), 4) if valid else 0.0


class AggregateStats(BaseModel):
    """Aggregate statistics across all samples for one metric."""

    metric: MetricName
    mean: float
    min: float
    max: float
    std: float
    failed_samples: int = 0


class EvalReport(BaseModel):
    """Full evaluation report."""

    dataset_name: str | None = None
    config: EvalConfig
    results: list[SampleResult]
    aggregate_stats: list[AggregateStats] = Field(default_factory=list)
    overall_score: float = 0.0
    passed: bool | None = None  # None = no threshold set

    def summary(self) -> str:
        """Return a plain-text summary of results."""
        lines = []
        if self.dataset_name:
            lines.append(f"Dataset: {self.dataset_name}")
        lines.append(f"Samples evaluated: {len(self.results)}")
        lines.append(f"Overall score: {self.overall_score:.3f}")
        if self.passed is not None:
            status = "PASSED" if self.passed else "FAILED"
            lines.append(f"Threshold check: {status}")
        lines.append("")
        lines.append("Per-metric averages:")
        for stat in self.aggregate_stats:
            lines.append(
                f"  {stat.metric.value:<22} mean={stat.mean:.3f}  "
                f"min={stat.min:.3f}  max={stat.max:.3f}"
            )
        return "\n".join(lines)
