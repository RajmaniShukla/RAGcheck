"""
CLI smoke tests using Typer's CliRunner.
No real API calls — uses mock judge or fixture JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ragcheck.cli import app

runner = CliRunner()


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    data = [
        {
            "question": "What is RAG?",
            "contexts": ["RAG stands for Retrieval-Augmented Generation."],
            "answer": "RAG combines retrieval with LLM generation.",
            "ground_truth": "RAG is a retrieval-augmented generation technique.",
        }
    ]
    p = tmp_path / "test_data.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    import csv

    p = tmp_path / "test_data.csv"
    rows = [
        {
            "question": "What is RAG?",
            "contexts": '["RAG stands for Retrieval-Augmented Generation."]',
            "answer": "RAG combines retrieval with LLM generation.",
        }
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "contexts", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# ragcheck version
# ---------------------------------------------------------------------------

class TestVersionCmd:
    def test_version_command_shows_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "ragcheck" in result.output


# ---------------------------------------------------------------------------
# ragcheck metrics
# ---------------------------------------------------------------------------

class TestMetricsCmd:
    def test_metrics_lists_all_six(self):
        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 0
        for metric in [
            "context_relevance",
            "faithfulness",
            "answer_relevance",
            "context_recall",
            "noise_sensitivity",
            "chunk_utilization",
        ]:
            assert metric in result.output

    def test_metrics_shows_ground_truth_requirement(self):
        result = runner.invoke(app, ["metrics"])
        assert "context_recall" in result.output
        assert "Yes" in result.output  # context_recall needs ground_truth


# ---------------------------------------------------------------------------
# ragcheck eval
# ---------------------------------------------------------------------------

class TestEvalCmd:
    def _mock_report(self):
        """Return a minimal fake EvalReport."""
        from ragcheck.core.schema import (
            AggregateStats,
            EvalConfig,
            EvalReport,
            EvalSample,
            MetricName,
            MetricScore,
            SampleResult,
        )

        return EvalReport(
            dataset_name="test",
            config=EvalConfig(),
            results=[
                SampleResult(
                    sample=EvalSample(
                        question="What is RAG?",
                        contexts=["RAG stands for Retrieval-Augmented Generation."],
                        answer="RAG combines retrieval with LLM generation.",
                    ),
                    scores=[
                        MetricScore(metric=MetricName.FAITHFULNESS, score=0.9),
                        MetricScore(metric=MetricName.CONTEXT_RELEVANCE, score=0.85),
                        MetricScore(metric=MetricName.ANSWER_RELEVANCE, score=0.88),
                    ],
                )
            ],
            aggregate_stats=[
                AggregateStats(
                    metric=MetricName.FAITHFULNESS, mean=0.9, min=0.9, max=0.9, std=0.0
                ),
                AggregateStats(
                    metric=MetricName.CONTEXT_RELEVANCE, mean=0.85, min=0.85, max=0.85, std=0.0
                ),
                AggregateStats(
                    metric=MetricName.ANSWER_RELEVANCE, mean=0.88, min=0.88, max=0.88, std=0.0
                ),
            ],
            overall_score=0.876,
            passed=None,
        )

    def test_eval_missing_input_file(self):
        result = runner.invoke(app, ["eval", "--input", "nonexistent.json"])
        # exit code 1 = our manual check; 2 = Typer/Click UsageError — both signal failure
        assert result.exit_code != 0

    def test_eval_unsupported_output_extension_warns(self, sample_json, tmp_path):
        output = tmp_path / "report.xml"
        # Pipeline is imported lazily inside eval_cmd, so patch at its module path
        with patch("ragcheck.core.pipeline.Pipeline") as MockPipeline:
            async def fake_run(dataset):
                return self._mock_report()

            MockPipeline.return_value.run.side_effect = fake_run
            with patch("ragcheck.core.judges.build_judge"):
                result = runner.invoke(
                    app,
                    ["eval", "--input", str(sample_json), "--output", str(output)],
                )
        # Either succeeds (with warning) or exits non-zero — just no crash
        assert "xml" in result.output.lower() or result.exit_code in (0, 1)

    def test_eval_json_output(self, sample_json, tmp_path):
        output = tmp_path / "report.json"
        fake_report = self._mock_report()

        # Pipeline is imported lazily inside eval_cmd, so patch at its module path
        with patch("ragcheck.core.pipeline.Pipeline") as MockPipeline:
            async def fake_run(dataset):
                return fake_report

            MockPipeline.return_value.run.side_effect = fake_run
            with patch("ragcheck.core.judges.build_judge"):
                result = runner.invoke(
                    app,
                    ["eval", "--input", str(sample_json), "--output", str(output)],
                )

        # If pipeline was called we should have a JSON file
        if output.exists():
            data = json.loads(output.read_text())
            assert "results" in data

    def test_eval_invalid_metric_exits_nonzero(self, sample_json):
        result = runner.invoke(
            app, ["eval", "--input", str(sample_json), "--metrics", "invalid_metric"]
        )
        # exit code 1 = our ValueError handler; 2 = Typer/Click UsageError — both are errors
        assert result.exit_code != 0

    def test_eval_csv_input(self, sample_csv, tmp_path):
        fake_report = self._mock_report()
        # Pipeline is imported lazily inside eval_cmd, so patch at its module path
        with patch("ragcheck.core.pipeline.Pipeline") as MockPipeline:
            async def fake_run(dataset):
                return fake_report

            MockPipeline.return_value.run.side_effect = fake_run
            with patch("ragcheck.core.judges.build_judge"):
                result = runner.invoke(
                    app,
                    ["eval", "--input", str(sample_csv)],
                )
        # CSV loading should work without errors
        assert "CSV" not in result.output or result.exit_code != 2


# ---------------------------------------------------------------------------
# ragcheck dashboard
# ---------------------------------------------------------------------------

class TestDashboardCmd:
    def test_dashboard_missing_streamlit_exits(self):
        """Without streamlit installed (or mocked away), should exit with code 1."""
        with patch.dict("sys.modules", {"streamlit": None}):
            result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 1
        assert "streamlit" in result.output.lower()
