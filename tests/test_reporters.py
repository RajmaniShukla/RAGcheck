"""Tests for reporters (JSON, HTML)."""
import json

import pytest

from ragcheck.core.schema import (
    AggregateStats,
    EvalConfig,
    EvalReport,
    EvalSample,
    MetricName,
    MetricScore,
    SampleResult,
)
from ragcheck.reporters.html_report import generate_html, save_html
from ragcheck.reporters.json_export import load_json, save_json, to_dict


@pytest.fixture
def sample_report():
    config = EvalConfig(metrics=[MetricName.FAITHFULNESS, MetricName.CONTEXT_RELEVANCE])
    sample = EvalSample(question="Q?", contexts=["Context"], answer="Answer.")
    scores = [
        MetricScore(metric=MetricName.FAITHFULNESS, score=0.9, reasoning="Good"),
        MetricScore(metric=MetricName.CONTEXT_RELEVANCE, score=0.8),
    ]
    result = SampleResult(sample=sample, scores=scores)
    return EvalReport(
        dataset_name="test_report",
        config=config,
        results=[result],
        aggregate_stats=[
            AggregateStats(metric=MetricName.FAITHFULNESS, mean=0.9, min=0.9, max=0.9, std=0.0),
            AggregateStats(metric=MetricName.CONTEXT_RELEVANCE, mean=0.8, min=0.8, max=0.8, std=0.0),
        ],
        overall_score=0.85,
        passed=True,
    )


class TestJsonExporter:
    def test_to_dict(self, sample_report):
        d = to_dict(sample_report)
        assert d["dataset_name"] == "test_report"
        assert d["overall_score"] == 0.85
        assert len(d["results"]) == 1

    def test_save_and_load(self, sample_report, tmp_path):
        p = tmp_path / "report.json"
        save_json(sample_report, p)
        assert p.exists()

        loaded = load_json(p)
        assert loaded.overall_score == sample_report.overall_score
        assert loaded.dataset_name == sample_report.dataset_name
        assert len(loaded.results) == len(sample_report.results)

    def test_json_is_valid(self, sample_report, tmp_path):
        p = tmp_path / "report.json"
        save_json(sample_report, p)
        with p.open() as f:
            data = json.load(f)
        assert "results" in data


class TestHtmlReporter:
    def test_generate_html_contains_key_content(self, sample_report):
        html = generate_html(sample_report)
        assert "test_report" in html
        assert "0.850" in html
        assert "faithfulness" in html.lower()
        assert "context_relevance" in html.lower()

    def test_generate_html_is_valid_html(self, sample_report):
        html = generate_html(sample_report)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_save_html(self, sample_report, tmp_path):
        p = tmp_path / "report.html"
        result = save_html(sample_report, p)
        assert result == p
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "RAGcheck" in content
