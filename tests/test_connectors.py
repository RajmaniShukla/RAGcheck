"""Tests for connectors."""
import json
from pathlib import Path

import pytest

from ragcheck.connectors.custom import from_dicts, from_json, from_csv, load


class TestCustomConnector:
    def test_from_dicts_basic(self):
        data = [
            {
                "question": "What is RAG?",
                "contexts": ["Context 1", "Context 2"],
                "answer": "RAG is...",
            }
        ]
        dataset = from_dicts(data, name="test")
        assert dataset.name == "test"
        assert len(dataset.samples) == 1
        assert dataset.samples[0].question == "What is RAG?"
        assert len(dataset.samples[0].contexts) == 2

    def test_from_dicts_with_ground_truth(self):
        data = [
            {
                "question": "Q",
                "contexts": ["C"],
                "answer": "A",
                "ground_truth": "GT",
            }
        ]
        dataset = from_dicts(data)
        assert dataset.samples[0].ground_truth == "GT"

    def test_from_dicts_missing_field(self):
        with pytest.raises(ValueError, match="missing required fields"):
            from_dicts([{"question": "Q", "answer": "A"}])  # missing contexts

    def test_from_json(self, tmp_path):
        p = tmp_path / "test.json"
        data = [
            {"question": "Q1", "contexts": ["C1"], "answer": "A1"},
            {"question": "Q2", "contexts": ["C2"], "answer": "A2"},
        ]
        p.write_text(json.dumps(data), encoding="utf-8")
        dataset = from_json(p)
        assert len(dataset.samples) == 2
        assert dataset.name == "test"

    def test_from_json_dict_format(self, tmp_path):
        p = tmp_path / "data.json"
        data = {
            "name": "my_experiment",
            "samples": [{"question": "Q", "contexts": ["C"], "answer": "A"}],
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        dataset = from_json(p)
        assert dataset.name == "my_experiment"

    def test_from_csv(self, tmp_path):
        import csv
        p = tmp_path / "test.csv"
        rows = [
            {"question": "Q1", "contexts": '["C1", "C2"]', "answer": "A1"},
        ]
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "contexts", "answer"])
            writer.writeheader()
            writer.writerows(rows)
        dataset = from_csv(p)
        assert len(dataset.samples) == 1
        assert len(dataset.samples[0].contexts) == 2

    def test_load_auto_detect_json(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"question": "Q", "contexts": ["C"], "answer": "A"}]))
        dataset = load(p)
        assert len(dataset.samples) == 1

    def test_load_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "data.xml"
        p.write_text("<data/>")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load(p)
