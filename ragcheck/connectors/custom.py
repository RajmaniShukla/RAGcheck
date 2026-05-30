"""
Custom connector — load EvalDataset from JSON, CSV, or Python dicts.
This is the primary connector for framework-agnostic use.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ragcheck.core.schema import EvalDataset, EvalSample


def from_dicts(
    data: list[dict[str, Any]],
    name: str | None = None,
) -> EvalDataset:
    """
    Build an EvalDataset from a list of dicts.

    Each dict must have: question, contexts (list[str]), answer.
    Optional: ground_truth, metadata.

    Example:
        dataset = from_dicts([
            {
                "question": "What is RAG?",
                "contexts": ["RAG stands for...", "It was introduced by..."],
                "answer": "RAG is a technique...",
                "ground_truth": "RAG stands for Retrieval-Augmented Generation..."
            }
        ])
    """
    samples = []
    for i, row in enumerate(data):
        try:
            # contexts can be a JSON string in CSV scenarios
            contexts = row["contexts"]
            if isinstance(contexts, str):
                try:
                    contexts = json.loads(contexts)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Row {i}: 'contexts' is not valid JSON: {contexts!r}"
                    ) from exc
            if not isinstance(contexts, list):
                raise ValueError(
                    f"Row {i}: 'contexts' must be a list of strings, got {type(contexts).__name__}"
                )
            # metadata can also be JSON-encoded in CSV
            metadata = row.get("metadata", {})
            if isinstance(metadata, str) and metadata:
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}  # ignore malformed metadata silently
            samples.append(
                EvalSample(
                    question=row["question"],
                    contexts=contexts,
                    answer=row["answer"],
                    ground_truth=row.get("ground_truth") or None,
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Row {i} is missing required fields: {exc}") from exc
    return EvalDataset(samples=samples, name=name)


def from_json(path: str | Path, name: str | None = None) -> EvalDataset:
    """
    Load from a JSON file.

    Supported formats:
    - List of sample dicts: [{"question": ..., "contexts": [...], "answer": ...}, ...]
    - {"samples": [...], "name": "..."} object
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return from_dicts(raw, name=name or p.stem)
    elif isinstance(raw, dict) and "samples" in raw:
        return from_dicts(raw["samples"], name=name or raw.get("name") or p.stem)
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")


def from_csv(path: str | Path, name: str | None = None) -> EvalDataset:
    """
    Load from a CSV file.

    Required columns: question, contexts (JSON-encoded list), answer
    Optional columns: ground_truth, metadata (JSON-encoded dict)
    """
    p = Path(path)
    rows = []
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return from_dicts(rows, name=name or p.stem)


def load(path: str | Path, name: str | None = None) -> EvalDataset:
    """Auto-detect format from file extension and load."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".json":
        return from_json(p, name=name)
    elif ext == ".csv":
        return from_csv(p, name=name)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .json or .csv")
