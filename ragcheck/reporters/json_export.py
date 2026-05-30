"""
JSON reporter — export full EvalReport to JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from ragcheck.core.schema import EvalReport


def to_dict(report: EvalReport) -> dict:
    """Convert EvalReport to a plain dict (JSON-serializable)."""
    return report.model_dump(mode="json")


def save_json(report: EvalReport, path: str | Path) -> Path:
    """Save EvalReport as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = to_dict(report)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return p


def load_json(path: str | Path) -> EvalReport:
    """Load a saved EvalReport from JSON."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return EvalReport.model_validate(data)
