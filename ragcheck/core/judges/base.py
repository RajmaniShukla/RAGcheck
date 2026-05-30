"""
Abstract base class for all LLM judges.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from ragcheck.core.schema import JudgeConfig


class JudgeError(Exception):
    """Raised when the judge fails to produce a valid response."""


class BaseJudge(ABC):
    """Abstract LLM-as-judge interface."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    @abstractmethod
    async def judge(self, prompt: str) -> dict[str, Any]:
        """
        Send a structured prompt to the LLM and return a parsed JSON dict.

        Implementations must:
        - Handle retries up to config.max_retries
        - Return a dict guaranteed to contain at least {"score": float}
        - Raise JudgeError on unrecoverable failure
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """
        Robustly extract the first JSON object from LLM output.

        Handles:
        - Plain JSON responses
        - JSON wrapped in ```json ... ``` fences
        - JSON embedded in longer prose
        """
        # Try direct parse first
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # Find first {...} block
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group())
            except json.JSONDecodeError:
                pass

        raise JudgeError(f"Could not extract JSON from judge response:\n{text[:500]}")

    @staticmethod
    def validate_score(data: dict[str, Any]) -> dict[str, Any]:
        """Ensure the parsed dict has a numeric score in [0, 1]."""
        if "score" not in data:
            raise JudgeError(f"Judge response missing 'score' key: {data}")
        try:
            score = float(data["score"])
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"Judge 'score' is not numeric: {data['score']}") from exc
        data["score"] = max(0.0, min(1.0, score))
        return data
