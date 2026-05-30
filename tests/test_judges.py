"""Tests for judge base class utilities."""
import pytest

from ragcheck.core.judges.base import BaseJudge, JudgeError


class TestJudgeHelpers:
    def test_extract_json_plain(self):
        text = '{"score": 0.8, "reasoning": "good"}'
        result = BaseJudge.extract_json(text)
        assert result["score"] == 0.8

    def test_extract_json_fenced(self):
        text = '```json\n{"score": 0.75, "reasoning": "ok"}\n```'
        result = BaseJudge.extract_json(text)
        assert result["score"] == 0.75

    def test_extract_json_embedded(self):
        text = 'Here is my evaluation: {"score": 0.9, "reasoning": "excellent"} Based on...'
        result = BaseJudge.extract_json(text)
        assert result["score"] == 0.9

    def test_extract_json_invalid_raises(self):
        with pytest.raises(JudgeError, match="Could not extract JSON"):
            BaseJudge.extract_json("This is just plain text with no JSON")

    def test_validate_score_clamps_above_1(self):
        data = {"score": 1.5, "reasoning": "test"}
        result = BaseJudge.validate_score(data)
        assert result["score"] == 1.0

    def test_validate_score_clamps_below_0(self):
        data = {"score": -0.3}
        result = BaseJudge.validate_score(data)
        assert result["score"] == 0.0

    def test_validate_score_missing_key_raises(self):
        with pytest.raises(JudgeError, match="missing 'score' key"):
            BaseJudge.validate_score({"reasoning": "no score here"})

    def test_validate_score_non_numeric_raises(self):
        with pytest.raises(JudgeError, match="not numeric"):
            BaseJudge.validate_score({"score": "high"})
