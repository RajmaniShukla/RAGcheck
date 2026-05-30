"""
Native Anthropic judge (direct anthropic SDK).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ragcheck.core.judges.base import BaseJudge, JudgeError
from ragcheck.core.schema import JudgeConfig

logger = logging.getLogger(__name__)

_ANTHROPIC_AVAILABLE = False
try:
    import anthropic  # type: ignore[import]
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    pass


# Current Anthropic model names (as of 2026)
# Haiku  (fast, cheap):  claude-haiku-4-5
# Sonnet (balanced):     claude-sonnet-4-5
# Opus   (powerful):     claude-opus-4-5
# Note: old names like 'claude-3-haiku-20240307' are deprecated/removed.
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicJudge(BaseJudge):
    """
    Judge backend using the native Anthropic async client.
    Falls back to ANTHROPIC_API_KEY env var when config.api_key is None.

    Recommended models:
        claude-haiku-4-5    — fast, cheap, great for bulk evaluation
        claude-sonnet-4-5   — balanced quality/cost
        claude-opus-4-5     — highest quality
    """

    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for AnthropicJudge. "
                "Install it with: pip install anthropic"
            )
        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=config.api_base,
            timeout=config.timeout,
            max_retries=0,
        )

    async def judge(self, prompt: str) -> dict[str, Any]:
        system = (
            "You are an expert RAG evaluation judge. "
            "Always respond with valid JSON only — no prose, no markdown fences. "
            "Your entire response must be a single JSON object."
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = await self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text if response.content else ""
                data = self.extract_json(content)
                return self.validate_score(data)
            except JudgeError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Anthropic judge attempt %d/%d failed: %s",
                    attempt,
                    self.config.max_retries + 1,
                    exc,
                )
                if attempt <= self.config.max_retries:
                    await asyncio.sleep(wait)

        raise JudgeError(
            f"Anthropic judge failed after {self.config.max_retries + 1} attempts"
        ) from last_exc
