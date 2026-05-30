"""
LiteLLM judge — supports any provider (OpenAI, Anthropic, Ollama, Gemini, etc.)
through a single unified interface.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragcheck.core.judges.base import BaseJudge, JudgeError
from ragcheck.core.schema import JudgeConfig

logger = logging.getLogger(__name__)

_LITELLM_AVAILABLE = False
try:
    import litellm  # type: ignore[import]
    _LITELLM_AVAILABLE = True
except ImportError:
    pass


class LiteLLMJudge(BaseJudge):
    """
    Judge backend powered by LiteLLM.

    Supports all LiteLLM model strings:
    - "gpt-4o-mini"
    - "anthropic/claude-3-haiku-20240307"
    - "ollama/llama3"
    - "gemini/gemini-pro"
    """

    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)
        if not _LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for LiteLLMJudge. Install it with: pip install litellm"
            )
        # Disable litellm verbose logging unless debug mode
        litellm.set_verbose = False

    async def judge(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert RAG evaluation judge. "
                        "Always respond with valid JSON only — no prose, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = await litellm.acompletion(**kwargs)
                content = response.choices[0].message.content or ""
                data = self.extract_json(content)
                return self.validate_score(data)
            except JudgeError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "LiteLLM judge attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    self.config.max_retries + 1,
                    exc,
                    wait,
                )
                if attempt <= self.config.max_retries:
                    await asyncio.sleep(wait)

        raise JudgeError(
            f"LiteLLM judge failed after {self.config.max_retries + 1} attempts"
        ) from last_exc
