"""
Native OpenAI judge (direct openai SDK, no LiteLLM dependency).
Use this if you want a lean setup with only OpenAI models.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ragcheck.core.judges.base import BaseJudge, JudgeError
from ragcheck.core.schema import JudgeConfig

logger = logging.getLogger(__name__)

_OPENAI_AVAILABLE = False
try:
    from openai import AsyncOpenAI  # type: ignore[import]
    _OPENAI_AVAILABLE = True
except ImportError:
    pass


class OpenAIJudge(BaseJudge):
    """
    Judge backend using the native OpenAI async client.
    Falls back to OPENAI_API_KEY env var when config.api_key is None.
    """

    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for OpenAIJudge. "
                "Install it with: pip install openai"
            )
        self._client = AsyncOpenAI(
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=config.api_base,
            timeout=config.timeout,
            max_retries=0,  # we handle retries ourselves
        )

    async def judge(self, prompt: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert RAG evaluation judge. "
                                "Always respond with valid JSON only — no prose, no markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                data = self.extract_json(content)
                return self.validate_score(data)
            except JudgeError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "OpenAI judge attempt %d/%d failed: %s", attempt, self.config.max_retries + 1, exc
                )
                if attempt <= self.config.max_retries:
                    await asyncio.sleep(wait)

        raise JudgeError(
            f"OpenAI judge failed after {self.config.max_retries + 1} attempts"
        ) from last_exc
