"""
Local judge for Ollama / vLLM deployments.

Uses the OpenAI-compatible API that both Ollama and vLLM expose.
Default Ollama base URL: http://localhost:11434/v1
Default vLLM base URL: http://localhost:8000/v1
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from ragcheck.core.judges.base import BaseJudge, JudgeError
from ragcheck.core.schema import JudgeConfig

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"


class LocalJudge(BaseJudge):
    """
    Judge backend for locally running models via OpenAI-compatible API.

    Examples:
        # Ollama
        config = JudgeConfig(provider="local", model="llama3", api_base="http://localhost:11434/v1")

        # vLLM
        config = JudgeConfig(provider="local", model="meta-llama/Meta-Llama-3-8B", api_base="http://localhost:8000/v1")
    """

    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)
        self._base = (config.api_base or DEFAULT_OLLAMA_BASE).rstrip("/")

    async def judge(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert RAG evaluation judge. "
                        "Always respond with valid JSON only. "
                        "Your entire response must be a single JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            for attempt in range(1, self.config.max_retries + 2):
                try:
                    resp = await client.post(
                        f"{self._base}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    content = body["choices"][0]["message"]["content"]
                    data = self.extract_json(content)
                    return self.validate_score(data)
                except JudgeError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        "Local judge attempt %d/%d failed: %s",
                        attempt,
                        self.config.max_retries + 1,
                        exc,
                    )
                    if attempt <= self.config.max_retries:
                        await asyncio.sleep(wait)

        raise JudgeError(
            f"Local judge failed after {self.config.max_retries + 1} attempts"
        ) from last_exc
