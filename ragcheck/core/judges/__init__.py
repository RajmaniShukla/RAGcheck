"""Judge factory and exports."""
from __future__ import annotations

from ragcheck.core.judges.anthropic_judge import AnthropicJudge
from ragcheck.core.judges.base import BaseJudge, JudgeError
from ragcheck.core.judges.litellm_judge import LiteLLMJudge
from ragcheck.core.judges.local_judge import LocalJudge
from ragcheck.core.judges.openai_judge import OpenAIJudge
from ragcheck.core.schema import JudgeConfig, JudgeProvider


def build_judge(config: JudgeConfig) -> BaseJudge:
    """Factory: create the right judge from config."""
    if config.provider == JudgeProvider.OPENAI:
        return OpenAIJudge(config)
    elif config.provider == JudgeProvider.ANTHROPIC:
        return AnthropicJudge(config)
    elif config.provider == JudgeProvider.LOCAL:
        return LocalJudge(config)
    else:
        # Default: LiteLLM handles everything
        return LiteLLMJudge(config)


__all__ = [
    "BaseJudge",
    "JudgeError",
    "LiteLLMJudge",
    "OpenAIJudge",
    "AnthropicJudge",
    "LocalJudge",
    "build_judge",
]
