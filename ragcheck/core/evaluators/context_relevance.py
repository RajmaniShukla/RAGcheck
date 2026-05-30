"""
Context Relevance Evaluator
----------------------------
Measures: Are the retrieved chunks actually relevant to the question?
Score: 0 (completely irrelevant) → 1 (perfectly relevant)
"""
from __future__ import annotations

from ragcheck.core.evaluators.base import BaseEvaluator
from ragcheck.core.schema import EvalSample, MetricName, MetricScore

PROMPT_TEMPLATE = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system.

## Question
{question}

## Retrieved Context Chunks
{contexts}

## Task
Evaluate how relevant the retrieved context chunks are to the question.
Consider:
- Do the chunks contain information needed to answer the question?
- Are there chunks that are completely off-topic?
- What fraction of the retrieved content is actually useful?

## Response Format (JSON only, no other text)
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<brief explanation>",
  "relevant_chunks": [<list of 1-based chunk indices that are relevant>],
  "irrelevant_chunks": [<list of 1-based chunk indices that are irrelevant>]
}}

Score guide:
1.0 = All chunks are highly relevant
0.7 = Most chunks relevant, minor noise
0.4 = Mixed — some relevant, some off-topic
0.1 = Mostly irrelevant context retrieved
0.0 = Completely irrelevant context
"""


class ContextRelevanceEvaluator(BaseEvaluator):
    metric = MetricName.CONTEXT_RELEVANCE

    async def evaluate(self, sample: EvalSample) -> MetricScore:
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=sample.question,
                contexts=self._fmt_contexts(sample.contexts),
            )
            result = await self.judge.judge(prompt)
            return MetricScore(
                metric=self.metric,
                score=result["score"],
                reasoning=result.get("reasoning", ""),
                details={
                    "relevant_chunks": result.get("relevant_chunks", []),
                    "irrelevant_chunks": result.get("irrelevant_chunks", []),
                },
            )
        except Exception as exc:
            return self._error_score(str(exc))
