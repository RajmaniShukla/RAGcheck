"""
Chunk Utilization Evaluator
-----------------------------
Measures: Which retrieved chunks were actually used by the LLM to generate the answer?
Score: ratio of utilized chunks to total chunks (high = LLM used most of what was retrieved)

This surfaces retriever over-fetching — when you retrieve 10 chunks but the LLM only
uses 2, your retriever is returning too much noise.
"""
from __future__ import annotations

from ragcheck.core.evaluators.base import BaseEvaluator
from ragcheck.core.schema import EvalSample, MetricName, MetricScore

PROMPT_TEMPLATE = """\
You are analyzing a RAG (Retrieval-Augmented Generation) system.

## Question
{question}

## Retrieved Context Chunks
{contexts}

## Generated Answer
{answer}

## Task
For each context chunk, determine whether the generated answer actually used information
from that chunk to construct its response.

A chunk is "utilized" if:
- The answer directly quotes or paraphrases content from it
- The answer's facts/claims can be traced to that chunk

A chunk is "ignored" if:
- The answer contains no information derived from it

## Response Format (JSON only, no other text)
{{
  "score": <float: utilized_count / total_chunks>,
  "reasoning": "<brief explanation>",
  "utilized_chunks": [<1-based indices of chunks that were used>],
  "ignored_chunks": [<1-based indices of chunks that were not used>],
  "utilization_details": [
    {{"chunk": 1, "utilized": true, "reason": "<why it was/wasn't used>"}},
    ...
  ]
}}
"""


class ChunkUtilizationEvaluator(BaseEvaluator):
    metric = MetricName.CHUNK_UTILIZATION

    async def evaluate(self, sample: EvalSample) -> MetricScore:
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=sample.question,
                contexts=self._fmt_contexts(sample.contexts),
                answer=sample.answer,
            )
            result = await self.judge.judge(prompt)

            # Recompute score from chunk lists for reliability
            utilized = result.get("utilized_chunks", [])
            ignored = result.get("ignored_chunks", [])
            total = len(utilized) + len(ignored)
            computed_score = (len(utilized) / total) if total > 0 else result["score"]

            return MetricScore(
                metric=self.metric,
                score=round(computed_score, 4),
                reasoning=result.get("reasoning", ""),
                details={
                    "total_chunks": len(sample.contexts),
                    "utilized_chunks": utilized,
                    "ignored_chunks": ignored,
                    "utilization_details": result.get("utilization_details", []),
                },
            )
        except Exception as exc:
            return self._error_score(str(exc))
