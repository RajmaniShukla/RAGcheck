"""
Context Recall Evaluator
-------------------------
Measures: Did retrieval cover all the facts needed to answer the question?
Requires: ground_truth answer in the sample.
Score: 0 (critical facts missing from context) → 1 (all required facts present)
"""
from __future__ import annotations

from ragcheck.core.evaluators.base import BaseEvaluator
from ragcheck.core.schema import EvalSample, MetricName, MetricScore

PROMPT_TEMPLATE = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system's retrieval coverage.

## Question
{question}

## Ground Truth Answer (reference)
{ground_truth}

## Retrieved Context Chunks
{contexts}

## Task
Identify all factual claims/statements in the Ground Truth Answer.
Then check whether each claim is covered (supported) by the Retrieved Context.

## Response Format (JSON only, no other text)
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<brief explanation>",
  "covered_claims": ["<claim present in context>"],
  "missing_claims": ["<claim in ground truth but absent from context>"]
}}

Score = (# covered claims) / (# total claims in ground truth)
Score guide:
1.0 = Context covers every fact needed to produce the ground truth answer
0.7 = Most facts present, a few gaps
0.4 = Major gaps — retriever missed important facts
0.1 = Almost no relevant facts retrieved
0.0 = Context is completely missing all ground truth information
"""

NO_GROUND_TRUTH_MSG = (
    "context_recall requires a ground_truth answer in the sample. Skipped."
)


class ContextRecallEvaluator(BaseEvaluator):
    metric = MetricName.CONTEXT_RECALL

    async def evaluate(self, sample: EvalSample) -> MetricScore:
        if not sample.ground_truth:
            return MetricScore(
                metric=self.metric,
                score=0.0,
                reasoning=NO_GROUND_TRUTH_MSG,
                error=NO_GROUND_TRUTH_MSG,
            )
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=sample.question,
                ground_truth=sample.ground_truth,
                contexts=self._fmt_contexts(sample.contexts),
            )
            result = await self.judge.judge(prompt)
            return MetricScore(
                metric=self.metric,
                score=result["score"],
                reasoning=result.get("reasoning", ""),
                details={
                    "covered_claims": result.get("covered_claims", []),
                    "missing_claims": result.get("missing_claims", []),
                },
            )
        except Exception as exc:
            return self._error_score(str(exc))
