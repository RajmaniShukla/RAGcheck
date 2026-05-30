"""
Faithfulness Evaluator
-----------------------
Measures: Is the answer grounded in the retrieved context? (no hallucination)
Score: 0 (completely hallucinated) → 1 (fully supported by context)
"""
from __future__ import annotations

from ragcheck.core.evaluators.base import BaseEvaluator
from ragcheck.core.schema import EvalSample, MetricName, MetricScore

PROMPT_TEMPLATE = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system for hallucination.

## Question
{question}

## Retrieved Context (source of truth)
{contexts}

## Generated Answer
{answer}

## Task
Evaluate how faithful (grounded) the generated answer is to the retrieved context.
An answer is faithful if every factual claim it makes can be traced back to the context.
Penalize claims that are:
- Not mentioned in the context
- Contradicted by the context
- Extrapolated beyond what the context supports

## Response Format (JSON only, no other text)
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<brief explanation>",
  "supported_claims": ["<claim1>", "<claim2>"],
  "unsupported_claims": ["<claim that has no basis in context>"]
}}

Score guide:
1.0 = Every claim is fully supported by context
0.7 = Minor unsupported details, core answer is grounded
0.4 = Significant unsupported or extrapolated claims
0.1 = Most claims are hallucinated or contradicted
0.0 = Answer is entirely fabricated / contradicts context
"""


class FaithfulnessEvaluator(BaseEvaluator):
    metric = MetricName.FAITHFULNESS

    async def evaluate(self, sample: EvalSample) -> MetricScore:
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=sample.question,
                contexts=self._fmt_contexts(sample.contexts),
                answer=sample.answer,
            )
            result = await self.judge.judge(prompt)
            return MetricScore(
                metric=self.metric,
                score=result["score"],
                reasoning=result.get("reasoning", ""),
                details={
                    "supported_claims": result.get("supported_claims", []),
                    "unsupported_claims": result.get("unsupported_claims", []),
                },
            )
        except Exception as exc:
            return self._error_score(str(exc))
