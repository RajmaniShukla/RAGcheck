"""
Answer Relevance Evaluator
---------------------------
Measures: Does the generated answer actually address the question asked?
Score: 0 (completely off-topic answer) → 1 (fully addresses the question)
"""
from __future__ import annotations

from ragcheck.core.evaluators.base import BaseEvaluator
from ragcheck.core.schema import EvalSample, MetricName, MetricScore

PROMPT_TEMPLATE = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system.

## Question
{question}

## Generated Answer
{answer}

## Task
Evaluate how well the generated answer addresses the question.
Consider:
- Does it directly answer what was asked?
- Does it stay on topic?
- Is it complete (covers all aspects of the question)?
- Does it avoid padding / irrelevant information?

Note: Do NOT penalize for lack of factual accuracy here — only evaluate relevance to the question.

## Response Format (JSON only, no other text)
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<brief explanation>",
  "addressed_aspects": ["<aspect of the question that was answered>"],
  "missing_aspects": ["<aspect of the question that was NOT answered>"]
}}

Score guide:
1.0 = Answer directly and completely addresses the question
0.7 = Mostly relevant with minor gaps
0.4 = Partially answers, significant parts of the question ignored
0.1 = Tangentially related but doesn't really answer
0.0 = Completely ignores the question
"""


class AnswerRelevanceEvaluator(BaseEvaluator):
    metric = MetricName.ANSWER_RELEVANCE

    async def evaluate(self, sample: EvalSample) -> MetricScore:
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=sample.question,
                answer=sample.answer,
            )
            result = await self.judge.judge(prompt)
            return MetricScore(
                metric=self.metric,
                score=result["score"],
                reasoning=result.get("reasoning", ""),
                details={
                    "addressed_aspects": result.get("addressed_aspects", []),
                    "missing_aspects": result.get("missing_aspects", []),
                },
            )
        except Exception as exc:
            return self._error_score(str(exc))
