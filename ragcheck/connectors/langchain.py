"""
LangChain connector — convert LangChain QA chain outputs to EvalDataset.

Works with:
- RetrievalQA
- ConversationalRetrievalChain
- Custom chains that return source_documents
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragcheck.core.schema import EvalDataset, EvalSample

if TYPE_CHECKING:
    pass  # avoid hard dependency at import time


def from_chain_outputs(
    outputs: list[dict[str, Any]],
    question_key: str = "query",
    answer_key: str = "result",
    source_docs_key: str = "source_documents",
    ground_truth_key: str | None = "ground_truth",
    name: str | None = None,
) -> EvalDataset:
    """
    Build an EvalDataset from a list of LangChain chain output dicts.

    Typical RetrievalQA output dict structure:
    {
        "query": "What is RAG?",
        "result": "RAG stands for ...",
        "source_documents": [Document(page_content="...", metadata={}), ...]
    }

    Example:
        chain = RetrievalQA.from_chain_type(...)
        outputs = [chain({"query": q}) for q in questions]
        dataset = from_chain_outputs(outputs)
    """
    samples = []
    for i, out in enumerate(outputs):
        try:
            question = out[question_key]
            answer = out[answer_key]

            # Extract text from LangChain Document objects
            source_docs = out.get(source_docs_key, [])
            contexts: list[str] = []
            for doc in source_docs:
                if hasattr(doc, "page_content"):
                    contexts.append(doc.page_content)
                elif isinstance(doc, str):
                    contexts.append(doc)
                elif isinstance(doc, dict):
                    contexts.append(doc.get("page_content", str(doc)))

            if not contexts:
                contexts = ["[No source documents returned]"]

            ground_truth: str | None = None
            if ground_truth_key and ground_truth_key in out:
                ground_truth = out[ground_truth_key]

            samples.append(
                EvalSample(
                    question=question,
                    contexts=contexts,
                    answer=answer,
                    ground_truth=ground_truth,
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Output {i} missing required key {exc}. "
                f"Available keys: {list(out.keys())}"
            ) from exc

    return EvalDataset(samples=samples, name=name)


def from_retriever_and_answers(
    questions: list[str],
    answers: list[str],
    retrieved_docs_per_question: list[list[Any]],
    ground_truths: list[str] | None = None,
    name: str | None = None,
) -> EvalDataset:
    """
    Build an EvalDataset from separate lists of questions, answers, and retrieved docs.
    Useful when you run the retriever and generator separately.
    """
    if not (len(questions) == len(answers) == len(retrieved_docs_per_question)):
        raise ValueError("questions, answers, and retrieved_docs_per_question must have equal length")

    samples = []
    for i, (q, a, docs) in enumerate(zip(questions, answers, retrieved_docs_per_question, strict=False)):
        contexts = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                contexts.append(doc.page_content)
            elif isinstance(doc, str):
                contexts.append(doc)

        samples.append(
            EvalSample(
                question=q,
                contexts=contexts or ["[No context]"],
                answer=a,
                ground_truth=ground_truths[i] if ground_truths else None,
            )
        )
    return EvalDataset(samples=samples, name=name)
