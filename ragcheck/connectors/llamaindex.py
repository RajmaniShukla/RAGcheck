"""
LlamaIndex connector — convert LlamaIndex query engine responses to EvalDataset.

Works with:
- VectorStoreIndex.as_query_engine()
- RetrieverQueryEngine
- Any engine returning a Response with source_nodes
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragcheck.core.schema import EvalDataset, EvalSample

if TYPE_CHECKING:
    pass


def from_query_responses(
    questions: list[str],
    responses: list[Any],  # list of llama_index.core.Response
    ground_truths: list[str] | None = None,
    name: str | None = None,
) -> EvalDataset:
    """
    Build an EvalDataset from LlamaIndex query engine responses.

    Example:
        query_engine = index.as_query_engine(similarity_top_k=5)
        responses = [query_engine.query(q) for q in questions]
        dataset = from_query_responses(questions, responses)
    """
    if len(questions) != len(responses):
        raise ValueError("questions and responses must have the same length")

    samples = []
    for i, (question, response) in enumerate(zip(questions, responses)):
        # Extract generated answer
        answer = str(response)

        # Extract source node texts
        contexts: list[str] = []
        source_nodes = getattr(response, "source_nodes", [])
        for node in source_nodes:
            # NodeWithScore or TextNode
            text = None
            if hasattr(node, "node"):
                text = getattr(node.node, "text", None) or getattr(node.node, "get_text", lambda: None)()
            elif hasattr(node, "text"):
                text = node.text
            elif hasattr(node, "get_text"):
                text = node.get_text()
            if text:
                contexts.append(str(text))

        if not contexts:
            # Fallback: try metadata
            contexts = ["[No source nodes returned]"]

        samples.append(
            EvalSample(
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truths[i] if ground_truths else None,
            )
        )
    return EvalDataset(samples=samples, name=name)


def from_retriever_results(
    questions: list[str],
    answers: list[str],
    retrieved_nodes_per_question: list[list[Any]],
    ground_truths: list[str] | None = None,
    name: str | None = None,
) -> EvalDataset:
    """
    Build an EvalDataset from separate retriever outputs and generated answers.
    """
    if not (len(questions) == len(answers) == len(retrieved_nodes_per_question)):
        raise ValueError("All input lists must have the same length")

    samples = []
    for i, (q, a, nodes) in enumerate(zip(questions, answers, retrieved_nodes_per_question)):
        contexts = []
        for node in nodes:
            text = None
            if hasattr(node, "node"):
                text = getattr(node.node, "text", None)
            elif hasattr(node, "text"):
                text = node.text
            if text:
                contexts.append(str(text))

        samples.append(
            EvalSample(
                question=q,
                contexts=contexts or ["[No context]"],
                answer=a,
                ground_truth=ground_truths[i] if ground_truths else None,
            )
        )
    return EvalDataset(samples=samples, name=name)
