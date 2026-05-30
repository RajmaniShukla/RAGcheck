"""
Real end-to-end test using the ragcheck Python SDK.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python run_test.py

    # Or with OpenAI:
    set OPENAI_API_KEY=sk-...
    python run_test.py --openai
"""
import os
import sys
from ragcheck import evaluate
from ragcheck.reporters.html_report import save_html

USE_OPENAI = "--openai" in sys.argv

if USE_OPENAI:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var not set.")
        sys.exit(1)
    judge_model = "gpt-4o-mini"
    judge_provider = "litellm"
else:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY env var not set.")
        sys.exit(1)
    judge_model = "claude-haiku-4-5"
    judge_provider = "anthropic"

questions = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "What are the main components of a RAG pipeline?",
]

contexts = [
    [
        "Retrieval-Augmented Generation (RAG) is a technique that combines information "
        "retrieval with language model generation. It was introduced by Facebook AI Research in 2020.",
        "In a RAG system, when a user asks a question, relevant documents are first retrieved "
        "from a knowledge base, then passed as context to a language model to generate an accurate answer.",
        "RAG helps reduce hallucination in LLMs by grounding responses in retrieved factual "
        "documents rather than relying solely on parametric knowledge.",
    ],
    [
        "A typical RAG pipeline consists of: (1) a document store or vector database, "
        "(2) an embedding model to encode queries and documents, (3) a retriever that finds "
        "relevant chunks, and (4) a generator (LLM) that produces the final answer.",
        "Vector databases like Pinecone, Weaviate, and Chroma are commonly used in RAG systems "
        "to store and search document embeddings efficiently.",
        "The quality of a RAG system depends heavily on chunk size, overlap, embedding model "
        "choice, and the number of retrieved chunks (top-k).",
    ],
]

answers = [
    "RAG (Retrieval-Augmented Generation) is a technique that combines document retrieval "
    "with LLM generation. It retrieves relevant context first, then uses it to generate "
    "grounded, accurate answers, reducing hallucination.",
    "The main components of a RAG pipeline are: a vector database (stores embeddings), "
    "an embedding model (encodes text), a retriever (finds relevant chunks), and an LLM "
    "generator (produces the final answer).",
]

ground_truths = [
    "Retrieval-Augmented Generation (RAG) is an AI framework introduced by Facebook AI "
    "Research that enhances language model outputs by retrieving relevant documents from "
    "a knowledge base and using them as context during generation.",
    "A RAG pipeline has four main components: a document store (vector database), an "
    "embedding model, a retriever for finding relevant chunks, and a language model generator.",
]

print(f"[ragcheck] Running real evaluation — judge: {judge_model}")
print("[ragcheck] 2 samples x 6 metrics\n")

results = evaluate(
    questions=questions,
    contexts=contexts,
    answers=answers,
    ground_truths=ground_truths,
    metrics=["all"],
    judge_model=judge_model,
    judge_provider=judge_provider,
    concurrency=3,
    fail_threshold=0.6,
)

print(results.summary())

save_html(results, "examples/ci_report.html")
print("\n[ragcheck] HTML report saved: examples/ci_report.html")
