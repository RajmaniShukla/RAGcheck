"""
RAGcheck Quickstart — evaluate a RAG pipeline in 10 lines.

Requirements:
    pip install ragcheck
    export OPENAI_API_KEY=sk-...  # or use Anthropic/Ollama
"""
from ragcheck import evaluate

# Your RAG pipeline outputs
questions = [
    "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG pipeline?",
]

contexts = [
    [
        "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with language model generation.",
        "It was introduced by Facebook AI Research in 2020 to reduce hallucinations in LLMs.",
    ],
    [
        "A RAG pipeline consists of: a vector database, an embedding model, a retriever, and an LLM generator.",
        "Vector databases like Pinecone, Weaviate, and Chroma store document embeddings for fast similarity search.",
    ],
]

answers = [
    "RAG is a technique that retrieves relevant documents and uses them as context to generate accurate answers.",
    "The main components are a vector database, an embedding model, a retriever, and a language model.",
]

ground_truths = [
    "RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation to reduce hallucinations.",
    "A RAG pipeline has four main components: vector database, embedding model, retriever, and LLM generator.",
]

# Run evaluation
results = evaluate(
    questions=questions,
    contexts=contexts,
    answers=answers,
    ground_truths=ground_truths,
    metrics=["faithfulness", "context_relevance", "answer_relevance", "context_recall"],
    judge_model="gpt-4o-mini",
)

# Print summary
print(results.summary())

# Access individual scores
for sample_result in results.results:
    print(f"\nQ: {sample_result.sample.question[:60]}")
    for score in sample_result.scores:
        status = "✅" if score.score >= 0.7 else "⚠️" if score.score >= 0.4 else "❌"
        print(f"  {status} {score.metric.value}: {score.score:.3f}")
        if score.reasoning:
            print(f"     → {score.reasoning[:100]}")
