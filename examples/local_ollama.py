"""
RAGcheck with Local Ollama Judge

Use a locally running Llama3 model as the evaluation judge.
No API keys needed!

Requirements:
    pip install ragcheck
    # Install Ollama: https://ollama.ai
    ollama pull llama3
    ollama serve  # starts on http://localhost:11434
"""
from ragcheck import evaluate
from ragcheck.core.schema import JudgeProvider

results = evaluate(
    questions=["What is RAG?"],
    contexts=[["RAG stands for Retrieval-Augmented Generation, combining retrieval with generation."]],
    answers=["RAG is a technique that enhances LLM responses with retrieved documents."],
    metrics=["faithfulness", "context_relevance", "answer_relevance"],
    judge_model="llama3",
    judge_provider="local",
    api_base="http://localhost:11434/v1",  # default Ollama endpoint
)

print(results.summary())
