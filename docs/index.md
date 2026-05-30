# RAGcheck

**An open-source toolkit to measure, debug, and improve your RAG pipeline quality.**

[![PyPI version](https://badge.fury.io/py/ragcheck.svg)](https://badge.fury.io/py/ragcheck)
[![Build Status](https://github.com/RajmaniShukla/ragcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/RajmaniShukla/ragcheck/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What is RAGcheck?

RAGcheck gives you **6 production-ready metrics** to evaluate every layer of your RAG pipeline — from retrieval quality to answer faithfulness — using **LLM-as-judge**.

```python
from ragcheck import evaluate

results = evaluate(
    questions=["What is RAG?"],
    contexts=[["RAG stands for Retrieval-Augmented Generation..."]],
    answers=["RAG is a technique that combines retrieval with generation."],
    metrics=["faithfulness", "context_relevance", "answer_relevance"],
)
print(results.summary())
```

## Metrics

| Metric | What it measures |
|--------|-----------------|
| `context_relevance` | Are the retrieved chunks relevant to the question? |
| `faithfulness` | Is the answer grounded in context? (no hallucination) |
| `answer_relevance` | Does the answer address the question? |
| `context_recall` | Did retrieval cover all facts needed to answer? |
| `noise_sensitivity` | How robust is the answer to irrelevant chunk injection? |
| `chunk_utilization` | Which retrieved chunks were actually used? |

## Why RAGcheck?

- 🔌 **Framework-agnostic** — works with LangChain, LlamaIndex, or plain Python
- 🤖 **Judge-agnostic** — GPT-4o, Claude, Gemini, or local Llama (Ollama)
- 🚀 **CI-native** — GitHub Action with pass/fail thresholds
- 🎯 **Actionable** — detailed reasoning, not just numbers
- 📦 **Lightweight** — no database, no server required
