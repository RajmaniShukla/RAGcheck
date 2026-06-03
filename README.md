# 🔍 RAGcheck
Rajmani Shukla
> **An open-source toolkit to measure, debug, and improve your RAG pipeline quality.**

[![PyPI version](https://badge.fury.io/py/ragcheck.svg)](https://badge.fury.io/py/ragcheck)
[![Build Status](https://github.com/RajmaniShukla/ragcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/RajmaniShukla/ragcheck/actions)
[![Coverage](https://codecov.io/gh/RajmaniShukla/ragcheck/branch/main/graph/badge.svg)](https://codecov.io/gh/RajmaniShukla/ragcheck)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Given a question, retrieved context, and a generated answer — RAGcheck scores how good your whole pipeline is across **6 dimensions** and surfaces actionable insights.

---

## ✨ Features

- 📊 **6 production-ready metrics** — faithfulness, context relevance, answer relevance, context recall, noise sensitivity, chunk utilization
- 🤖 **Model-agnostic judging** — GPT-4o, Claude Haiku 4, Gemini, or local Llama via Ollama/vLLM
- 🔌 **Framework-agnostic** — native connectors for LangChain & LlamaIndex, generic JSON/CSV input
- 🚀 **CI-native** — GitHub Action with configurable pass/fail thresholds
- 🎨 **Beautiful output** — Rich terminal tables, standalone HTML dashboard, JSON export
- 📱 **Streamlit dashboard** — interactive visualization of evaluation reports
- ⚡ **Async & fast** — concurrent judge calls with configurable parallelism

---

## 📦 Installation

```bash
pip install ragcheck
```

With Streamlit dashboard:
```bash
pip install "ragcheck[dashboard]"
```

---

## 🚀 Quickstart

### Python SDK

```python
from ragcheck import evaluate

results = evaluate(
    questions=["What is RAG?"],
    contexts=[["RAG stands for Retrieval-Augmented Generation...", "It was introduced by Facebook AI..."]],
    answers=["RAG is a technique that combines retrieval with LLM generation for accurate answers."],
    ground_truths=["RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation."],
    metrics=["faithfulness", "context_relevance", "answer_relevance", "context_recall"],
    judge_model="gpt-4o-mini",   # or: "claude-haiku-4-5", "ollama/llama3"
)

print(results.summary())
# Dataset: None
# Samples evaluated: 1
# Overall score: 0.876
# 
# Per-metric averages:
#   faithfulness           mean=0.920  min=0.920  max=0.920
#   context_relevance      mean=0.850  min=0.850  max=0.850
#   answer_relevance       mean=0.880  min=0.880  max=0.880
#   context_recall         mean=0.855  min=0.855  max=0.855
```

### CLI

```bash
# Evaluate from a JSON file
ragcheck eval --input tests/data.json --metrics all --judge gpt-4o --output report.html

# Use Anthropic as judge
ragcheck eval --input data.json --judge claude-haiku-4-5 --provider anthropic --output report.json

# Use local Ollama (no API key needed!)
ragcheck eval --input data.json --judge llama3 --provider local --api-base http://localhost:11434/v1

# CI mode with threshold
ragcheck eval --input data.json --fail-threshold 0.7

# List all available metrics
ragcheck metrics
```

### GitHub Action

```yaml
- name: Evaluate RAG pipeline quality
  uses: ragcheck/eval-action@v1
  with:
    input: tests/rag_test_cases.json
    judge: gpt-4o-mini
    metrics: all
    fail_threshold: "0.7"
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## 📏 Metrics

| Metric | What it measures | Needs `ground_truth`? |
|--------|-----------------|----------------------|
| `context_relevance` | Are retrieved chunks relevant to the question? | No |
| `faithfulness` | Is the answer grounded in context? (no hallucination) | No |
| `answer_relevance` | Does the answer actually address the question? | No |
| `context_recall` | Did retrieval cover all facts needed to answer? | **Yes** |
| `noise_sensitivity` | How robust is the answer to irrelevant chunk injection? | No |
| `chunk_utilization` | Which retrieved chunks were actually used by the LLM? | No |

All metrics use **LLM-as-judge** and return:
- `score` (0.0 – 1.0)
- `reasoning` (natural language explanation)
- `details` (structured data: unsupported claims, missing facts, etc.)

---

## 🤖 Supported Judges

```python
# OpenAI
evaluate(..., judge_model="gpt-4o-mini")
evaluate(..., judge_model="gpt-4o")

# Anthropic
evaluate(..., judge_model="claude-haiku-4-5", judge_provider="anthropic")
evaluate(..., judge_model="claude-sonnet-4-5", judge_provider="anthropic")

# Google
evaluate(..., judge_model="gemini/gemini-pro")

# Local Ollama (no API key!)
evaluate(..., judge_model="llama3", judge_provider="local", api_base="http://localhost:11434/v1")

# Any LiteLLM-supported model
evaluate(..., judge_model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1")
```

---

## 🔌 Framework Integrations

### LangChain

```python
from ragcheck.connectors.langchain import from_chain_outputs

chain = RetrievalQA.from_chain_type(llm=..., retriever=...)
outputs = [chain({"query": q}) for q in questions]

dataset = from_chain_outputs(outputs)
```

### LlamaIndex

```python
from ragcheck.connectors.llamaindex import from_query_responses

query_engine = index.as_query_engine(similarity_top_k=5)
responses = [query_engine.query(q) for q in questions]

dataset = from_query_responses(questions, responses)
```

### Custom JSON

```json
[
  {
    "question": "What is RAG?",
    "contexts": ["RAG stands for...", "It was introduced by..."],
    "answer": "RAG is a technique...",
    "ground_truth": "RAG combines retrieval with generation..."
  }
]
```

```bash
ragcheck eval --input your_data.json
```

---

## 📊 Streamlit Dashboard

```bash
pip install "ragcheck[dashboard]"
streamlit run ragcheck/dashboard/app.py
```

Or generate an HTML report:
```bash
ragcheck eval --input data.json --output report.html
# Open report.html in any browser — no server needed
```

---

## 🏗️ Architecture

```
ragcheck/
├── core/
│   ├── evaluators/          # One module per metric
│   │   ├── context_relevance.py
│   │   ├── faithfulness.py
│   │   ├── answer_relevance.py
│   │   ├── context_recall.py
│   │   ├── noise_sensitivity.py
│   │   └── chunk_utilization.py
│   ├── judges/              # LLM-as-judge backends
│   │   ├── base.py
│   │   ├── litellm_judge.py  # default (all providers)
│   │   ├── openai_judge.py
│   │   ├── anthropic_judge.py
│   │   └── local_judge.py    # Ollama / vLLM
│   ├── pipeline.py          # Orchestrates all evaluators
│   └── schema.py            # Pydantic v2 I/O models
├── connectors/              # Framework adapters
│   ├── langchain.py
│   ├── llamaindex.py
│   └── custom.py
├── reporters/
│   ├── terminal.py          # Rich CLI tables
│   ├── html_report.py       # Standalone HTML dashboard
│   └── json_export.py
├── github_action/           # GitHub Action integration
├── dashboard/               # Streamlit UI
└── cli.py                   # ragcheck eval CLI
```

---

## 🔬 How It Works

Every evaluator uses **LLM-as-judge**: a structured prompt is sent to an LLM asking it to score a specific dimension, then the score + reasoning are parsed from the JSON response.

**Example (faithfulness):**
```
Given:
  Context: {retrieved_chunks}
  Answer: {generated_answer}

Rate how faithful the answer is to the context on a scale of 0–1.
Return JSON: {"score": float, "reasoning": str, "unsupported_claims": list}
```

This makes RAGcheck fully model-agnostic — swap in any LLM as judge.

---

## 🌟 Comparison

| Tool | ragcheck | RAGAS | TruLens | DeepEval |
|------|----------|-------|---------|----------|
| Framework-agnostic | ✅ | ⚠️ | ❌ | ⚠️ |
| Local LLM judge | ✅ | ❌ | ❌ | ❌ |
| CI/GitHub Action | ✅ | ❌ | ❌ | ⚠️ |
| Lightweight (no DB) | ✅ | ✅ | ❌ | ⚠️ |
| HTML report | ✅ | ❌ | ✅ | ✅ |
| Open source | ✅ | ✅ | ✅ | ⚠️ |

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/RajmaniShukla/ragcheck
cd ragcheck
poetry install --with dev
poetry run pytest
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<p align="center">Built with ❤️ by <a href="https://github.com/RajmaniShukla">Rajmani Shukla</a></p>
