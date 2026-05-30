"""
RAGcheck + LangChain Integration Example

Evaluates a LangChain RetrievalQA pipeline using ragcheck.

Requirements:
    pip install ragcheck langchain langchain-openai langchain-community faiss-cpu
    export OPENAI_API_KEY=sk-...
"""
from ragcheck.connectors.langchain import from_chain_outputs
from ragcheck import evaluate_dataset
from ragcheck.core.schema import EvalConfig, MetricName
import asyncio

# ---- Simulate LangChain chain outputs ----
# In a real scenario, you'd run:
#   chain = RetrievalQA.from_chain_type(llm=..., retriever=...)
#   outputs = [chain({"query": q}) for q in questions]

from langchain_core.documents import Document  # type: ignore

simulated_outputs = [
    {
        "query": "What is RAG?",
        "result": "RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLM generation.",
        "source_documents": [
            Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={}),
            Document(page_content="RAG was introduced by Facebook AI Research in 2020.", metadata={}),
        ],
    },
]

# Convert to ragcheck EvalDataset
dataset = from_chain_outputs(simulated_outputs, name="langchain_eval")
print(f"Loaded {len(dataset.samples)} samples")

# Run evaluation
config = EvalConfig(
    metrics=[MetricName.FAITHFULNESS, MetricName.CONTEXT_RELEVANCE, MetricName.ANSWER_RELEVANCE]
)

async def main():
    from ragcheck import evaluate_dataset
    report = await evaluate_dataset(dataset, config)
    print(report.summary())

asyncio.run(main())
