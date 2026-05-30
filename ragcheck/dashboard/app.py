"""
RAGcheck Streamlit Dashboard
-----------------------------
Launch with: streamlit run ragcheck/dashboard/app.py
Or via: ragcheck dashboard (future CLI command)
"""
from __future__ import annotations

import json

import streamlit as st

# Page config
st.set_page_config(
    page_title="RAGcheck Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS ----
st.markdown("""
<style>
    .metric-card { background: #1e2130; border-radius: 12px; padding: 1.2rem; text-align: center; }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .green { color: #22c55e; } .yellow { color: #eab308; }
    .orange { color: #f97316; } .red { color: #ef4444; }
    .sample-row { padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


def score_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    elif score >= 0.4:
        return "orange"
    else:
        return "red"


def score_emoji(score: float) -> str:
    if score >= 0.8:
        return "🟢"
    elif score >= 0.6:
        return "🟡"
    elif score >= 0.4:
        return "🟠"
    else:
        return "🔴"


# ---- Sidebar ----
with st.sidebar:
    st.image("https://raw.githubusercontent.com/RajmaniShukla/ragcheck/main/docs/logo.png",
             width=120, use_column_width=False)
    st.title("🔍 RAGcheck")
    st.caption("RAG Pipeline Quality Evaluator")
    st.divider()

    st.subheader("Load Report")
    uploaded = st.file_uploader("Upload JSON report", type=["json"])
    st.divider()
    st.caption("Or run from CLI:")
    st.code("ragcheck eval --input data.json --output report.json", language="bash")

# ---- Main content ----
st.title("RAG Pipeline Evaluation Dashboard")

if uploaded is None:
    # Demo / landing state
    st.info("👆 Upload a `ragcheck eval --output report.json` file to visualize results.")

    st.subheader("What ragcheck measures")
    cols = st.columns(3)
    metrics_info = [
        ("🎯 Context Relevance", "Are the retrieved chunks actually relevant to the question?"),
        ("🛡️ Faithfulness", "Is the answer grounded in context? (no hallucination)"),
        ("💬 Answer Relevance", "Does the answer actually address the question?"),
        ("📚 Context Recall", "Did retrieval cover all facts needed to answer?"),
        ("🔊 Noise Sensitivity", "How robust is the answer to irrelevant chunk injection?"),
        ("🧩 Chunk Utilization", "Which retrieved chunks were actually used by the LLM?"),
    ]
    for i, (title, desc) in enumerate(metrics_info):
        with cols[i % 3]:
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.divider()
    st.stop()

# ---- Load report ----
try:
    data = json.load(uploaded)
except Exception as exc:
    st.error(f"Failed to parse JSON: {exc}")
    st.stop()

# ---- Overview header ----
overall_score = data.get("overall_score", 0.0)
passed = data.get("passed")
dataset_name = data.get("dataset_name") or "Unnamed Dataset"
results = data.get("results", [])
agg_stats = data.get("aggregate_stats", [])
judge_model = data.get("config", {}).get("judge", {}).get("model", "unknown")

col1, col2, col3, col4 = st.columns(4)
with col1:
    cls = score_color(overall_score)
    st.markdown(f"<div class='metric-card'><div class='metric-value {cls}'>{overall_score:.3f}</div><div>Overall Score</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(results)}</div><div>Samples</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(agg_stats)}</div><div>Metrics</div></div>", unsafe_allow_html=True)
with col4:
    if passed is True:
        badge = "✅ PASSED"
    elif passed is False:
        badge = "❌ FAILED"
    else:
        badge = "—"
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{badge}</div><div>Threshold</div></div>", unsafe_allow_html=True)

st.caption(f"**Dataset:** {dataset_name} &nbsp;|&nbsp; **Judge:** {judge_model}")
st.divider()

# ---- Per-metric bar chart ----
if agg_stats:
    try:
        import plotly.graph_objects as go

        metric_names = [s["metric"].replace("_", " ").title() for s in agg_stats]
        means = [s["mean"] for s in agg_stats]
        colors = [
            "#22c55e" if m >= 0.8 else "#eab308" if m >= 0.6 else "#f97316" if m >= 0.4 else "#ef4444"
            for m in means
        ]

        fig = go.Figure(go.Bar(
            x=metric_names,
            y=means,
            marker_color=colors,
            text=[f"{m:.3f}" for m in means],
            textposition="outside",
        ))
        fig.update_layout(
            title="Per-Metric Average Scores",
            yaxis=dict(range=[0, 1.15], title="Score"),
            plot_bgcolor="#0f1117",
            paper_bgcolor="#0f1117",
            font_color="#e2e8f0",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback: plain table
        st.subheader("Per-Metric Summary")
        for stat in agg_stats:
            cls = score_color(stat["mean"])
            st.markdown(
                f"{score_emoji(stat['mean'])} **{stat['metric']}** — "
                f"mean: `{stat['mean']:.3f}` | min: `{stat['min']:.3f}` | max: `{stat['max']:.3f}`"
            )

st.divider()

# ---- Per-sample breakdown ----
st.subheader(f"Per-Sample Breakdown ({len(results)} samples)")

filter_metric = st.selectbox(
    "Sort / filter by metric",
    options=["aggregate"] + [s["metric"] for s in agg_stats],
)

sort_asc = st.checkbox("Sort ascending (worst first)", value=True)


def get_sort_score(result: dict) -> float:
    if filter_metric == "aggregate":
        valid = [s["score"] for s in result.get("scores", []) if not s.get("error")]
        return sum(valid) / len(valid) if valid else 0.0
    for s in result.get("scores", []):
        if s["metric"] == filter_metric:
            return s["score"] if not s.get("error") else 0.0
    return 0.0


sorted_results = sorted(results, key=get_sort_score, reverse=not sort_asc)

for i, result in enumerate(sorted_results):
    sample = result.get("sample", {})
    question = sample.get("question", "")[:100]
    answer = sample.get("answer", "")[:200]
    scores = result.get("scores", [])

    valid_scores = [s["score"] for s in scores if not s.get("error")]
    agg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    cls = score_color(agg)

    with st.expander(f"{score_emoji(agg)} Sample {i + 1}: {question}…  `({agg:.3f})`"):
        st.markdown(f"**Question:** {sample.get('question', '')}")
        st.markdown(f"**Answer:** {answer}{'…' if len(sample.get('answer','')) > 200 else ''}")

        if sample.get("ground_truth"):
            st.markdown(f"**Ground Truth:** {sample['ground_truth'][:200]}")

        st.markdown("---")
        score_cols = st.columns(len(scores)) if scores else []
        for j, score_data in enumerate(scores):
            metric = score_data["metric"].replace("_", " ").title()
            sc = score_data["score"]
            err = score_data.get("error")
            c = score_color(sc)
            with score_cols[j] if score_cols else st:
                if err:
                    st.metric(metric, "ERR", help=err)
                else:
                    st.metric(metric, f"{sc:.3f}")

        # Show reasoning for each metric
        for score_data in scores:
            reasoning = score_data.get("reasoning", "")
            if reasoning:
                st.caption(f"**{score_data['metric']}:** {reasoning}")

        # Context chunks
        contexts = sample.get("contexts", [])
        if contexts:
            with st.expander(f"📄 Retrieved Chunks ({len(contexts)})"):
                for k, chunk in enumerate(contexts):
                    st.markdown(f"**Chunk {k + 1}:**")
                    st.text(chunk[:500] + ("…" if len(chunk) > 500 else ""))
