# Contributing to RAGcheck

First off — thank you for considering contributing! 🎉

RAGcheck is a community-driven project and we welcome all kinds of contributions:
bug fixes, new evaluators, new judge backends, documentation improvements, and more.

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/ragcheck
cd ragcheck
```

### 2. Set Up Environment

```bash
# Install Poetry (if not already)
pip install poetry

# Install all dependencies (including dev tools)
poetry install --with dev

# Activate the virtual environment
poetry shell
```

### 3. Run Tests

```bash
pytest                   # run all tests
pytest -x                # stop on first failure
pytest tests/test_evaluators.py  # specific file
pytest -v --tb=short     # verbose with short tracebacks
```

### 4. Lint & Format

```bash
ruff check ragcheck/     # lint
ruff format ragcheck/    # format
```

---

## 🧩 Project Structure

```
ragcheck/
├── core/
│   ├── evaluators/     # Add new metrics here
│   ├── judges/         # Add new judge backends here
│   ├── pipeline.py
│   └── schema.py
├── connectors/         # Add framework integrations here
├── reporters/          # Add output formats here
└── cli.py
```

---

## ➕ Adding a New Evaluator

1. Create `ragcheck/core/evaluators/your_metric.py`
2. Subclass `BaseEvaluator` and implement `evaluate(sample) -> MetricScore`
3. Add `MetricName.YOUR_METRIC` to `ragcheck/core/schema.py`
4. Register it in `ragcheck/core/evaluators/__init__.py`
5. Add tests in `tests/test_evaluators.py`
6. Document it in `docs/guides/metrics.md`

```python
class YourMetricEvaluator(BaseEvaluator):
    metric = MetricName.YOUR_METRIC

    async def evaluate(self, sample: EvalSample) -> MetricScore:
        try:
            prompt = YOUR_PROMPT_TEMPLATE.format(
                question=sample.question,
                contexts=self._fmt_contexts(sample.contexts),
                answer=sample.answer,
            )
            result = await self.judge.judge(prompt)
            return MetricScore(
                metric=self.metric,
                score=result["score"],
                reasoning=result.get("reasoning", ""),
            )
        except Exception as exc:
            return self._error_score(str(exc))
```

---

## ➕ Adding a New Judge Backend

1. Create `ragcheck/core/judges/your_judge.py`
2. Subclass `BaseJudge` and implement `async judge(prompt) -> dict`
3. Add it to `JudgeProvider` enum in `schema.py`
4. Register it in `ragcheck/core/judges/__init__.py`'s `build_judge()` factory

---

## 📋 Good First Issues

Look for issues labeled `good first issue`:

- Adding a new output format (CSV reporter)
- Adding a new connector (Haystack, custom API)
- Improving prompt templates for existing evaluators
- Adding more test coverage
- Improving documentation
- Adding example notebooks

---

## 🔀 Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-new-evaluator`
2. Write tests for your changes
3. Ensure all tests pass: `pytest`
4. Ensure linting passes: `ruff check ragcheck/`
5. Update documentation if needed
6. Submit a PR with a clear description

### PR Title Format
- `feat: add coherence evaluator`
- `fix: handle empty contexts in faithfulness evaluator`
- `docs: add LlamaIndex integration guide`
- `test: improve pipeline test coverage`

---

## 🏷️ Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new metric
fix: handle edge case in judge retry logic
docs: update quickstart guide
test: add context recall tests
chore: update dependencies
```

---

## 📜 Code of Conduct

Be kind, be respectful, be constructive. We're all here to build something useful together.

---

## ❓ Questions?

Open an issue or start a discussion on GitHub. We're happy to help!
