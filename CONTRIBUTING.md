# Contributing to rageval

## Setup

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/rageval
cd rageval
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[all]"
pip install pytest
```

## Running tests

```bash
# Unit tests — no API key needed
python -m pytest tests/ -v --ignore=tests/integration

# Integration tests — requires at least one API key
export ANTHROPIC_API_KEY=your_key   # or OPENAI_API_KEY or GROQ_API_KEY
python -m pytest tests/integration/ -v -s
```

## Adding a new metric

Read the "Adding a New Metric" section in GUIDE.md. The pattern is:

1. Create `rageval/metrics/your_metric.py` following the `BaseMetric` pattern
2. Write a prompt that ends with `Respond ONLY with a JSON object. No explanation. No markdown fences.`
3. Implement `score(self, sample: RAGSample) -> MetricResult`
4. Wrap every LLM call in try/except — errors go to `MetricResult.reasoning`, never raise
5. Register in `rageval/cli.py` metric_registry dict
6. Write at least 5 unit tests using `unittest.mock.MagicMock` for the judge

## Adding a new judge

Read the "Adding a New Judge" section in GUIDE.md. The pattern is:

1. Create `rageval/judges/your_judge.py`
2. Inherit `BaseJudge`
3. Implement `complete(self, prompt: str, system: str = "") -> str`
4. `complete_json()` is inherited — do not reimplement it
5. Write at least 3 unit tests

## Pull request checklist

- [ ] `python -m pytest tests/` passes with no failures
- [ ] New metric has at least 5 unit tests
- [ ] New judge has at least 3 unit tests
- [ ] GUIDE.md updated if new feature added
- [ ] CHANGELOG.md entry added under the appropriate version
- [ ] No new hard dependencies added to `pyproject.toml` without discussion
