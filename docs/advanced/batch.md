# Batch Evaluation

Evaluate many samples in parallel using `batch_evaluate()`.

## Basic usage

```python
from rageval import batch_evaluate, summary, RAGSample
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()
metrics = [
    Faithfulness(judge=judge, threshold=0.8),
    ContextPrecision(judge=judge, threshold=0.7),
]

samples = [
    RAGSample(query=q, retrieved_docs=docs, answer=ans)
    for q, docs, ans in your_data
]

results = batch_evaluate(
    samples=samples,
    metrics=metrics,
    max_workers=4,       # parallel threads — keep low to avoid rate limits
    show_progress=True,
)
```

## Aggregated statistics

```python
stats = summary(results)
print(stats)
# {
#   "total_samples": 100,
#   "overall_pass_rate": 0.84,
#   "avg_overall_score": 0.79,
#   "per_metric": {
#     "faithfulness": {"avg_score": 0.87, "pass_rate": 0.91, "samples": 100},
#     "context_precision": {"avg_score": 0.79, "pass_rate": 0.83, "samples": 100},
#   }
# }
```

## Export results

```python
from rageval.reporters.json_csv import to_json, to_csv

to_json(results, "results.json")
to_csv(results, "results.csv")
```

## Weighted overall score

```python
results = batch_evaluate(
    samples=samples,
    metrics=metrics,
    weights={"faithfulness": 2.0, "context_precision": 1.0},
)
```

## Order preservation

Results are always returned in the same order as input samples, regardless of which samples finish first.

## Rate limit guidance

- `max_workers=4` — safe for most providers
- `max_workers=1` — if hitting rate limit errors
- `max_workers=8` — safe for Groq (very high rate limits)
