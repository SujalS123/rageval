# API Reference

## evaluate()

Run all metrics on one sample. Never raises — errors are stored in MetricResult.

```python
from rageval import evaluate

def evaluate(
    sample: RAGSample,
    metrics: list[BaseMetric],
    weights: dict[str, float] = None,
) -> EvalResult:
    ...
```

**Parameters:**

- `sample` — The RAGSample to evaluate
- `metrics` — List of metric instances to run
- `weights` — Optional `{metric_name: weight}` dict for overall score. Default is equal weights.

**Returns:** `EvalResult`

**Example:**

```python
result = evaluate(
    sample=sample,
    metrics=[Faithfulness(judge=judge, threshold=0.8)],
)
print(result.overall_score)  # 0.85
print(result.passed)         # True
```

---

## batch_evaluate()

Evaluate a list of samples in parallel.

```python
def batch_evaluate(
    samples: list[RAGSample],
    metrics: list[BaseMetric],
    weights: dict[str, float] = None,
    max_workers: int = 4,
    show_progress: bool = True,
) -> list[EvalResult]:
    ...
```

**Parameters:**

- `samples` — List of RAGSamples to evaluate
- `metrics` — Metric instances to run on every sample
- `weights` — Optional metric weights
- `max_workers` — Parallel threads. Keep at 4–10 to avoid rate limits.
- `show_progress` — Print progress to terminal

**Returns:** `list[EvalResult]` in the same order as input

---

## summary()

Compute aggregate statistics across a batch of results.

```python
def summary(results: list[EvalResult]) -> dict:
    ...
```

**Returns:**

```python
{
    "total_samples": int,
    "overall_pass_rate": float,
    "avg_overall_score": float,
    "per_metric": {
        "metric_name": {
            "avg_score": float,
            "pass_rate": float,
            "samples": int,
        }
    }
}
```

---

## RAGSample

The input contract. Every evaluation starts with a RAGSample.

```python
from rageval import RAGSample

@dataclass
class RAGSample:
    query: str
    retrieved_docs: list[str | RetrievedDoc]
    answer: str
    ground_truth: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def retrieved_texts(self) -> list[str]: ...
```

**Parameters:**

- `query` — The user's question
- `retrieved_docs` — Documents returned by the retriever. Accepts `list[str]` or `list[RetrievedDoc]`.
- `answer` — The LLM's generated answer
- `ground_truth` — The correct answer. Required only for ContextRecall.
- `metadata` — Arbitrary key-value pairs for tracking

Validates on construction. Raises `ValueError` if query, docs, or answer are empty.

---

## MetricResult

The output of every metric.

```python
@dataclass
class MetricResult:
    metric_name: str
    score: float           # 0.0 to 1.0
    passed: bool           # score >= threshold
    reasoning: str         # why the score is this value
    evidence: list[str]    # specific items that drove the score
    threshold: float
    hallucinations: list   # list[Hallucination], Faithfulness only
```

---

## EvalResult

Aggregated result from evaluate() or batch_evaluate().

```python
@dataclass
class EvalResult:
    sample: RAGSample
    metric_results: dict[str, MetricResult]
    overall_score: float
    passed: bool
    latency_ms: float

    def summary(self) -> str: ...
```

---

## RunTracker

```python
from rageval import RunTracker

tracker = RunTracker()  # creates .rageval/runs.db
tracker.save_run(run_name="v2.3", results=results)
tracker.list_runs()
tracker.compare_runs("v2.2", "v2.3")
```

---

## EvalDatasetGenerator

```python
from rageval import EvalDatasetGenerator

generator = EvalDatasetGenerator(judge=judge)
questions = generator.generate(documents=my_docs, n_questions=50)
generator.save(questions, "eval_data.json")
```

---

## ConsistencyAnalyzer

```python
from rageval import ConsistencyAnalyzer
from rageval.judges.heuristic import HeuristicJudge

analyzer = ConsistencyAnalyzer(judge=judge, embedding_judge=HeuristicJudge())
report = analyzer.analyze(
    query="original question",
    paraphrases=["paraphrase 1", "paraphrase 2"],
    pipeline_fn=my_pipeline,
)
print(report.consistency_score)
print(report.root_cause_hypothesis)
```

---

## RAGTracer

```python
from rageval import RAGTracer

tracer = RAGTracer(metrics=[])
with tracer.trace(trace_id="req-123") as trace:
    trace._current_step.inputs["query"] = query
    trace.log_retrieval(docs=docs)
    trace.log_generation(answer=answer)
result = tracer.evaluate_trace(trace)
print(result.root_cause_step)
```
