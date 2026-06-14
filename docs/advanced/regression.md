# Regression Tracking

RunTracker saves evaluation results to a local SQLite database and lets you compare runs over time. It has zero external dependencies — `sqlite3` is in Python's standard library.

## Setup

```python
from rageval import RunTracker

tracker = RunTracker()  # creates .rageval/runs.db in current directory
```

## Saving a run

```python
results = batch_evaluate(samples=samples, metrics=metrics)
tracker.save_run(run_name="v2.3-deploy", results=results)
```

## CLI commands

```bash
# List all saved runs
rageval history

# Compare two runs
rageval diff v2.2-deploy v2.3-deploy

# Save during a CLI evaluation run
rageval run eval_data.json --judge anthropic --save-run v2.3-deploy
```

## Output

```
Run History — faithfulness

v2.3  0.91  ▲ +0.06
v2.2  0.85  ▲ +0.03
v2.1  0.82  ▼ -0.04
v2.0  0.86
```

## What is stored

For each run: name, timestamp, total samples, overall score, pass rate, per-metric averages, hallucination type counts.

## Use case

Use RunTracker when you are iterating on your RAG pipeline — changing prompts, embedding models, or chunking strategies — and want to know whether each change improved or degraded evaluation quality.
