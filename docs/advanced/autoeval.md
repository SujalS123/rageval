# Production Monitoring

AutoEval samples production queries at a configurable rate and evaluates them in a background thread with zero latency impact on your users.

## Setup

```python
from rageval import AutoEval
from rageval.metrics.faithfulness import Faithfulness
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()
monitor = AutoEval(
    metrics=[Faithfulness(judge=judge, threshold=0.8)],
    sample_rate=0.1,          # evaluate 10% of queries
    alert_fn=my_alert,        # called when rolling score drops below threshold
    rolling_window=100,       # rolling average over last 100 evaluated queries
)
```

## Using the decorator

```python
@monitor.watch
def my_rag_pipeline(query: str) -> tuple[list[str], str]:
    docs = retriever.search(query)
    answer = llm.generate(query, docs)
    return docs, answer

# Your pipeline now works exactly as before.
# 10% of calls are evaluated in the background.
result = my_rag_pipeline("What is the capital of France?")
```

## Alert function

```python
def my_alert(metric_name: str, rolling_score: float, threshold: float):
    # Called when rolling_score drops below threshold
    slack.send(f"rageval alert: {metric_name} score {rolling_score:.2f} < {threshold}")
    pagerduty.trigger(f"RAG quality degradation: {metric_name}")
```

## Use case

Use AutoEval when you have a production RAG system and want to detect quality degradation before users report it. The evaluation runs in a background thread and has no impact on response latency.
