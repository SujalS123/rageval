# Metrics Overview

rageval ships seven metrics. None of them return only a number. Every metric returns reasoning and evidence — the specific claims, documents, or sentences that drove the score.

| Metric | Measures | Needs ground truth | What low score means |
|---|---|---|---|
| Faithfulness | Does the answer make claims supported by context? | No | LLM is hallucinating |
| ContextPrecision | What fraction of retrieved docs were useful? | No | Retriever is returning noise |
| AnswerRelevancy | Does the answer address the original question? | No | Answer drifted off topic |
| ContextRecall | Did retrieval find all needed information? | Yes | Retriever is missing documents |
| NoiseSensitivity | Does pipeline degrade when noise is injected? | No | Pipeline is fragile |
| AnswerCompleteness | Does answer cover all available relevant facts? | No | Answer is incomplete |
| ContradictionDetector | Does answer contradict the context? | No | Answer reverses context facts |

## Using metrics

Every metric takes a judge and a threshold:

```python
from rageval.metrics.faithfulness import Faithfulness
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()
metric = Faithfulness(judge=judge, threshold=0.8)

result = metric.score(sample)
print(result.score)     # float 0.0–1.0
print(result.passed)    # score >= threshold
print(result.reasoning) # why
print(result.evidence)  # what specifically failed
```

## MetricResult fields

| Field | Type | Description |
|---|---|---|
| metric_name | str | Name of the metric |
| score | float | 0.0 to 1.0 |
| passed | bool | score >= threshold |
| reasoning | str | Explanation of the score |
| evidence | list[str] | Specific items that drove the score down |
| threshold | float | The configured threshold |
| hallucinations | list | Structured Hallucination objects (Faithfulness only) |

## Which metrics to use

Start with Faithfulness, ContextPrecision, and AnswerRelevancy. These three cover the most common failure modes and require no ground truth.

Add ContextRecall when you have ground truth answers and want to measure retriever coverage.

Add NoiseSensitivity when you want to know how robust your pipeline is to bad retrievals.
