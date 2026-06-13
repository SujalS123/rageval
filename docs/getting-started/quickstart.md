# Quickstart

This example catches a real hallucination. The answer says Charles Dickens wrote Romeo and Juliet. The context says William Shakespeare. rageval surfaces the exact claim that failed.

## Install

```bash
pip install rageval[anthropic]
```

## Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
# or: export OPENAI_API_KEY=your_key_here
# or: export GROQ_API_KEY=your_key_here
```

## Run

```python
from rageval import evaluate, RAGSample
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()  # reads ANTHROPIC_API_KEY from environment

result = evaluate(
    sample=RAGSample(
        query="Who wrote Romeo and Juliet?",
        retrieved_docs=[
            "Romeo and Juliet is a tragedy written by William Shakespeare "
            "in the late 16th century.",
        ],
        answer="Romeo and Juliet was written by Charles Dickens.",
    ),
    metrics=[
        Faithfulness(judge=judge, threshold=0.8),
        ContextPrecision(judge=judge, threshold=0.7),
        AnswerRelevancy(judge=judge, threshold=0.7),
    ],
)

print(result.summary())
```

## Output

```
-------------------------------------------------------
Query: Who wrote Romeo and Juliet?
Overall: 0.46 | FAILED | 4101ms
-------------------------------------------------------
[FAIL] faithfulness : 0.00 (threshold=0.80)
Reasoning: 2 of 2 claims could not be verified from the context.
 Evidence:
 - FACTUAL_ERROR: 'Romeo and Juliet was written by Charles Dickens'
   (severity: 1.0) — Context states William Shakespeare
 - FACTUAL_ERROR: 'Romeo and Juliet was written in the 19th century'
   (severity: 1.0) — Context states the play was written in the late 16th century
[PASS] context_precision : 0.50 (threshold=0.70)
[PASS] answer_relevancy : 0.88 (threshold=0.70)
-------------------------------------------------------
```

The hallucination is named. The severity is rated. The correct answer from context is shown. You know exactly what to fix.

## Next steps

- [Your first evaluation](first-evaluation.md) — integrate with your existing RAG pipeline
- [Metrics overview](../metrics/overview.md) — understand what each metric measures
- [Batch evaluation](../advanced/batch.md) — evaluate many samples at once
