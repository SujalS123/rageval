# Semantic Drift Detection

RAG pipelines are built and evaluated against a static dataset (the "baseline"). But in production, users inevitably start asking questions the system was never designed to handle. 

`SemanticDriftDetector` monitors your production queries and compares them to your baseline dataset. It alerts you when the distribution of topics changes, causing performance to drop.

## Tutorial: Catching Drift

Imagine you built a RAG bot for HR policies (leave, benefits, payroll). Six months later, the company introduces a new "Remote Work" policy, but no one updated the bot. Users start asking about remote work.

Here is how to detect this drift.

```python
from rageval import SemanticDriftDetector
from rageval.judges.heuristic import HeuristicJudge

# We use the fast, free heuristic judge for embeddings
detector = SemanticDriftDetector(embedding_judge=HeuristicJudge())

# 1. Fit the detector on your known good baseline (e.g., your test set)
baseline_queries = [
    "How many PTO days do I get?",
    "When is the next payday?",
    "How do I enroll in health insurance?",
]
detector.set_baseline(baseline_queries)
# We must also set the knowledge base for coverage math
detector.set_knowledge_base(["Your knowledge base docs..."])

# 2. Analyze recent production queries
recent_queries = [
    "How many PTO days do I get?",        # Normal
    "Can I work from a coffee shop?",     # Drift!
    "What is the remote work equipment stipend?", # Drift!
    "Do I need to come into the office on Tuesdays?", # Drift!
]

report = detector.detect(recent_queries)

print(f"Drift Score: {report.drift_score}")
print("New Clusters Identified:")
print(report.summary())
```

## The Report Output

```text
Drift Score: 0.75

New Clusters Identified:
- "Remote work locations and policies" (3 queries)

Predicted Performance Degradation:
Expect a 15-20% drop in ContextPrecision, as the retriever likely lacks documents for "Remote work locations and policies".
```

### When to use this

Run `SemanticDriftDetector` on a cron job every week using the last 7 days of production queries. If the drift score exceeds `0.4`, it is time to:
1. Review the new clusters.
2. Add the missing documents to your vector database.
3. Add the new queries to your evaluation dataset.
