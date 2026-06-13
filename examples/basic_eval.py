"""
rageval quickstart example.

This file shows the complete workflow:
1. Construct RAGSamples from your pipeline output
2. Choose a judge and metrics
3. Evaluate and read the results

Replace the sample data with real output from your RAG pipeline.
Run with: python examples/basic_eval.py
"""

from unittest.mock import MagicMock
from rageval.core.sample import RAGSample
from rageval.core.pipeline import evaluate, batch_evaluate, summary
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.reporters.json_csv import print_summary, to_json


# ── Replace this with your real judge ───────────────────────────────────────
# from rageval.judges.openai_judge import OpenAIJudge
# from rageval.judges.anthropic_judge import AnthropicJudge
# judge = OpenAIJudge(model="gpt-4o-mini")
# judge = AnthropicJudge(model="claude-haiku-4-5")

# Using mock judge for demonstration — replace with real judge above
judge = MagicMock()
judge.complete_json.side_effect = [
    # faithfulness: claims
    {"claims": [
        "The Eiffel Tower is 330 meters tall.",
        "It was completed in 1889.",
        "It was designed by Leonardo da Vinci.",
    ]},
    # faithfulness: verifications
    {"verifications": [
        {"claim": "The Eiffel Tower is 330 meters tall.", "supported": True,
         "reason": "Context confirms 330 meters."},
        {"claim": "It was completed in 1889.", "supported": True,
         "reason": "Context confirms 1889."},
        {"claim": "It was designed by Leonardo da Vinci.", "supported": False,
         "reason": "Context says Gustave Eiffel, not da Vinci."},
    ]},
    # context_precision: doc 1
    {"is_relevant": True, "reason": "Directly about the Eiffel Tower."},
    # context_precision: doc 2
    {"is_relevant": False, "reason": "About French cuisine, not the tower."},
    # answer_relevancy: reverse questions
    {"questions": [
        "How tall is the Eiffel Tower?",
        "What is the height of the Eiffel Tower?",
        "What are the dimensions of the Eiffel Tower?",
    ]},
]

# ── Build your sample ────────────────────────────────────────────────────────
sample = RAGSample(
    query="How tall is the Eiffel Tower and when was it built?",
    retrieved_docs=[
        "The Eiffel Tower stands 330 meters tall and was completed in 1889. "
        "It was designed by engineer Gustave Eiffel.",
        "French cuisine is known for its sophistication and use of fresh ingredients.",
    ],
    answer=(
        "The Eiffel Tower is 330 meters tall and was completed in 1889. "
        "It was designed by Leonardo da Vinci."
    ),
)

# ── Run evaluation ───────────────────────────────────────────────────────────
metrics = [
    Faithfulness(judge=judge, threshold=0.8),
    ContextPrecision(judge=judge, threshold=0.7),
    AnswerRelevancy(judge=judge, threshold=0.7),
]

result = evaluate(sample=sample, metrics=metrics)

# ── Read the results ─────────────────────────────────────────────────────────
print(result.summary())

print("\nFailed metrics:")
for r in result.failed_metrics():
    print(f"  {r.metric_name}: {r.score:.2f}")
    for e in r.evidence:
        print(f"    -> {e}")

# ── Save to JSON ─────────────────────────────────────────────────────────────
to_json([result], "example_results.json")
print("\nResults saved to example_results.json")