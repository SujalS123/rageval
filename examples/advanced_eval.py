"""
Advanced rageval example showing:
- Batch evaluation
- RunTracker for regression tracking
- ExplainabilityReporter for HTML output
- ConsistencyAnalyzer for paraphrase testing
"""

import os
from rageval import RAGSample, batch_evaluate, summary, RunTracker, ConsistencyAnalyzer, ExplainabilityReporter
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.judges.heuristic import HeuristicJudge

# In a real scenario, use OpenAIJudge or AnthropicJudge.
# We use HeuristicJudge here so this script runs without an API key.
judge = HeuristicJudge()

def run_advanced_eval():
    print("--- 1. Batch Evaluation ---")
    samples = [
        RAGSample(
            query="What is Python?",
            retrieved_docs=["Python is a high-level programming language."],
            answer="Python is a programming language."
        ),
        RAGSample(
            query="Who wrote Romeo and Juliet?",
            retrieved_docs=["Romeo and Juliet is a tragedy written by William Shakespeare."],
            answer="Romeo and Juliet was written by Charles Dickens." # Hallucination
        )
    ]

    metrics = [
        Faithfulness(judge=judge, threshold=0.8),
        ContextPrecision(judge=judge, threshold=0.7),
        AnswerRelevancy(judge=judge, threshold=0.7)
    ]

    results = batch_evaluate(samples, metrics, show_progress=True)
    stats = summary(results)
    
    print("\nBatch Summary:")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Overall Pass Rate: {stats['overall_pass_rate']}")
    
    print("\n--- 2. RunTracker ---")
    tracker = RunTracker()
    tracker.save_run("advanced-run-v1", results)
    print("Run saved to local SQLite tracker (.rageval/runs.db)")
    
    # Show history
    runs = tracker.list_runs()
    print(f"Tracked runs: {[r['name'] for r in runs]}")

    print("\n--- 3. ExplainabilityReporter ---")
    # Generate HTML report for the first failed sample
    failed_results = [r for r in results if not r.passed]
    if failed_results:
        reporter = ExplainabilityReporter()
        report_html = reporter.generate_report(failed_results[0])
        report_path = "advanced_eval_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        print(f"HTML explainability report saved to {report_path}")

    print("\n--- 4. ConsistencyAnalyzer ---")
    # A mock pipeline function to demonstrate consistency analysis
    def mock_pipeline(query: str) -> RAGSample:
        return RAGSample(
            query=query,
            retrieved_docs=["Python is a high-level language."],
            answer="Python is a programming language." if "Python" in query else "I don't know."
        )

    analyzer = ConsistencyAnalyzer(judge=judge, embedding_judge=judge)
    report = analyzer.analyze(
        query="What is Python?",
        paraphrases=["Explain Python to me.", "What exactly is Python?"],
        pipeline_fn=mock_pipeline
    )
    
    print(f"Consistency Score: {report.consistency_score}")
    print(f"Root Cause Hypothesis: {report.root_cause_hypothesis}")

if __name__ == "__main__":
    run_advanced_eval()
