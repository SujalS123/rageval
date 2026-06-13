# tests/integration/test_pipeline_real.py
"""
Real LLM call tests for the full evaluate() and batch_evaluate() pipeline.
"""

import pytest
import time
from rageval import evaluate, batch_evaluate
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.core.sample import RAGSample


class TestPipelineReal:

    def test_evaluate_returns_all_metrics(self, judge, embedding_judge, perfect_sample):
        """evaluate() must return results for every metric provided."""
        metrics = [
            Faithfulness(judge=judge, threshold=0.8),
            ContextPrecision(judge=judge, threshold=0.7),
            AnswerRelevancy(judge=judge, threshold=0.7, embedding_judge=embedding_judge),
        ]

        result = evaluate(sample=perfect_sample, metrics=metrics)

        print(result.summary())

        assert "faithfulness" in result.metric_results
        assert "context_precision" in result.metric_results
        assert "answer_relevancy" in result.metric_results
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms > 0

    def test_evaluate_overall_score_is_weighted_average(self, judge, embedding_judge, perfect_sample):
        """Overall score must be the average of metric scores."""
        metrics = [
            Faithfulness(judge=judge, threshold=0.8),
            ContextPrecision(judge=judge, threshold=0.7),
        ]

        result = evaluate(sample=perfect_sample, metrics=metrics)

        faithfulness_score = result.metric_results["faithfulness"].score
        precision_score = result.metric_results["context_precision"].score
        expected_overall = (faithfulness_score + precision_score) / 2

        assert abs(result.overall_score - expected_overall) < 0.01, (
            f"Overall score {result.overall_score} should be average of "
            f"faithfulness {faithfulness_score} and precision {precision_score} = {expected_overall}"
        )

    def test_one_metric_failure_does_not_crash_others(self, judge, embedding_judge):
        """If one metric crashes, others must still return results."""
        from rageval.metrics.base import BaseMetric
        from rageval.core.result import MetricResult

        class CrashingMetric(BaseMetric):
            name = "crashing"
            def score(self, sample):
                raise RuntimeError("Intentional crash for testing")

        metrics = [
            CrashingMetric(judge=judge, threshold=0.5),
            Faithfulness(judge=judge, threshold=0.8),
        ]

        sample = RAGSample(
            query="test query",
            retrieved_docs=["test context document"],
            answer="test answer",
        )

        result = evaluate(sample=sample, metrics=metrics)

        assert "crashing" in result.metric_results
        assert result.metric_results["crashing"].score == 0.0
        assert "Intentional crash" in result.metric_results["crashing"].reasoning
        assert "faithfulness" in result.metric_results
        assert result.metric_results["faithfulness"].score > 0

    def test_batch_evaluate_returns_correct_count(self, judge, embedding_judge,
                                                   perfect_sample, hallucination_sample):
        """batch_evaluate must return one result per input sample."""
        metrics = [Faithfulness(judge=judge, threshold=0.8)]
        samples = [perfect_sample, hallucination_sample, perfect_sample]

        results = batch_evaluate(samples=samples, metrics=metrics, show_progress=False)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    def test_batch_results_in_same_order(self, judge, perfect_sample, hallucination_sample):
        """batch_evaluate must preserve input order."""
        metrics = [Faithfulness(judge=judge, threshold=0.8)]
        samples = [perfect_sample, hallucination_sample]

        results = batch_evaluate(samples=samples, metrics=metrics, show_progress=False)

        perfect_score = results[0].metric_results["faithfulness"].score
        hallucination_score = results[1].metric_results["faithfulness"].score

        print(f"\nPerfect sample (index 0): {perfect_score:.3f}")
        print(f"Hallucination sample (index 1): {hallucination_score:.3f}")

        assert perfect_score >= hallucination_score, (
            f"Perfect sample ({perfect_score:.3f}) should score >= "
            f"hallucination sample ({hallucination_score:.3f})"
        )

    def test_latency_is_recorded(self, judge, perfect_sample):
        """EvalResult must record real latency."""
        metrics = [Faithfulness(judge=judge, threshold=0.8)]
        result = evaluate(sample=perfect_sample, metrics=metrics)

        assert result.latency_ms > 100, (
            f"Latency {result.latency_ms}ms seems too low for a real LLM call. "
            "Is this using a real judge?"
        )
        assert result.latency_ms < 120000, (
            f"Latency {result.latency_ms}ms is over 2 minutes — something is wrong"
        )

    def test_weighted_overall_score(self, judge, embedding_judge, perfect_sample):
        """Custom weights must affect the overall score correctly."""
        metrics = [
            Faithfulness(judge=judge, threshold=0.8),
            ContextPrecision(judge=judge, threshold=0.7),
        ]

        result_equal = evaluate(sample=perfect_sample, metrics=metrics)
        result_weighted = evaluate(
            sample=perfect_sample,
            metrics=metrics,
            weights={"faithfulness": 3.0, "context_precision": 1.0},
        )

        faithfulness_score = result_equal.metric_results["faithfulness"].score
        precision_score = result_equal.metric_results["context_precision"].score

        expected_weighted = (faithfulness_score * 3 + precision_score * 1) / 4

        print(f"\nEqual weights overall: {result_equal.overall_score:.3f}")
        print(f"Weighted overall: {result_weighted.overall_score:.3f}")
        print(f"Expected weighted: {expected_weighted:.3f}")

        assert abs(result_weighted.overall_score - expected_weighted) < 0.01
