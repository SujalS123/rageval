# tests/integration/test_json_parsing_real.py
"""
Tests that validate JSON parsing robustness with real LLM outputs.
This is the most common source of production failures.
"""

import pytest
from rageval import evaluate
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.core.sample import RAGSample


class TestJSONParsingReal:

    def test_faithfulness_handles_real_llm_json(self, judge):
        """
        Run faithfulness on a sample and verify the JSON from the LLM
        was parsed correctly into a MetricResult with expected fields.
        """
        sample = RAGSample(
            query="What year was Python created?",
            retrieved_docs=["Python was created by Guido van Rossum and released in 1991."],
            answer="Python was created by Guido van Rossum in 1991.",
        )

        metric = Faithfulness(judge=judge, threshold=0.8)
        result = metric.score(sample)

        print(f"\nParsing test faithfulness score: {result.score:.3f}")
        print(f"Reasoning: {result.reasoning}")

        assert result is not None
        assert isinstance(result.score, float)
        assert isinstance(result.reasoning, str)
        assert isinstance(result.evidence, list)
        assert result.score > 0.5, (
            f"A clearly faithful answer should score above 0.5, got {result.score}\n"
            f"Reasoning: {result.reasoning}"
        )

    def test_context_precision_handles_real_llm_json(self, judge):
        """ContextPrecision JSON parsing works with real LLM output."""
        sample = RAGSample(
            query="Who invented the telephone?",
            retrieved_docs=[
                "Alexander Graham Bell is credited with inventing the telephone in 1876.",
                "The telephone revolutionized long-distance communication.",
            ],
            answer="Alexander Graham Bell invented the telephone in 1876.",
        )

        metric = ContextPrecision(judge=judge, threshold=0.7)
        result = metric.score(sample)

        print(f"\nContext precision JSON parsing score: {result.score:.3f}")

        assert result is not None
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_answer_relevancy_handles_real_llm_json(self, judge, embedding_judge):
        """AnswerRelevancy JSON parsing works with real LLM output."""
        sample = RAGSample(
            query="What is photosynthesis?",
            retrieved_docs=[
                "Photosynthesis is the process plants use to convert sunlight into energy."
            ],
            answer="Photosynthesis is how plants convert sunlight into glucose using chlorophyll.",
        )

        metric = AnswerRelevancy(
            judge=judge, threshold=0.7, embedding_judge=embedding_judge
        )
        result = metric.score(sample)

        print(f"\nAnswer relevancy JSON parsing score: {result.score:.3f}")
        print(f"Evidence: {result.evidence}")

        assert result is not None
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert len(result.evidence) > 0

    def test_no_metric_raises_uncaught_exception(self, judge, embedding_judge):
        """
        No metric should raise an uncaught exception on any valid RAGSample.
        All errors must be caught and returned as score=0.0 in MetricResult.
        """
        edge_cases = [
            RAGSample(
                query="a",
                retrieved_docs=["single character query"],
                answer="single char",
            ),
            RAGSample(
                query="What is this? " * 20,
                retrieved_docs=["short doc"],
                answer="short answer",
            ),
            RAGSample(
                query="Normal query",
                retrieved_docs=["doc " * 200],
                answer="Normal answer",
            ),
        ]

        metrics = [
            Faithfulness(judge=judge, threshold=0.8),
            ContextPrecision(judge=judge, threshold=0.7),
            AnswerRelevancy(
                judge=judge, threshold=0.7, embedding_judge=embedding_judge
            ),
        ]

        for i, sample in enumerate(edge_cases):
            try:
                result = evaluate(sample=sample, metrics=metrics)
                print(f"\nEdge case {i}: overall={result.overall_score:.3f}")
                assert result is not None
            except Exception as e:
                pytest.fail(
                    f"Edge case {i} raised uncaught exception: "
                    f"{type(e).__name__}: {e}"
                )
