# tests/integration/test_context_precision_real.py
"""
Real LLM call tests for ContextPrecision metric.
"""

import pytest
from rageval.metrics.context_precision import ContextPrecision


class TestContextPrecisionReal:

    def test_all_relevant_docs_scores_high(self, judge, perfect_sample):
        """All relevant docs must produce a high precision score."""
        metric = ContextPrecision(judge=judge, threshold=0.7)
        result = metric.score(perfect_sample)

        print(f"\nAll relevant docs precision: {result.score:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Evidence: {result.evidence}")

        assert result.score >= 0.5, (
            f"Expected score >= 0.5 for all-relevant docs, got {result.score}"
        )

    def test_noisy_docs_score_low(self, judge, noisy_retrieval_sample):
        """Docs with obvious noise must produce a lower precision score."""
        metric = ContextPrecision(judge=judge, threshold=0.7)
        result = metric.score(noisy_retrieval_sample)

        print(f"\nNoisy docs precision: {result.score:.3f}")
        print(f"Evidence: {result.evidence}")

        assert result.score < 0.9, (
            f"Expected score < 0.9 for noisy retrieval, got {result.score}.\n"
            f"The rainforest and French cuisine docs should be flagged as irrelevant."
        )
        assert len(result.evidence) >= 1, (
            "Expected at least one irrelevant doc in evidence"
        )

    def test_noise_docs_identified_in_evidence(self, judge, noisy_retrieval_sample):
        """The specific irrelevant docs must appear in evidence."""
        metric = ContextPrecision(judge=judge, threshold=0.7)
        result = metric.score(noisy_retrieval_sample)

        print(f"\nEvidence items: {result.evidence}")

        evidence_text = " ".join(result.evidence).lower()
        assert any(term in evidence_text for term in ["amazon", "rainforest", "french", "cuisine"]), (
            f"Expected rainforest or French cuisine noise to appear in evidence.\n"
            f"Got: {result.evidence}"
        )

    def test_score_reflects_noise_ratio(self, judge):
        """More noise docs should produce lower precision score."""
        from rageval.core.sample import RAGSample

        clean_sample = RAGSample(
            query="What is machine learning?",
            retrieved_docs=[
                "Machine learning is a subset of AI that enables systems to learn from data.",
                "Supervised learning uses labeled data to train models.",
            ],
            answer="Machine learning enables systems to learn from data.",
        )

        noisy_sample = RAGSample(
            query="What is machine learning?",
            retrieved_docs=[
                "Machine learning is a subset of AI that enables systems to learn from data.",
                "The Great Wall of China stretches over 13,000 miles.",
                "Italian pasta comes in hundreds of different shapes and sizes.",
                "Machine learning models require large amounts of training data.",
            ],
            answer="Machine learning enables systems to learn from data.",
        )

        metric = ContextPrecision(judge=judge, threshold=0.7)
        clean_result = metric.score(clean_sample)
        noisy_result = metric.score(noisy_sample)

        print(f"\nClean precision: {clean_result.score:.3f}")
        print(f"Noisy precision: {noisy_result.score:.3f}")

        assert clean_result.score >= noisy_result.score, (
            f"Clean sample ({clean_result.score:.3f}) should score >= "
            f"noisy sample ({noisy_result.score:.3f})"
        )
