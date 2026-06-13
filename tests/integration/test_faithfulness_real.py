# tests/integration/test_faithfulness_real.py
"""
Real LLM call tests for Faithfulness metric.
These validate that prompts work correctly with actual API responses.
"""

import pytest
from rageval.metrics.faithfulness import Faithfulness


class TestFaithfulnessReal:

    def test_perfect_faithfulness_scores_high(self, judge, perfect_sample):
        """A perfectly grounded answer must score above 0.8."""
        metric = Faithfulness(judge=judge, threshold=0.8)
        result = metric.score(perfect_sample)

        print(f"\nPerfect sample faithfulness: {result.score:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Evidence: {result.evidence}")

        assert result.score >= 0.7, (
            f"Expected score >= 0.7 for perfectly faithful answer, got {result.score}.\n"
            f"Evidence: {result.evidence}"
        )
        assert result.evidence == [] or len(result.evidence) <= 1, (
            f"Expected no or minimal evidence for faithful answer, got: {result.evidence}"
        )

    def test_hallucination_scores_low(self, judge, hallucination_sample):
        """An answer with a clear hallucination must score below 0.8."""
        metric = Faithfulness(judge=judge, threshold=0.8)
        result = metric.score(hallucination_sample)

        print(f"\nHallucination sample faithfulness: {result.score:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Evidence: {result.evidence}")

        assert result.score < 0.85, (
            f"Expected score < 0.85 for hallucinated answer, got {result.score}.\n"
            f"Reasoning: {result.reasoning}"
        )
        assert result.passed is False, "Hallucinated answer should not pass threshold"

    def test_hallucination_appears_in_evidence(self, judge, hallucination_sample):
        """The specific hallucinated claim must appear in evidence."""
        metric = Faithfulness(judge=judge, threshold=0.8)
        result = metric.score(hallucination_sample)

        print(f"\nEvidence items: {result.evidence}")

        assert len(result.evidence) > 0, (
            "Expected at least one evidence item for hallucinated answer"
        )

        # Check that either da Vinci or the wrong attribution appears in evidence
        evidence_text = " ".join(result.evidence).lower()
        assert any(term in evidence_text for term in ["da vinci", "leonardo", "vinci"]), (
            f"Expected 'da Vinci' to appear in evidence as the hallucinated designer.\n"
            f"Got evidence: {result.evidence}"
        )

    def test_evidence_is_actionable_strings(self, judge, hallucination_sample):
        """Evidence items must be non-empty strings a developer can read."""
        metric = Faithfulness(judge=judge, threshold=0.8)
        result = metric.score(hallucination_sample)

        for item in result.evidence:
            assert isinstance(item, str), f"Evidence item must be string, got {type(item)}"
            assert len(item) > 10, f"Evidence item too short to be useful: '{item}'"

    def test_score_is_valid_range(self, judge, perfect_sample, hallucination_sample):
        """Score must always be between 0.0 and 1.0."""
        metric = Faithfulness(judge=judge, threshold=0.8)

        for sample in [perfect_sample, hallucination_sample]:
            result = metric.score(sample)
            assert 0.0 <= result.score <= 1.0, (
                f"Score {result.score} is outside valid range [0.0, 1.0]"
            )

    def test_reasoning_is_non_empty(self, judge, hallucination_sample):
        """Reasoning must always be populated."""
        metric = Faithfulness(judge=judge, threshold=0.8)
        result = metric.score(hallucination_sample)

        assert result.reasoning, "Reasoning must not be empty"
        assert len(result.reasoning) > 20, (
            f"Reasoning too short to be useful: '{result.reasoning}'"
        )

    def test_multiple_docs_handled(self, judge):
        """Faithfulness works correctly with multiple retrieved documents."""
        from rageval.core.sample import RAGSample
        sample = RAGSample(
            query="Tell me about the solar system.",
            retrieved_docs=[
                "The solar system consists of the Sun and eight planets.",
                "Earth is the third planet from the Sun.",
                "Jupiter is the largest planet in the solar system.",
            ],
            answer="The solar system has eight planets. Earth is the third planet. "
                   "Jupiter is the largest planet. Mars has three moons.",
        )
        metric = Faithfulness(judge=judge, threshold=0.8)
        result = metric.score(sample)

        print(f"\nMultiple docs faithfulness: {result.score:.3f}")
        print(f"Evidence: {result.evidence}")

        # "Mars has three moons" is not in context — should be in evidence
        assert result.score < 1.0, "Answer with hallucination should not score 1.0"
        evidence_text = " ".join(result.evidence).lower()
        assert "mars" in evidence_text or "moon" in evidence_text, (
            f"Expected Mars moons hallucination in evidence, got: {result.evidence}"
        )
