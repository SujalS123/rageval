# tests/integration/test_answer_relevancy_real.py
"""
Real LLM call tests for AnswerRelevancy metric.
"""

import pytest
from rageval.metrics.answer_relevancy import AnswerRelevancy


class TestAnswerRelevancyReal:

    def test_relevant_answer_scores_high(self, judge, embedding_judge, perfect_sample):
        """A directly relevant answer must score above 0.7."""
        metric = AnswerRelevancy(judge=judge, threshold=0.7, embedding_judge=embedding_judge)
        result = metric.score(perfect_sample)

        print(f"\nRelevant answer score: {result.score:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Evidence: {result.evidence}")

        assert result.score >= 0.6, (
            f"Expected score >= 0.6 for relevant answer, got {result.score}"
        )

    def test_off_topic_answer_scores_low(self, judge, embedding_judge, off_topic_answer_sample):
        """An off-topic answer must score lower than a relevant answer."""
        metric = AnswerRelevancy(judge=judge, threshold=0.7, embedding_judge=embedding_judge)
        result = metric.score(off_topic_answer_sample)

        print(f"\nOff-topic answer score: {result.score:.3f}")
        print(f"Reasoning: {result.reasoning}")

        assert result.score < 0.9, (
            f"Expected score < 0.9 for off-topic answer, got {result.score}"
        )

    def test_relevant_beats_off_topic(self, judge, embedding_judge, perfect_sample, off_topic_answer_sample):
        """Relevant answer must score higher than off-topic answer."""
        metric = AnswerRelevancy(judge=judge, threshold=0.7, embedding_judge=embedding_judge)

        relevant_result = metric.score(perfect_sample)
        off_topic_result = metric.score(off_topic_answer_sample)

        print(f"\nRelevant score: {relevant_result.score:.3f}")
        print(f"Off-topic score: {off_topic_result.score:.3f}")

        assert relevant_result.score > off_topic_result.score, (
            f"Relevant answer ({relevant_result.score:.3f}) should score higher than "
            f"off-topic answer ({off_topic_result.score:.3f})"
        )

    def test_evidence_shows_generated_questions(self, judge, embedding_judge, perfect_sample):
        """Evidence must show the generated questions with similarity scores."""
        metric = AnswerRelevancy(judge=judge, threshold=0.7, embedding_judge=embedding_judge)
        result = metric.score(perfect_sample)

        assert len(result.evidence) >= 1, "Evidence must contain generated questions"
        evidence_text = " ".join(result.evidence)
        assert "similarity" in evidence_text.lower() or "generated" in evidence_text.lower(), (
            f"Evidence should mention generated questions and similarity scores.\n"
            f"Got: {result.evidence}"
        )
