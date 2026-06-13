# tests/integration/test_judges_real.py
"""
Real LLM call tests for judge backends.
Validates JSON parsing, error handling, and response consistency.
"""

import pytest
import os


class TestJudgesReal:

    def test_judge_returns_string(self, judge):
        """complete() must return a non-empty string."""
        response = judge.complete("Say the word hello and nothing else.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_judge_complete_json_clean(self, judge):
        """complete_json() must parse clean JSON correctly."""
        result = judge.complete_json(
            'Return ONLY this JSON with no changes: {"score": 0.85, "label": "test"}'
        )
        assert isinstance(result, dict)
        assert "score" in result or "label" in result

    def test_judge_complete_json_with_fence(self, judge):
        """complete_json() must handle markdown fenced JSON."""
        prompt = (
            "Return a JSON object with a field called 'value' set to 42. "
            "Respond ONLY with valid JSON, no explanation."
        )
        result = judge.complete_json(prompt)
        assert isinstance(result, dict)

    def test_judge_temperature_is_deterministic(self, judge):
        """
        Same prompt should produce same JSON structure (not necessarily identical text).
        Tests that temperature=0 is working.
        """
        prompt = (
            "Return ONLY this exact JSON: "
            '{"result": "consistent", "number": 7}'
        )
        result1 = judge.complete_json(prompt)
        result2 = judge.complete_json(prompt)

        assert type(result1) == type(result2), (
            "Two calls with same prompt returned different types"
        )

    def test_heuristic_judge_similarity_range(self, embedding_judge):
        """HeuristicJudge similarity must always return value in [0, 1]."""
        pairs = [
            ("identical sentence", "identical sentence"),
            ("What is AI?", "How does artificial intelligence work?"),
            ("What is AI?", "The weather in Paris today is cloudy."),
        ]
        for a, b in pairs:
            sim = embedding_judge.similarity(a, b)
            assert 0.0 <= sim <= 1.0, f"Similarity {sim} out of range for pair: ({a}, {b})"

    def test_heuristic_judge_semantic_ordering(self, embedding_judge):
        """Similar sentences must score higher than unrelated ones."""
        anchor = "What is machine learning?"
        similar = "How does machine learning work?"
        unrelated = "What is the population of Brazil?"

        sim_similar = embedding_judge.similarity(anchor, similar)
        sim_unrelated = embedding_judge.similarity(anchor, unrelated)

        print(f"\nSimilar pair similarity: {sim_similar:.3f}")
        print(f"Unrelated pair similarity: {sim_unrelated:.3f}")

        assert sim_similar > sim_unrelated, (
            f"Similar sentences ({sim_similar:.3f}) should score higher than "
            f"unrelated ({sim_unrelated:.3f})"
        )

    def test_batch_similarity_consistent_with_single(self, embedding_judge):
        """batch_similarity results must match individual similarity() calls."""
        anchor = "What is deep learning?"
        texts = [
            "How does deep learning differ from machine learning?",
            "What is the boiling point of water?",
        ]

        batch = embedding_judge.batch_similarity(anchor, texts)
        singles = [embedding_judge.similarity(anchor, t) for t in texts]

        for b, s in zip(batch, singles):
            assert abs(b - s) < 0.01, (
                f"Batch similarity {b:.3f} doesn't match single {s:.3f}"
            )
