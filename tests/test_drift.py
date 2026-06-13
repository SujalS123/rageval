# tests/test_drift.py

import pytest
from rageval.drift import SemanticDriftDetector


# All tests use the n-gram fallback (no embedding_judge) for speed and zero deps.


def test_identical_distributions_score_zero_drift():
    """Same queries as baseline → drift_score should be 0."""
    queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What year was the Eiffel Tower built?",
    ]
    kb = [
        "Paris is the capital of France and a major European city.",
        "Photosynthesis is the process by which plants convert sunlight to energy.",
        "The Eiffel Tower was built in 1889 by Gustave Eiffel.",
    ]

    detector = SemanticDriftDetector(coverage_threshold=0.0)  # everything is covered
    detector.set_baseline(queries)
    detector.set_knowledge_base(kb)

    report = detector.detect(queries)  # same as baseline
    assert report.drift_score == 0.0


def test_completely_different_queries_score_high_drift():
    """Queries about topics totally absent from the KB → high uncovered fraction."""
    kb = ["The Eiffel Tower stands 330 metres tall."]

    # Queries about completely unrelated topics
    recent = [
        "quantum computing error correction protocols",
        "CRISPR gene editing techniques",
        "blockchain consensus mechanisms",
        "neural architecture search methods",
    ]

    detector = SemanticDriftDetector(coverage_threshold=0.99)  # very strict
    detector.set_baseline([])
    detector.set_knowledge_base(kb)

    report = detector.detect(recent)
    assert report.drift_score > 0.0
    assert report.uncovered_query_count > 0


def test_uncovered_queries_appear_in_report():
    """Uncovered queries must be reflected in uncovered_query_count."""
    kb = ["Paris is the capital of France."]
    recent = [
        "What is the capital of France?",  # covered
        "explain quantum entanglement in detail",  # not covered
    ]

    detector = SemanticDriftDetector(coverage_threshold=0.8)
    detector.set_baseline([])
    detector.set_knowledge_base(kb)

    report = detector.detect(recent)
    # At minimum the quantum query should be uncovered
    assert report.uncovered_query_count >= 1


def test_new_topics_identified_in_clusters():
    """Uncovered queries should produce at least one new_topic_cluster entry."""
    kb = ["France is a country in Western Europe."]
    recent = [
        "quantum entanglement explained simply",
        "quantum computing error rates",
        "quantum decoherence problems",
    ]

    detector = SemanticDriftDetector(coverage_threshold=0.9)
    detector.set_baseline([])
    detector.set_knowledge_base(kb)

    report = detector.detect(recent)
    if report.uncovered_query_count > 0:
        assert len(report.new_topic_clusters) > 0


def test_predicted_degradation_is_drift_times_constant():
    """predicted_faithfulness_degradation = drift_score * 0.35."""
    kb = ["irrelevant content about cooking recipes"]
    recent = ["quantum physics superposition principles"] * 5

    detector = SemanticDriftDetector(coverage_threshold=0.99)
    detector.set_baseline([])
    detector.set_knowledge_base(kb)

    report = detector.detect(recent)
    expected = round(report.drift_score * 0.35, 4)
    assert report.predicted_faithfulness_degradation == pytest.approx(expected, abs=0.001)


def test_empty_recent_queries_returns_zero_drift():
    """No recent queries → drift_score = 0, no errors."""
    detector = SemanticDriftDetector()
    detector.set_baseline(["some query"])
    detector.set_knowledge_base(["some document"])

    report = detector.detect([])
    assert report.drift_score == 0.0
    assert report.uncovered_query_count == 0
