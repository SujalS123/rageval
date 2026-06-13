# tests/test_context_recall.py

import pytest
from unittest.mock import MagicMock
from rageval.metrics.context_recall import ContextRecall
from rageval.core.sample import RAGSample


def make_mock_judge(extraction_result: dict, coverage_result: dict):
    judge = MagicMock()
    judge.complete_json.side_effect = [extraction_result, coverage_result]
    return judge


def test_perfect_recall():
    """All ground truth claims found in context — should score 1.0 and pass."""
    judge = make_mock_judge(
        extraction_result={"claims": ["The Eiffel Tower is 330 metres tall."]},
        coverage_result={
            "coverage": [
                {
                    "claim": "The Eiffel Tower is 330 metres tall.",
                    "found": True,
                    "reason": "Context explicitly states the height.",
                }
            ]
        },
    )

    metric = ContextRecall(judge=judge, threshold=0.8)
    sample = RAGSample(
        query="How tall is the Eiffel Tower?",
        retrieved_docs=["The Eiffel Tower stands 330 metres tall and was completed in 1889."],
        answer="The Eiffel Tower is 330 metres tall.",
        ground_truth="The Eiffel Tower is 330 metres tall.",
    )

    result = metric.score(sample)

    assert result.score == 1.0
    assert result.passed is True
    assert result.evidence == []
    assert "All" in result.reasoning


def test_zero_recall():
    """No ground truth claims found in context — should score 0.0 and fail."""
    judge = make_mock_judge(
        extraction_result={
            "claims": [
                "The treaty was signed in 1847.",
                "It was ratified by 12 nations.",
            ]
        },
        coverage_result={
            "coverage": [
                {
                    "claim": "The treaty was signed in 1847.",
                    "found": False,
                    "reason": "Context does not mention the treaty date.",
                },
                {
                    "claim": "It was ratified by 12 nations.",
                    "found": False,
                    "reason": "Context does not mention ratification.",
                },
            ]
        },
    )

    metric = ContextRecall(judge=judge, threshold=0.8)
    sample = RAGSample(
        query="Tell me about the treaty.",
        retrieved_docs=["A peace agreement was reached between the two parties."],
        answer="The treaty was signed in 1847 and ratified by 12 nations.",
        ground_truth="The treaty was signed in 1847. It was ratified by 12 nations.",
    )

    result = metric.score(sample)

    assert result.score == 0.0
    assert result.passed is False
    assert len(result.evidence) == 2


def test_partial_recall():
    """Two claims, one found, one missing — should score 0.5."""
    judge = make_mock_judge(
        extraction_result={
            "claims": [
                "Python was created by Guido van Rossum.",
                "Python was first released in 1991.",
            ]
        },
        coverage_result={
            "coverage": [
                {
                    "claim": "Python was created by Guido van Rossum.",
                    "found": True,
                    "reason": "Context mentions Guido van Rossum as creator.",
                },
                {
                    "claim": "Python was first released in 1991.",
                    "found": False,
                    "reason": "Context does not mention the release year.",
                },
            ]
        },
    )

    metric = ContextRecall(judge=judge, threshold=0.8)
    sample = RAGSample(
        query="Tell me about Python.",
        retrieved_docs=["Python is a programming language created by Guido van Rossum."],
        answer="Python was created by Guido van Rossum and first released in 1991.",
        ground_truth="Python was created by Guido van Rossum. Python was first released in 1991.",
    )

    result = metric.score(sample)

    assert result.score == 0.5
    assert result.passed is False
    assert len(result.evidence) == 1


def test_evidence_contains_missing_facts():
    """Evidence must list the specific claims the retriever failed to cover."""
    judge = make_mock_judge(
        extraction_result={"claims": ["Revenue grew by 42% in Q3."]},
        coverage_result={
            "coverage": [
                {
                    "claim": "Revenue grew by 42% in Q3.",
                    "found": False,
                    "reason": "Context mentions growth but not the specific percentage.",
                }
            ]
        },
    )

    metric = ContextRecall(judge=judge, threshold=0.8)
    sample = RAGSample(
        query="What was the revenue growth?",
        retrieved_docs=["The company reported strong revenue growth this quarter."],
        answer="Revenue grew by 42% in Q3.",
        ground_truth="Revenue grew by 42% in Q3.",
    )

    result = metric.score(sample)

    assert result.passed is False
    assert any("42%" in e for e in result.evidence)
    assert any("MISSING" in e for e in result.evidence)


def test_validate_catches_missing_ground_truth():
    """Metric must raise ValueError when ground_truth is None."""
    judge = MagicMock()
    metric = ContextRecall(judge=judge, threshold=0.8)

    sample = RAGSample(
        query="What is the boiling point of water?",
        retrieved_docs=["Water boils at 100 degrees Celsius."],
        answer="Water boils at 100 degrees Celsius.",
        ground_truth=None,
    )

    with pytest.raises(ValueError, match="ground_truth"):
        metric.score(sample)
