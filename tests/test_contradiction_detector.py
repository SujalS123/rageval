# tests/test_contradiction_detector.py

import pytest
from unittest.mock import MagicMock
from rageval.metrics.contradiction_detector import ContradictionDetector
from rageval.core.sample import RAGSample


def make_mock_judge(detection_result: dict):
    judge = MagicMock()
    judge.complete_json.return_value = detection_result
    return judge


def make_sample(answer="The treaty was signed in 1850."):
    return RAGSample(
        query="When was the treaty signed?",
        retrieved_docs=["The treaty was signed in 1847 after two years of negotiation."],
        answer=answer,
    )


def test_no_contradictions_scores_1():
    """Answer with no contradictions must score 1.0 and pass."""
    judge = make_mock_judge({"contradictions": []})
    metric = ContradictionDetector(judge=judge, threshold=0.9)
    result = metric.score(make_sample(answer="The treaty was signed in 1847."))

    assert result.score == 1.0
    assert result.passed is True
    assert result.evidence == []
    assert "No contradictions" in result.reasoning


def test_direct_contradiction_scores_low():
    """A single direct contradiction in a short answer must score below threshold."""
    judge = make_mock_judge({
        "contradictions": [{
            "claim": "The treaty was signed in 1850.",
            "context_says": "The treaty was signed in 1847.",
            "reason": "Year is different.",
            "severity": 1.0,
        }]
    })
    metric = ContradictionDetector(judge=judge, threshold=0.9)
    result = metric.score(make_sample())

    assert result.score < 0.9
    assert result.passed is False


def test_evidence_shows_claim_and_context():
    """Evidence must include both the contradicting claim and what context actually says."""
    judge = make_mock_judge({
        "contradictions": [{
            "claim": "The policy was approved.",
            "context_says": "The policy was rejected.",
            "reason": "Context says rejected, answer says approved.",
            "severity": 1.0,
        }]
    })
    metric = ContradictionDetector(judge=judge, threshold=0.9)
    result = metric.score(RAGSample(
        query="Was the policy approved?",
        retrieved_docs=["The policy was rejected by the committee in March."],
        answer="The policy was approved.",
    ))

    assert len(result.evidence) == 1
    assert "approved" in result.evidence[0]
    assert "rejected" in result.evidence[0]
    assert "CONTRADICTION" in result.evidence[0]


def test_partial_contradictions():
    """Multiple contradictions out of several claims — score between 0 and 1."""
    judge = make_mock_judge({
        "contradictions": [
            {
                "claim": "Revenue fell by 10%.",
                "context_says": "Revenue grew by 10%.",
                "reason": "Direction is opposite.",
                "severity": 0.9,
            }
        ]
    })
    # Answer has ~3 sentences, 1 contradiction → score = 1 - 1/3 ≈ 0.67
    metric = ContradictionDetector(judge=judge, threshold=0.9)
    result = metric.score(RAGSample(
        query="How did revenue perform?",
        retrieved_docs=["Revenue grew by 10%. Customer count increased. Margins held steady."],
        answer="Revenue fell by 10%. Customer count increased. Margins held steady.",
    ))

    assert 0.0 < result.score < 1.0
    assert result.passed is False
