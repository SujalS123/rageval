# tests/test_answer_completeness.py

import pytest
from unittest.mock import MagicMock
from rageval.metrics.answer_completeness import AnswerCompleteness
from rageval.core.sample import RAGSample


def make_mock_judge(extraction_result: dict, coverage_result: dict):
    judge = MagicMock()
    judge.complete_json.side_effect = [extraction_result, coverage_result]
    return judge


def make_sample(answer="The sky is blue and contains water vapor."):
    return RAGSample(
        query="What is the sky like?",
        retrieved_docs=["The sky is blue due to Rayleigh scattering. It contains water vapor which forms clouds."],
        answer=answer,
    )


def test_perfect_completeness():
    """All relevant facts mentioned in answer — should score 1.0 and pass."""
    judge = make_mock_judge(
        extraction_result={"facts": ["The sky is blue.", "The sky contains water vapor."]},
        coverage_result={
            "coverage": [
                {"fact": "The sky is blue.", "mentioned": True, "reason": "Answer states this."},
                {"fact": "The sky contains water vapor.", "mentioned": True, "reason": "Answer states this."},
            ]
        },
    )
    metric = AnswerCompleteness(judge=judge, threshold=0.8)
    result = metric.score(make_sample())

    assert result.score == 1.0
    assert result.passed is True
    assert result.evidence == []
    assert "all" in result.reasoning.lower()


def test_partial_completeness():
    """One fact mentioned, one missing — should score 0.5."""
    judge = make_mock_judge(
        extraction_result={"facts": ["The sky is blue.", "Clouds form from water vapor."]},
        coverage_result={
            "coverage": [
                {"fact": "The sky is blue.", "mentioned": True, "reason": "Covered."},
                {"fact": "Clouds form from water vapor.", "mentioned": False, "reason": "Not mentioned."},
            ]
        },
    )
    metric = AnswerCompleteness(judge=judge, threshold=0.8)
    result = metric.score(make_sample(answer="The sky is blue."))

    assert result.score == 0.5
    assert result.passed is False
    assert len(result.evidence) == 1


def test_zero_completeness():
    """No facts mentioned — should score 0.0 and fail."""
    judge = make_mock_judge(
        extraction_result={"facts": ["The sky is blue.", "It contains water vapor."]},
        coverage_result={
            "coverage": [
                {"fact": "The sky is blue.", "mentioned": False, "reason": "Not mentioned."},
                {"fact": "It contains water vapor.", "mentioned": False, "reason": "Not mentioned."},
            ]
        },
    )
    metric = AnswerCompleteness(judge=judge, threshold=0.8)
    result = metric.score(make_sample(answer="I don't know."))

    assert result.score == 0.0
    assert result.passed is False
    assert len(result.evidence) == 2


def test_evidence_contains_missing_facts():
    """Evidence must list the specific facts the answer omitted."""
    judge = make_mock_judge(
        extraction_result={"facts": ["Revenue grew 42% in Q3."]},
        coverage_result={
            "coverage": [
                {"fact": "Revenue grew 42% in Q3.", "mentioned": False, "reason": "Answer does not state the percentage."},
            ]
        },
    )
    metric = AnswerCompleteness(judge=judge, threshold=0.8)
    result = metric.score(RAGSample(
        query="What was revenue growth?",
        retrieved_docs=["Revenue grew 42% in Q3 of this year."],
        answer="Revenue grew this quarter.",
    ))

    assert result.passed is False
    assert any("42%" in e for e in result.evidence)
    assert any("MISSING FROM ANSWER" in e for e in result.evidence)


def test_empty_context_facts_scores_1():
    """If no relevant facts are found in context, score should be 1.0 (nothing to cover)."""
    judge = make_mock_judge(
        extraction_result={"facts": []},
        coverage_result={},  # never called
    )
    metric = AnswerCompleteness(judge=judge, threshold=0.8)
    result = metric.score(make_sample())

    assert result.score == 1.0
    assert result.passed is True


def test_validate_catches_missing_query():
    """RAGSample with blank query raises before any LLM call."""
    judge = MagicMock()
    metric = AnswerCompleteness(judge=judge, threshold=0.8)

    with pytest.raises(ValueError):
        RAGSample(
            query="  ",
            retrieved_docs=["some doc"],
            answer="some answer",
        )
