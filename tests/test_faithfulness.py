# tests/test_faithfulness.py
import pytest
from unittest.mock import MagicMock
from rageval.metrics.faithfulness import Faithfulness
from rageval.core.sample import RAGSample
# from rageval.judges.base import JudgeResponse

def make_mock_judge(extraction_result: dict , verification_result: dict):
    """
    Create a mock judge that returns controlled responses.
    Never makes real API calls - tests run instantly and free.
    """
    judge = MagicMock()

    #complete_json is called twice: once for extraction , one for verification
    judge.complete_json.side_effect = [extraction_result , verification_result]
    return judge


def test_perfect_faithfulness():
    """All claims supported - should score 1.0 and pass."""
    judge = make_mock_judge(
        extraction_result ={
            "claims": ["Water boils at 100 degrees Celsius."]
        },
        verification_result={
            "verifications": [
                {
                    "claim": "Water boils at 100 degrees celsius.",
                    "supported": True,
                    "reason":"Context explicitly states this."
                }
            ]
        }

    )

    metric = Faithfulness(judge=judge , threshold=0.8)
    sample = RAGSample(
        query="what is the boiling point of water?",
        retrieved_docs=["WAter boils at 100 degrees Celsius at sea level."],
        answer="Water boils at 100 degrees Celsius.",
    )

    result = metric.score(sample)

    assert result.score == 1.0
    assert result.passed is True
    assert result.evidence == []
    assert "All" in result.reasoning

def test_zero_faithfulness():
    """No claims supported - should score 0.0 and fail."""
    judge = make_mock_judge(
        extraction_result={
            "claims":[
                "The capital of France is Berlin."
                "France was founded in 1200 BC.",
            ]
        },
        verification_result={
            "verifications": [
                {
                    "claims":"The capital of France is Berlin.",
                    "supported":False,
                    "reason":"context says capital is Paris , not Berlin."
                },
                {
                    "claim":"France was Founded in 1200 Bc.",
                    "supported":False,
                    "reason":"context does not mention founding date."
                },
            ]
        }
    )

    metric = Faithfulness(judge=judge , threshold=0.8)
    sample = RAGSample(
        query="Tell me about France.",
        retrieved_docs=["France is a country in Weastern Europe.Its capital is paris"],
        answer="The capital of France is Berlin. France was founded in 1200 BC.",
    )

    result = metric.score(sample)

    assert result.score == 0.0
    assert result.passed is False
    assert len(result.evidence) == 2

def test_partial_faithfulness():
    """Two claims , one supported , one not - should score 0.5"""
    judge = make_mock_judge(
       extraction_result={
          "claims":[
             "Paris is the capital of France.",
             "Paris has a population of 50 million.",
          ]
       },
       verification_result={
          "verifications":[
          {
             "claim":"Paris is the capital of France.",
             "supported":True,
             "reason":"Context confirms this."
          },
          {
             "claims":"Paris has a population of 50 million.",
             "supported": False,
             "reason":"Context says 2 million , not 50 million."
          },
       ]
       }
    )
    metric = Faithfulness(judge = judge , threshold = 0.8)
    sample = RAGSample(
      query="Tell me about Paris.",
      retrieved_docs=['Paris is the capital of France with a population of 2 million.'],
      answer="Paris is the capital of France. Paris has a population of 50 million",
   )
    result = metric.score(sample)

    assert result.score == 0.5
    assert result.passed is False
    assert len(result.evidence) == 1
    assert "50 million" in result.evidence[0]

def test_evidence_contains_failing_claim():
    """The specific hallucinated claim must appear in evidence."""
    judge = make_mock_judge(
        extraction_result={
            "claims":['Aliens built the pyramids.']
        },
        verification_result={
            "verifications":[
                {
                    "claim":"Aliens built the pyramids.",
                    "supported":False,
                    "reason":"context says ancient Egyptians built them."
                }
            ]
        }
    )
    metric = Faithfulness(judge=judge , threshold=0.8)
    sample = RAGSample(
        query = "Who built the pyramids?",
        retrieved_docs= ["The pyramids were built by ancient Egyptions over 4000 years ago."],
        answer="Aliens built the pyramids.",
    )

    result = metric.score(sample)

    assert result.passed is False
    assert any("Aliens" in e for e in result.evidence)

def test_empty_anser_claims():
    """Answer with no factual claims should score 1.0."""
    judge = make_mock_judge(

        extraction_result = {"claims":[]},
        verification_result ={} #never called
    )
    
    metric = Faithfulness(judge=judge , threshold = 0.8)
    sample = RAGSample(
        query = "What is the capital of France?",
        retrieved_docs=["France is in Europe."],
        answer="I don't know the answer to that question.",
    )

    result = metric.score(sample)

    assert result.score == 1.0
    assert result.passed is True


def test_validate_catches_missing_answer():
    """RAGSample with blank answer should raise ValueError before any LLM call."""
    judge = MagicMock()
    metric = Faithfulness(judge=judge , threshold=0.8)

    with pytest.raises(ValueError):
        RAGSample(
            query = "test query",
            retrieved_docs=["some context"],
            answer=" ",#blank
        )
