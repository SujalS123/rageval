# tests/test_query_classifier.py

import pytest
from unittest.mock import MagicMock
from rageval.query_classifier import QueryClassifier, QueryType
from rageval.core.result import EvalResult, MetricResult
from rageval.core.sample import RAGSample


def make_judge(query_type: str):
    judge = MagicMock()
    judge.complete_json.return_value = {"query_type": query_type}
    return judge


def make_eval_result(query: str, score: float, passed: bool) -> EvalResult:
    sample = RAGSample(
        query=query,
        retrieved_docs=["some doc"],
        answer="some answer",
    )
    mr = MetricResult(
        metric_name="faithfulness",
        score=score,
        passed=passed,
        reasoning="test",
        evidence=[],
        threshold=0.8,
    )
    return EvalResult(
        sample=sample,
        metric_results={"faithfulness": mr},
        overall_score=score,
        passed=passed,
    )


def test_classify_factual():
    classifier = QueryClassifier(judge=make_judge("factual"))
    assert classifier.classify("What is the capital of France?") == QueryType.FACTUAL


def test_classify_comparison():
    classifier = QueryClassifier(judge=make_judge("comparison"))
    assert classifier.classify("How does Python compare to Java?") == QueryType.COMPARISON


def test_classify_multi_hop():
    classifier = QueryClassifier(judge=make_judge("multi_hop"))
    assert classifier.classify("Who founded the company that built the Eiffel Tower?") == QueryType.MULTI_HOP


def test_classify_falls_back_to_ambiguous_on_unknown_type():
    """An unrecognised type string must fall back to AMBIGUOUS without raising."""
    judge = MagicMock()
    judge.complete_json.return_value = {"query_type": "totally_made_up_type"}
    classifier = QueryClassifier(judge=judge)
    result = classifier.classify("some query")
    assert result == QueryType.AMBIGUOUS


def test_classify_falls_back_on_llm_failure():
    """An LLM exception must fall back to AMBIGUOUS."""
    judge = MagicMock()
    judge.complete_json.side_effect = RuntimeError("API error")
    classifier = QueryClassifier(judge=judge)
    assert classifier.classify("any query") == QueryType.AMBIGUOUS


def test_classify_batch_groups_by_type():
    """Batch classification must group results by type with correct counts."""
    judge = MagicMock()
    # Two factual, one comparison
    judge.complete_json.side_effect = [
        {"query_type": "factual"},
        {"query_type": "factual"},
        {"query_type": "comparison"},
    ]
    classifier = QueryClassifier(judge=judge)

    results = [
        make_eval_result("Q1", 0.9, True),
        make_eval_result("Q2", 0.8, True),
        make_eval_result("Q3", 0.5, False),
    ]
    breakdown = classifier.classify_batch(results)

    assert "factual" in breakdown
    assert breakdown["factual"]["count"] == 2
    assert "comparison" in breakdown
    assert breakdown["comparison"]["count"] == 1
    assert breakdown["comparison"]["pass_rate"] == 0.0


def test_classify_batch_computes_avg_scores():
    """avg_scores must be the mean per metric across all samples of that type."""
    judge = MagicMock()
    judge.complete_json.side_effect = [
        {"query_type": "factual"},
        {"query_type": "factual"},
    ]
    classifier = QueryClassifier(judge=judge)

    results = [
        make_eval_result("Q1", 0.6, False),
        make_eval_result("Q2", 1.0, True),
    ]
    breakdown = classifier.classify_batch(results)

    assert breakdown["factual"]["avg_scores"]["faithfulness"] == pytest.approx(0.8, abs=0.01)
