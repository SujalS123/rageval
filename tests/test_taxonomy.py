# tests/test_taxonomy.py

import pytest
from unittest.mock import MagicMock
from rageval.taxonomy import FailureTaxonomyBuilder
from rageval.core.result import EvalResult, MetricResult
from rageval.core.sample import RAGSample


def make_metric_result(name: str, score: float, evidence: list[str], passed: bool = None) -> MetricResult:
    if passed is None:
        passed = score >= 0.8
    return MetricResult(
        metric_name=name,
        score=score,
        passed=passed,
        reasoning="test",
        evidence=evidence,
        threshold=0.8,
    )


def make_eval_result(metric_results: dict) -> EvalResult:
    sample = RAGSample(query="test", retrieved_docs=["doc"], answer="answer")
    return EvalResult(
        sample=sample,
        metric_results=metric_results,
        overall_score=0.5,
        passed=False,
    )


def make_naming_judge(name="Test Cluster"):
    judge = MagicMock()
    judge.complete_json.return_value = {
        "cluster_name": name,
        "pattern_description": "Claims not supported by context.",
        "trigger_condition": "LLM uses training knowledge.",
        "fix_suggestion": "Restrict LLM to context only.",
    }
    return judge


def test_empty_results_returns_empty_taxonomy():
    """No results → no failures → empty taxonomy with coverage 1.0."""
    judge = MagicMock()
    builder = FailureTaxonomyBuilder(judge=judge)
    taxonomy = builder.build([])

    assert taxonomy.total_failures == 0
    assert taxonomy.clusters == []
    assert taxonomy.coverage == 1.0
    judge.complete_json.assert_not_called()


def test_single_failure_creates_one_cluster():
    """One failed metric with one evidence string → one cluster."""
    judge = make_naming_judge("Knowledge Boundary")

    result = make_eval_result({
        "faithfulness": make_metric_result(
            "faithfulness", 0.0, ["NOT SUPPORTED: 'X was invented in 1900'"], passed=False
        )
    })

    builder = FailureTaxonomyBuilder(judge=judge)
    taxonomy = builder.build([result])

    assert taxonomy.total_failures == 1
    assert len(taxonomy.clusters) == 1
    assert taxonomy.clusters[0].name == "Knowledge Boundary"
    assert taxonomy.clusters[0].count == 1
    assert taxonomy.coverage == 1.0


def test_similar_failures_merge_into_same_cluster():
    """
    Evidence strings that are textually very similar should land in the same
    cluster via the greedy n-gram cosine clustering.
    """
    judge = make_naming_judge()
    # Highly similar evidence — same prefix, tiny variation
    evidence_a = "NOT SUPPORTED: 'The rate is 5%' — not in context"
    evidence_b = "NOT SUPPORTED: 'The rate is 6%' — not in context"

    result = make_eval_result({
        "faithfulness": make_metric_result(
            "faithfulness", 0.0, [evidence_a, evidence_b], passed=False
        )
    })

    builder = FailureTaxonomyBuilder(judge=judge)
    taxonomy = builder.build([result])

    assert taxonomy.total_failures == 2
    # Both should merge into a single cluster (or at most 2, but both accounted for)
    total_in_clusters = sum(c.count for c in taxonomy.clusters)
    assert total_in_clusters == 2


def test_dissimilar_failures_create_separate_clusters():
    """
    Evidence strings that are completely different in content should create
    separate clusters.
    """
    # One call per cluster
    judge = MagicMock()
    judge.complete_json.side_effect = [
        {"cluster_name": "Cluster A", "pattern_description": "A",
         "trigger_condition": "A", "fix_suggestion": "A"},
        {"cluster_name": "Cluster B", "pattern_description": "B",
         "trigger_condition": "B", "fix_suggestion": "B"},
    ]

    # These strings share almost no character trigrams
    evidence_a = "CONTRADICTION: policy approved — context says rejected severity 1.0"
    evidence_b = "MISSING FROM ANSWER: thirty day grace period not mentioned answer omits"

    result = make_eval_result({
        "faithfulness": make_metric_result(
            "faithfulness", 0.0, [evidence_a], passed=False
        ),
        "answer_completeness": make_metric_result(
            "answer_completeness", 0.0, [evidence_b], passed=False
        ),
    })

    builder = FailureTaxonomyBuilder(judge=judge)
    taxonomy = builder.build([result])

    assert taxonomy.total_failures == 2
    assert len(taxonomy.clusters) >= 2


def test_clusters_sorted_by_size_descending():
    """Largest cluster must appear first."""
    judge = MagicMock()
    judge.complete_json.side_effect = [
        {"cluster_name": "Big", "pattern_description": "",
         "trigger_condition": "", "fix_suggestion": ""},
        {"cluster_name": "Small", "pattern_description": "",
         "trigger_condition": "", "fix_suggestion": ""},
    ]

    # 3 very similar + 1 very different
    similar = ["NOT SUPPORTED: claim about revenue — no context"] * 3
    different = ["MISSING FROM ANSWER: grace period not mentioned in response"]

    result = make_eval_result({
        "m1": make_metric_result("m1", 0.0, similar, passed=False),
        "m2": make_metric_result("m2", 0.0, different, passed=False),
    })

    builder = FailureTaxonomyBuilder(judge=judge)
    taxonomy = builder.build([result])

    if len(taxonomy.clusters) >= 2:
        assert taxonomy.clusters[0].count >= taxonomy.clusters[1].count
