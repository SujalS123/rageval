# tests/test_prompt_vc.py

import pytest
from rageval.prompt_vc import PromptVersionControl
from rageval.core.result import EvalResult, MetricResult
from rageval.core.sample import RAGSample


def make_metric_result(name: str, score: float, threshold: float = 0.8) -> MetricResult:
    return MetricResult(
        metric_name=name, score=score, passed=score >= threshold,
        reasoning="test", evidence=[], threshold=threshold,
    )


def make_results(faithfulness_scores: list[float]) -> list[EvalResult]:
    results = []
    for s in faithfulness_scores:
        sample = RAGSample(query="q", retrieved_docs=["doc"], answer="ans")
        mr = make_metric_result("faithfulness", s)
        results.append(EvalResult(
            sample=sample,
            metric_results={"faithfulness": mr},
            overall_score=s,
            passed=s >= 0.8,
        ))
    return results


@pytest.fixture
def pvc(tmp_path):
    return PromptVersionControl(db_path=str(tmp_path / "runs.db"))


def test_register_and_retrieve(pvc):
    pvc.register("v1", "Answer only from context.")
    assert pvc.get("v1") == "Answer only from context."


def test_get_missing_raises_key_error(pvc):
    with pytest.raises(KeyError, match="not found"):
        pvc.get("nonexistent")


def test_list_versions_returns_all(pvc):
    pvc.register("v1", "Prompt A")
    pvc.register("v2", "Prompt B")
    versions = pvc.list_versions()
    names = [v["version_name"] for v in versions]
    assert "v1" in names
    assert "v2" in names


def test_compare_improvement_recommends_deploy_b(pvc):
    """When scores in B are significantly higher than A → recommend deploy_b."""
    pvc.register("v1", "Original prompt.")
    pvc.register("v2", "Improved prompt.")

    # A: low scores, B: high scores — large enough gap to be significant
    results_a = make_results([0.3, 0.35, 0.28, 0.32, 0.31, 0.34, 0.29, 0.30, 0.33, 0.27])
    results_b = make_results([0.9, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87, 0.90, 0.92, 0.88])

    report = pvc.compare("v1", "v2", results_a, results_b)
    assert report.recommendation == "deploy_b"
    assert report.metric_deltas["faithfulness"]["direction"] == "improved"
    assert report.metric_deltas["faithfulness"]["significant"] is True


def test_compare_regression_recommends_keep_a(pvc):
    """When scores in B are significantly lower than A → recommend keep_a."""
    pvc.register("v1", "Good prompt.")
    pvc.register("v2", "Bad prompt.")

    results_a = make_results([0.9, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87, 0.90, 0.92, 0.88])
    results_b = make_results([0.3, 0.28, 0.32, 0.29, 0.31, 0.27, 0.33, 0.30, 0.28, 0.32])

    report = pvc.compare("v1", "v2", results_a, results_b)
    assert report.recommendation == "keep_a"
    assert report.metric_deltas["faithfulness"]["direction"] == "degraded"


def test_compare_no_significant_change_is_inconclusive(pvc):
    """When scores are nearly identical → inconclusive."""
    pvc.register("v1", "Prompt A.")
    pvc.register("v2", "Prompt B.")

    # Almost identical scores — z-test will not be significant
    scores = [0.80, 0.81, 0.79, 0.80, 0.82, 0.80, 0.81, 0.79, 0.80, 0.81]
    report = pvc.compare("v1", "v2", make_results(scores), make_results(scores))
    assert report.recommendation == "inconclusive"


def test_compare_includes_prompt_diff(pvc):
    """PromptComparisonReport must include a unified diff of the two prompts."""
    pvc.register("v1", "Answer from context only.")
    pvc.register("v2", "Answer from context only. Never fabricate.")

    results = make_results([0.8, 0.8])
    report = pvc.compare("v1", "v2", results, results)
    assert "fabricate" in report.prompt_diff or "Never" in report.prompt_diff
