# tests/test_tracker.py

import pytest
import tempfile
import os
from unittest.mock import MagicMock
from rageval.tracker import RunTracker
from rageval.core.result import EvalResult, MetricResult
from rageval.core.sample import RAGSample


def make_metric_result(name: str, score: float, threshold: float = 0.8) -> MetricResult:
    return MetricResult(
        metric_name=name,
        score=score,
        passed=score >= threshold,
        reasoning="test",
        evidence=[],
        threshold=threshold,
        hallucinations=[],
    )


def make_eval_result(score: float, passed: bool, metrics: dict[str, float]) -> EvalResult:
    sample = RAGSample(
        query="test query",
        retrieved_docs=["test doc"],
        answer="test answer",
    )
    metric_results = {name: make_metric_result(name, s) for name, s in metrics.items()}
    return EvalResult(
        sample=sample,
        metric_results=metric_results,
        overall_score=score,
        passed=passed,
    )


@pytest.fixture
def tracker(tmp_path):
    """Fresh tracker backed by a temp directory for each test."""
    db_path = str(tmp_path / "runs.db")
    return RunTracker(db_path=db_path)


def test_save_and_retrieve_run(tracker):
    """Saved run must be retrievable with correct aggregated scores."""
    results = [
        make_eval_result(0.9, True, {"faithfulness": 0.9, "context_precision": 0.85}),
        make_eval_result(0.7, False, {"faithfulness": 0.7, "context_precision": 0.75}),
    ]
    tracker.save_run("v1.0", results)
    run = tracker.get_run("v1.0")

    assert run["run_name"] == "v1.0"
    assert run["total_samples"] == 2
    assert run["overall_score"] == pytest.approx(0.8, abs=0.01)
    assert run["pass_rate"] == pytest.approx(0.5, abs=0.01)
    assert "faithfulness" in run["per_metric"]
    assert run["per_metric"]["faithfulness"] == pytest.approx(0.8, abs=0.01)


def test_list_runs_returns_all(tracker):
    """list_runs must return all saved runs."""
    results = [make_eval_result(0.8, True, {"faithfulness": 0.8})]
    tracker.save_run("run-a", results)
    tracker.save_run("run-b", results)

    runs = tracker.list_runs()
    names = [r["run_name"] for r in runs]
    assert "run-a" in names
    assert "run-b" in names
    assert len(runs) == 2


def test_compare_runs_delta(tracker):
    """compare_runs must compute correct deltas and directions."""
    results_a = [make_eval_result(0.7, False, {"faithfulness": 0.70})]
    results_b = [make_eval_result(0.9, True, {"faithfulness": 0.90})]

    tracker.save_run("v1", results_a)
    tracker.save_run("v2", results_b)

    comparison = tracker.compare_runs("v1", "v2")
    faith = comparison["metrics"]["faithfulness"]

    assert faith["run_a"] == pytest.approx(0.70, abs=0.01)
    assert faith["run_b"] == pytest.approx(0.90, abs=0.01)
    assert faith["delta"] == pytest.approx(0.20, abs=0.01)
    assert faith["direction"] == "improved"
    assert comparison["overall_delta"] == pytest.approx(0.20, abs=0.01)


def test_compare_runs_degraded(tracker):
    """Scores that drop must be marked as degraded."""
    results_a = [make_eval_result(0.9, True, {"faithfulness": 0.90})]
    results_b = [make_eval_result(0.6, False, {"faithfulness": 0.60})]

    tracker.save_run("good", results_a)
    tracker.save_run("bad", results_b)

    comparison = tracker.compare_runs("good", "bad")
    assert comparison["metrics"]["faithfulness"]["direction"] == "degraded"


def test_get_missing_run_raises(tracker):
    """get_run on a nonexistent name must raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        tracker.get_run("does-not-exist")


def test_overwrite_run(tracker):
    """Saving the same run name twice must overwrite, not duplicate."""
    results_v1 = [make_eval_result(0.7, False, {"faithfulness": 0.7})]
    results_v2 = [make_eval_result(0.95, True, {"faithfulness": 0.95})]

    tracker.save_run("my-run", results_v1)
    tracker.save_run("my-run", results_v2)

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["overall_score"] == pytest.approx(0.95, abs=0.01)
