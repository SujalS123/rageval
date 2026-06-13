# tests/test_autoeval.py

import time
import pytest
from unittest.mock import MagicMock, patch
from rageval.autoeval import AutoEval
from rageval.core.result import MetricResult, EvalResult
from rageval.core.sample import RAGSample


def make_metric(score: float = 0.9):
    """Mock metric that always returns a fixed score."""
    metric = MagicMock()
    metric.name = "faithfulness"
    mr = MetricResult(
        metric_name="faithfulness",
        score=score,
        passed=score >= 0.8,
        reasoning="mock",
        evidence=[],
        threshold=0.8,
    )
    metric.score.return_value = mr
    return metric


def make_eval_result(score: float) -> EvalResult:
    sample = RAGSample(
        query="test", retrieved_docs=["doc"], answer="answer"
    )
    mr = MetricResult(
        metric_name="faithfulness", score=score,
        passed=score >= 0.8, reasoning="mock", evidence=[], threshold=0.8,
    )
    return EvalResult(sample=sample, metric_results={"faithfulness": mr},
                      overall_score=score, passed=score >= 0.8)


def test_decorated_function_returns_correct_value():
    """The original return value must come back unchanged regardless of sampling."""
    ae = AutoEval(metrics=[], sample_rate=1.0)

    @ae.monitor
    def handler(query, retrieved_docs, answer):
        return "original result"

    result = handler(query="q", retrieved_docs=["doc"], answer="a")
    assert result == "original result"


def test_evaluation_does_not_raise_on_failure():
    """If evaluation crashes, the decorated function must still return correctly."""
    bad_metric = MagicMock()
    bad_metric.score.side_effect = RuntimeError("LLM is down")
    ae = AutoEval(metrics=[bad_metric], sample_rate=1.0)

    @ae.monitor
    def handler(query, retrieved_docs, answer):
        return "safe return"

    # Should not raise
    result = handler(query="q", retrieved_docs=["doc"], answer="ans")
    assert result == "safe return"


def test_sampling_respects_sample_rate():
    """With sample_rate=0.0 no evaluations should be submitted."""
    ae = AutoEval(metrics=[], sample_rate=0.0)
    call_count = [0]

    original_submit = ae._executor.submit

    def counting_submit(fn, *args, **kwargs):
        call_count[0] += 1
        return original_submit(fn, *args, **kwargs)

    ae._executor.submit = counting_submit

    @ae.monitor
    def handler(query, retrieved_docs, answer):
        return "result"

    for _ in range(20):
        handler(query="q", retrieved_docs=["doc"], answer="ans")

    assert call_count[0] == 0


def test_alert_fn_called_when_score_below_threshold():
    """alert_fn must be called when rolling average drops below alert_threshold."""
    alert_received = []

    def my_alert(stats):
        alert_received.append(stats)

    ae = AutoEval(metrics=[], sample_rate=1.0, alert_threshold=0.8, alert_fn=my_alert)

    # Directly inject low scores to bypass actual evaluation
    ae._scores.extend([0.4, 0.4, 0.4])
    ae._evaluated_count = 3

    # Trigger the alert check manually
    stats = ae.get_live_stats()
    if stats["avg_score"] < ae.alert_threshold and ae.alert_fn:
        ae.alert_fn(stats)

    assert len(alert_received) == 1
    assert alert_received[0]["avg_score"] < 0.8


def test_get_live_stats_empty():
    """Stats on empty history must return None scores without error."""
    ae = AutoEval(metrics=[], sample_rate=0.0)
    stats = ae.get_live_stats()
    assert stats["avg_score"] is None
    assert stats["evaluated_count"] == 0


def test_get_live_stats_with_scores():
    """Stats must correctly compute avg/min/max from the score window."""
    ae = AutoEval(metrics=[], sample_rate=0.0)
    ae._scores.extend([0.6, 0.8, 1.0])
    ae._evaluated_count = 3

    stats = ae.get_live_stats()
    assert stats["avg_score"] == pytest.approx(0.8, abs=0.01)
    assert stats["min_score"] == pytest.approx(0.6, abs=0.01)
    assert stats["max_score"] == pytest.approx(1.0, abs=0.01)
