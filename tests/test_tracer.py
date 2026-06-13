# tests/test_tracer.py

import pytest
from unittest.mock import MagicMock
from rageval.trace import RAGTracer, RAGTrace, TraceStep
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.faithfulness import Faithfulness


def make_precision_judge(is_relevant: bool = True):
    judge = MagicMock()
    judge.complete_json.return_value = {
        "is_relevant": is_relevant,
        "reason": "test reason",
    }
    return judge


def make_faith_judge(supported: bool = True):
    judge = MagicMock()
    judge.complete_json.side_effect = [
        {"claims": ["The sky is blue."]},
        {"verifications": [
            {"claim": "The sky is blue.", "supported": supported,
             "reason": "context confirms" if supported else "not in context",
             "type": None, "severity": None}
        ]},
    ]
    return judge


@pytest.fixture
def tracer(tmp_path):
    return RAGTracer(db_path=str(tmp_path / "runs.db"))


def test_trace_captures_all_steps(tracer):
    """All steps logged inside the with-block must appear in the trace."""
    with tracer.trace("test-001") as t:
        t.step("retrieval")
        t.log_retrieval(["doc about finance"], scores=[0.9])
        t.step("generation")
        t.log_output("The answer is X.")

    trace = t._trace
    assert trace.trace_id == "test-001"
    assert len(trace.steps) == 2
    assert trace.steps[0].name == "retrieval"
    assert trace.steps[1].name == "generation"
    assert trace.final_answer == "The answer is X."


def test_trace_captures_retrieval_docs(tracer):
    """log_retrieval must store docs and scores on the current step."""
    with tracer.trace("test-002") as t:
        t.step("retrieval")
        t.log_retrieval(["doc1", "doc2"], scores=[0.91, 0.72])

    step = t._trace.steps[0]
    assert step.retrieval_docs == ["doc1", "doc2"]
    assert step.retrieval_scores == [0.91, 0.72]


def test_evaluate_trace_runs_precision_on_retrieval_steps(tmp_path):
    """evaluate_trace must call ContextPrecision on steps that have retrieval_docs."""
    precision_judge = make_precision_judge(is_relevant=True)
    metric = ContextPrecision(judge=precision_judge, threshold=0.8)
    tracer = RAGTracer(metrics=[metric], db_path=str(tmp_path / "runs.db"))

    with tracer.trace("test-003") as t:
        t.step("retrieval")
        t.log_retrieval(["Relevant document about finance."], scores=[0.9])
        t.step("generation")
        t.log_output("Finance answer.")

    result = tracer.evaluate_trace(t._trace)

    assert "retrieval" in result["steps"]
    assert "context_precision" in result["steps"]["retrieval"]["scores"]


def test_root_cause_identifies_first_failing_step(tmp_path):
    """Root cause must be the name of the first step where quality drops."""
    # Make precision judge return not relevant → step fails
    bad_judge = make_precision_judge(is_relevant=False)
    metric = ContextPrecision(judge=bad_judge, threshold=0.8)
    tracer = RAGTracer(metrics=[metric], db_path=str(tmp_path / "runs.db"))

    with tracer.trace("test-004") as t:
        t.step("retrieval")
        t.log_retrieval(["Unrelated document about cooking."], scores=[0.5])
        t.log_output("An answer.")

    result = tracer.evaluate_trace(t._trace)

    assert result["root_cause"] == "retrieval"


def test_no_root_cause_when_all_pass(tmp_path):
    """Root cause must be None when all steps pass."""
    good_judge = make_precision_judge(is_relevant=True)
    metric = ContextPrecision(judge=good_judge, threshold=0.8)
    tracer = RAGTracer(metrics=[metric], db_path=str(tmp_path / "runs.db"))

    with tracer.trace("test-005") as t:
        t.step("retrieval")
        t.log_retrieval(["Fully relevant document."], scores=[0.95])
        t.log_output("Good answer.")

    result = tracer.evaluate_trace(t._trace)

    assert result["root_cause"] is None


def test_trace_is_persisted_and_retrievable(tmp_path):
    """Traces must be saved to SQLite and loadable via get_trace."""
    tracer = RAGTracer(db_path=str(tmp_path / "runs.db"))

    with tracer.trace("persist-001") as t:
        t.step("retrieval")
        t.log_retrieval(["some doc"])
        t.log_output("some answer")

    tracer.evaluate_trace(t._trace)

    loaded = tracer.get_trace("persist-001")
    assert loaded["trace_id"] == "persist-001"
    assert len(loaded["steps"]) == 1
