# tests/test_explainer.py

import json
import pytest
from rageval.explainer import ExplainabilityReporter
from rageval.core.sample import RAGSample
from rageval.core.result import EvalResult, MetricResult
from rageval.core.hallucination import Hallucination, HallucinationType


def make_eval_result(faithfulness_score: float = 0.67) -> EvalResult:
    sample = RAGSample(
        query="How tall is the Eiffel Tower?",
        retrieved_docs=["The Eiffel Tower is 330 metres tall and was completed in 1889."],
        answer="The Eiffel Tower is 330 metres tall. It was designed by Leonardo da Vinci.",
    )
    hallucination = Hallucination(
        claim="It was designed by Leonardo da Vinci.",
        type=HallucinationType.UNSUPPORTED_CLAIM,
        severity=0.9,
        reason="Context says Gustave Eiffel, not da Vinci.",
    )
    mr = MetricResult(
        metric_name="faithfulness",
        score=faithfulness_score,
        passed=faithfulness_score >= 0.8,
        reasoning="1 of 2 claims unsupported.",
        evidence=["UNSUPPORTED_CLAIM: 'It was designed by Leonardo da Vinci.'"],
        threshold=0.8,
        hallucinations=[hallucination],
    )
    return EvalResult(
        sample=sample,
        metric_results={"faithfulness": mr},
        overall_score=faithfulness_score,
        passed=faithfulness_score >= 0.8,
    )


def make_sample() -> RAGSample:
    return RAGSample(
        query="How tall is the Eiffel Tower?",
        retrieved_docs=["The Eiffel Tower is 330 metres tall and was completed in 1889."],
        answer="The Eiffel Tower is 330 metres tall. It was designed by Leonardo da Vinci.",
    )


def test_report_contains_all_sections():
    """ExplanationReport must have all required fields populated."""
    reporter = ExplainabilityReporter()
    sample = make_sample()
    eval_result = make_eval_result()

    report = reporter.explain(sample, eval_result)

    assert report.query_analysis is not None
    assert "query_type" in report.query_analysis
    assert "entities" in report.query_analysis
    assert isinstance(report.per_doc_analysis, list)
    assert isinstance(report.answer_analysis, list)
    assert isinstance(report.metric_summary, dict)
    assert isinstance(report.action_items, list)
    assert len(report.action_items) > 0


def test_supported_claims_correctly_labeled():
    """The first sentence (supported) should be labeled 'supported'."""
    reporter = ExplainabilityReporter()
    sample = make_sample()
    eval_result = make_eval_result()

    report = reporter.explain(sample, eval_result)

    # First sentence is "The Eiffel Tower is 330 metres tall." — supported
    first = report.answer_analysis[0]
    assert first.sentence == "The Eiffel Tower is 330 metres tall."
    assert first.label == "supported"


def test_unsupported_claims_correctly_labeled():
    """The second sentence (unsupported hallucination) should be labeled 'unsupported'."""
    reporter = ExplainabilityReporter()
    sample = make_sample()
    eval_result = make_eval_result()

    report = reporter.explain(sample, eval_result)

    unsupported = [s for s in report.answer_analysis if s.label == "unsupported"]
    assert len(unsupported) >= 1
    assert any("da Vinci" in s.sentence or "Leonardo" in s.sentence for s in unsupported)


def test_html_output_is_valid_string(tmp_path):
    """to_html must write a non-empty file containing HTML tags."""
    reporter = ExplainabilityReporter()
    sample = make_sample()
    eval_result = make_eval_result()
    report = reporter.explain(sample, eval_result)

    out = str(tmp_path / "report.html")
    reporter.to_html(report, out)

    content = open(out, encoding="utf-8").read()
    assert len(content) > 100
    assert "<html" in content
    assert "</html>" in content
    assert "rageval" in content


def test_to_dict_is_json_serializable():
    """to_dict output must be serializable to JSON without errors."""
    reporter = ExplainabilityReporter()
    sample = make_sample()
    eval_result = make_eval_result()
    report = reporter.explain(sample, eval_result)

    d = reporter.to_dict(report)
    serialized = json.dumps(d)  # must not raise
    loaded = json.loads(serialized)

    assert "query_analysis" in loaded
    assert "metric_summary" in loaded
    assert "action_items" in loaded
    assert "answer_analysis" in loaded
    assert "per_doc_analysis" in loaded


def test_metric_summary_has_interpretation():
    """Each metric in metric_summary must have score and interpretation."""
    reporter = ExplainabilityReporter()
    report = reporter.explain(make_sample(), make_eval_result(0.4))

    assert "faithfulness" in report.metric_summary
    entry = report.metric_summary["faithfulness"]
    assert "score" in entry
    assert "interpretation" in entry
    assert isinstance(entry["interpretation"], str)
    assert len(entry["interpretation"]) > 5


def test_action_items_ordered_by_impact():
    """Critical items (faithfulness) should appear before medium ones."""
    reporter = ExplainabilityReporter()
    report = reporter.explain(make_sample(), make_eval_result(0.3))

    assert any("[CRITICAL]" in item for item in report.action_items)
