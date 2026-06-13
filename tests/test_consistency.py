# tests/test_consistency.py

import pytest
from unittest.mock import MagicMock, patch
from rageval.consistency import ConsistencyAnalyzer, ConsistencyReport


# ── Helpers ────────────────────────────────────────────────────────────────

def make_judge(*responses):
    """Mock judge whose complete_json returns responses in order."""
    judge = MagicMock()
    judge.complete_json.side_effect = list(responses)
    return judge


def make_embedding_judge(similarities: list[float]):
    """Mock embedding judge with fixed batch_similarity return."""
    ej = MagicMock()
    ej.batch_similarity.return_value = similarities
    return ej


def make_pipeline(answers: list[str], docs_per_call: list[list[str]] = None):
    """
    Returns a pipeline_fn whose successive calls return (docs, answer) pairs.
    First call = original query, subsequent calls = paraphrases.
    """
    calls = [0]
    def pipeline_fn(query: str):
        idx = calls[0]
        calls[0] += 1
        docs = (docs_per_call[idx] if docs_per_call else ["doc about topic"])
        return docs, answers[idx]
    return pipeline_fn


# ── Tests ──────────────────────────────────────────────────────────────────

def test_identical_answers_score_1():
    """
    When pipeline returns the exact same answer for query and paraphrase,
    the LLM should find no contradictions → score = 1.0.

    LLM calls: 2 extraction + 1 comparison = 3 total.
    """
    judge = make_judge(
        # extraction for answer 1
        {"claims": ["The sky is blue."]},
        # extraction for answer 2
        {"claims": ["The sky is blue."]},
        # cross-comparison — no contradictions, no inconsistencies
        {"contradictions": [], "inconsistencies": []},
    )

    analyzer = ConsistencyAnalyzer(judge=judge)
    pipeline = make_pipeline(
        answers=["The sky is blue.", "The sky is blue."],
    )

    report = analyzer.analyze(
        query="What colour is the sky?",
        paraphrases=["What is the colour of the sky?"],
        pipeline_fn=pipeline,
    )

    assert report.consistency_score == 1.0
    assert report.contradictions() == []


def test_contradicting_answers_score_low():
    """
    When answers directly contradict each other, score must be below 1.0.

    LLM calls: 2 extraction + 1 comparison = 3 total.
    """
    judge = make_judge(
        # extraction answer 1
        {"claims": ["The treaty was signed in 1847."]},
        # extraction answer 2
        {"claims": ["The treaty was signed in 1850."]},
        # cross-comparison — direct contradiction
        {
            "contradictions": [{
                "claim_a": "The treaty was signed in 1847.",
                "claim_b": "The treaty was signed in 1850.",
                "reason": "Different years given.",
            }],
            "inconsistencies": [],
        },
    )

    analyzer = ConsistencyAnalyzer(judge=judge)
    pipeline = make_pipeline(
        answers=["The treaty was signed in 1847.", "The treaty was signed in 1850."],
    )

    report = analyzer.analyze(
        query="When was the treaty signed?",
        paraphrases=["What year was the treaty signed?"],
        pipeline_fn=pipeline,
    )

    assert report.consistency_score < 1.0
    assert len(report.contradictions()) == 1


def test_inconsistencies_appear_in_report():
    """
    Inconsistencies (present in one answer, absent in another) must be
    captured in the report even when they are not full contradictions.
    """
    judge = make_judge(
        {"claims": ["Revenue grew 40%.", "The CEO resigned."]},
        {"claims": ["Revenue grew 40%."]},
        {
            "contradictions": [],
            "inconsistencies": [{
                "claim_a": "The CEO resigned.",
                "claim_b": "",
                "reason": "Answer B does not mention the CEO.",
            }],
        },
    )

    analyzer = ConsistencyAnalyzer(judge=judge)
    pipeline = make_pipeline(
        answers=[
            "Revenue grew 40%. The CEO resigned.",
            "Revenue grew 40%.",
        ],
    )

    report = analyzer.analyze(
        query="What happened at the company?",
        paraphrases=["What were the company's recent events?"],
        pipeline_fn=pipeline,
    )

    assert len(report.inconsistencies) >= 1
    assert any(it.type == "inconsistency" for it in report.inconsistencies)
    assert any("CEO" in it.claim_a for it in report.inconsistencies)


def test_root_cause_detected_vocabulary_mismatch():
    """
    When paraphrases retrieve semantically different documents (low similarity),
    root cause must mention vocabulary mismatch.
    """
    # Doc similarity < 0.7 → vocabulary mismatch
    ej = make_embedding_judge(similarities=[0.45])

    judge = make_judge(
        {"claims": ["Claim A."]},
        {"claims": ["Claim A."]},
        {"contradictions": [], "inconsistencies": []},
    )

    analyzer = ConsistencyAnalyzer(judge=judge, embedding_judge=ej)
    pipeline = make_pipeline(
        answers=["Answer A.", "Answer A."],
        docs_per_call=[["finance doc"], ["cooking doc"]],
    )

    report = analyzer.analyze(
        query="What caused the crisis?",
        paraphrases=["Why did the crisis happen?"],
        pipeline_fn=pipeline,
    )

    assert "mismatch" in report.root_cause_hypothesis.lower() or \
           "different" in report.root_cause_hypothesis.lower()


def test_root_cause_generation_when_docs_consistent():
    """
    When docs are similar (high similarity) but answers still differ,
    root cause must point to the generation step.
    """
    ej = make_embedding_judge(similarities=[0.95])

    judge = make_judge(
        {"claims": ["The rate is 5%."]},
        {"claims": ["The rate is 7%."]},
        {
            "contradictions": [{
                "claim_a": "The rate is 5%.",
                "claim_b": "The rate is 7%.",
                "reason": "Different rates.",
            }],
            "inconsistencies": [],
        },
    )

    analyzer = ConsistencyAnalyzer(judge=judge, embedding_judge=ej)
    pipeline = make_pipeline(
        answers=["The rate is 5%.", "The rate is 7%."],
        docs_per_call=[["interest rate doc"], ["interest rate doc"]],
    )

    report = analyzer.analyze(
        query="What is the interest rate?",
        paraphrases=["What rate is charged?"],
        pipeline_fn=pipeline,
    )

    assert "generation" in report.root_cause_hypothesis.lower() or \
           "consistent" in report.root_cause_hypothesis.lower()


def test_pipeline_failure_handled_gracefully():
    """
    If pipeline_fn raises on one paraphrase, the analyzer must still
    return a report using the successful calls.
    """
    judge = make_judge(
        {"claims": ["Paris is the capital."]},
    )

    calls = [0]
    def flaky_pipeline(query):
        calls[0] += 1
        if calls[0] == 1:
            return ["doc"], "Paris is the capital."
        raise RuntimeError("retriever down")

    analyzer = ConsistencyAnalyzer(judge=judge)
    # Only 1 valid answer — should return score 1.0 with a note
    report = analyzer.analyze(
        query="What is the capital of France?",
        paraphrases=["Name the capital city of France."],
        pipeline_fn=flaky_pipeline,
    )

    # With only 1 valid answer, returns immediately with score 1.0
    assert report.consistency_score == 1.0


def test_answer_similarity_scores_populated_when_embedding_judge_provided():
    """
    When embedding_judge is provided, answer_similarity_scores must be
    a non-empty list of floats.
    """
    ej = make_embedding_judge(similarities=[0.88])
    ej.batch_similarity.return_value = [0.88]

    judge = make_judge(
        {"claims": ["Claim X."]},
        {"claims": ["Claim X."]},
        {"contradictions": [], "inconsistencies": []},
    )

    analyzer = ConsistencyAnalyzer(judge=judge, embedding_judge=ej)
    pipeline = make_pipeline(answers=["Answer X.", "Answer X."])

    report = analyzer.analyze(
        query="Q?",
        paraphrases=["Q rephrased?"],
        pipeline_fn=pipeline,
    )

    assert len(report.answer_similarity_scores) == 1
    assert report.answer_similarity_scores[0] == pytest.approx(0.88, abs=0.01)
