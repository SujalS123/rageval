# tests/integration/test_end_to_end_real.py
"""
Full end-to-end integration test.
Simulates a real developer workflow from evaluation to reporting.
"""

import pytest
import json
import tempfile
import os
from rageval import evaluate, batch_evaluate, summary
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.answer_relevancy import AnswerRelevancy
from rageval.core.sample import RAGSample
from rageval.reporters.json_csv import to_json, to_csv


REAL_SAMPLES = [
    {
        "query": "What is the speed of light?",
        "retrieved_docs": [
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "Einstein's theory of relativity states that nothing can travel faster than light.",
        ],
        "answer": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "is_faithful": True,
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "retrieved_docs": [
            "Romeo and Juliet is a tragedy written by William Shakespeare in the late 16th century.",
            "The play was first performed around 1595 and published in 1597.",
        ],
        "answer": "Romeo and Juliet was written by Charles Dickens in the 19th century.",
        "is_faithful": False,
    },
    {
        "query": "What is DNA?",
        "retrieved_docs": [
            "DNA (deoxyribonucleic acid) is a molecule that carries genetic information.",
            "DNA is found in the nucleus of cells and consists of a double helix structure.",
        ],
        "answer": "DNA is a molecule that carries genetic information and has a double helix structure.",
        "is_faithful": True,
    },
]


class TestEndToEndReal:

    @pytest.fixture(scope="class")
    def samples(self):
        return [
            RAGSample(
                query=s["query"],
                retrieved_docs=s["retrieved_docs"],
                answer=s["answer"],
            )
            for s in REAL_SAMPLES
        ]

    @pytest.fixture(scope="class")
    def metrics(self, judge, embedding_judge):
        return [
            Faithfulness(judge=judge, threshold=0.8),
            ContextPrecision(judge=judge, threshold=0.7),
            AnswerRelevancy(
                judge=judge, threshold=0.7, embedding_judge=embedding_judge
            ),
        ]

    @pytest.fixture(scope="class")
    def results(self, samples, metrics):
        return batch_evaluate(
            samples=samples, metrics=metrics, show_progress=True
        )

    def test_batch_produces_correct_count(self, results):
        assert len(results) == len(REAL_SAMPLES)

    def test_faithful_samples_score_higher_than_unfaithful(self, results):
        """Faithful samples must have higher faithfulness than unfaithful."""
        faithful_score = results[0].metric_results["faithfulness"].score
        unfaithful_score = results[1].metric_results["faithfulness"].score

        print(f"\nFaithful sample faithfulness:   {faithful_score:.3f}")
        print(f"Unfaithful sample faithfulness: {unfaithful_score:.3f}")

        assert faithful_score > unfaithful_score, (
            f"Faithful sample ({faithful_score:.3f}) should score higher than "
            f"unfaithful sample ({unfaithful_score:.3f})"
        )

    def test_unfaithful_evidence_contains_wrong_author(self, results):
        """The Dickens hallucination must appear in faithfulness evidence."""
        unfaithful_result = results[1]
        evidence_text = " ".join(
            unfaithful_result.metric_results["faithfulness"].evidence
        ).lower()

        print(f"\nUnfaithful evidence: {unfaithful_result.metric_results['faithfulness'].evidence}")

        assert any(
            term in evidence_text for term in ["dickens", "charles", "shakespeare"]
        ), (
            f"Expected wrong author in evidence.\nGot: {evidence_text}"
        )

    def test_summary_statistics_correct(self, results):
        stats = summary(results)

        print(f"\nSummary: {json.dumps(stats, indent=2)}")

        assert stats["total_samples"] == len(REAL_SAMPLES)
        assert 0.0 <= stats["avg_overall_score"] <= 1.0
        assert 0.0 <= stats["overall_pass_rate"] <= 1.0
        assert "faithfulness" in stats["per_metric"]
        assert "context_precision" in stats["per_metric"]
        assert "answer_relevancy" in stats["per_metric"]

    def test_json_export_produces_valid_file(self, results):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            to_json(results, path)
            content = json.loads(open(path, encoding="utf-8").read())

            assert isinstance(content, list)
            assert len(content) == len(results)

            for item in content:
                assert "query" in item
                assert "overall_score" in item
                assert "metrics" in item
                assert "faithfulness" in item["metrics"]

        finally:
            os.unlink(path)

    def test_csv_export_produces_valid_file(self, results):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        try:
            to_csv(results, path)
            import csv
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            assert len(rows) == len(results)
            assert "query" in rows[0]
            assert "faithfulness_score" in rows[0]
            assert "overall_score" in rows[0]

        finally:
            os.unlink(path)

    def test_all_results_have_valid_structure(self, results):
        for i, result in enumerate(results):
            assert 0.0 <= result.overall_score <= 1.0, (
                f"Sample {i} overall score out of range"
            )
            assert isinstance(result.passed, bool), (
                f"Sample {i} passed must be bool"
            )
            assert result.latency_ms > 0, (
                f"Sample {i} latency must be positive"
            )
            for name, mr in result.metric_results.items():
                assert 0.0 <= mr.score <= 1.0, (
                    f"Sample {i} {name} score out of range"
                )
                assert isinstance(mr.reasoning, str), (
                    f"Sample {i} {name} reasoning must be string"
                )
                assert isinstance(mr.evidence, list), (
                    f"Sample {i} {name} evidence must be list"
                )

    def test_print_full_results(self, results):
        """Print full results for manual inspection."""
        print("\n" + "=" * 60)
        print("FULL END-TO-END RESULTS")
        print("=" * 60)
        for i, result in enumerate(results):
            print(f"\nSample {i+1}: {REAL_SAMPLES[i]['query']}")
            print(f"Expected faithful: {REAL_SAMPLES[i]['is_faithful']}")
            print(result.summary())
