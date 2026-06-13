# tests/test_dataset.py

import json
import pytest
from unittest.mock import MagicMock
from rageval.dataset import EvalDatasetGenerator


def make_judge(*side_effects):
    judge = MagicMock()
    judge.complete_json.side_effect = list(side_effects)
    return judge


def test_generates_correct_number_of_questions():
    """Should produce exactly n_questions samples across documents."""
    # 1 doc, ask for 2 questions → 1 question-gen call + 2 ground-truth calls
    judge = make_judge(
        {"questions": [
            {"query": "What is X?", "question_type": "factual"},
            {"query": "Why does Y happen?", "question_type": "inferential"},
        ]},
        {"answer": "X is a thing."},
        {"answer": "Y happens because of Z."},
    )
    gen = EvalDatasetGenerator(judge=judge)
    results = gen.generate(documents=["Doc about X and Y."], n_questions=2)

    assert len(results) == 2


def test_output_format_matches_ragsample_fields():
    """Every sample must have the four fields rageval run expects."""
    judge = make_judge(
        {"questions": [{"query": "What is A?", "question_type": "factual"}]},
        {"answer": "A is the first letter."},
    )
    gen = EvalDatasetGenerator(judge=judge)
    results = gen.generate(documents=["A is the first letter of the alphabet."], n_questions=1)

    assert len(results) == 1
    sample = results[0]
    assert "query" in sample
    assert "retrieved_docs" in sample
    assert "answer" in sample
    assert "ground_truth" in sample
    assert isinstance(sample["retrieved_docs"], list)
    assert len(sample["retrieved_docs"]) == 1


def test_save_writes_valid_json(tmp_path):
    """save() must write a JSON file directly loadable by rageval run."""
    judge = make_judge(
        {"questions": [{"query": "What is B?", "question_type": "factual"}]},
        {"answer": "B is the second letter."},
    )
    gen = EvalDatasetGenerator(judge=judge)
    results = gen.generate(documents=["B is the second letter."], n_questions=1)

    out_path = str(tmp_path / "eval_data.json")
    gen.save(results, out_path)

    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert isinstance(loaded, list)
    assert len(loaded) == 1
    assert loaded[0]["query"] == "What is B?"


def test_source_doc_index_is_set():
    """source_doc_index must correspond to the document position."""
    judge = make_judge(
        {"questions": [{"query": "Q from doc 0?", "question_type": "factual"}]},
        {"answer": "Answer from doc 0."},
        {"questions": [{"query": "Q from doc 1?", "question_type": "factual"}]},
        {"answer": "Answer from doc 1."},
    )
    gen = EvalDatasetGenerator(judge=judge)
    results = gen.generate(
        documents=["First document.", "Second document."],
        n_questions=2,
    )

    assert results[0]["source_doc_index"] == 0
    assert results[1]["source_doc_index"] == 1


def test_empty_documents_raises():
    """Passing an empty documents list must raise ValueError immediately."""
    judge = MagicMock()
    gen = EvalDatasetGenerator(judge=judge)
    with pytest.raises(ValueError, match="empty"):
        gen.generate(documents=[], n_questions=5)


def test_llm_failure_skips_document_gracefully():
    """If the judge raises on a document, it is skipped without crashing."""
    judge = MagicMock()
    judge.complete_json.side_effect = RuntimeError("API error")
    gen = EvalDatasetGenerator(judge=judge)
    results = gen.generate(documents=["Some document."], n_questions=2)
    # Should return empty list rather than raising
    assert results == []
