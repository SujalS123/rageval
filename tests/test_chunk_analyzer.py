# tests/test_chunk_analyzer.py

import pytest
from rageval.chunk_analyzer import ChunkQualityAnalyzer, ChunkIssueType


@pytest.fixture
def analyzer():
    return ChunkQualityAnalyzer()  # no embedding judge — uses Jaccard fallback


def test_broken_context_start_detected(analyzer):
    """Chunk starting with lowercase must be flagged as BROKEN_CONTEXT."""
    chunks = ["this chunk starts with lowercase and has enough tokens " * 3]
    report = analyzer.analyze(chunks)
    types = [i.issue_type for i in report.issues]
    assert ChunkIssueType.BROKEN_CONTEXT in types


def test_broken_context_end_detected(analyzer):
    """Chunk ending without punctuation must be flagged."""
    chunks = ["This chunk ends without any sentence-ending punctuation at all"]
    report = analyzer.analyze(chunks)
    types = [i.issue_type for i in report.issues]
    assert ChunkIssueType.BROKEN_CONTEXT in types


def test_clean_chunk_no_broken_context(analyzer):
    """A properly formed chunk must not trigger BROKEN_CONTEXT."""
    chunks = ["This is a complete sentence with proper punctuation."]
    report = analyzer.analyze(chunks)
    broken = [i for i in report.issues if i.issue_type == ChunkIssueType.BROKEN_CONTEXT]
    assert broken == []


def test_duplicate_detected_with_jaccard(analyzer):
    """Two nearly identical chunks must trigger DUPLICATE_CONTENT."""
    base = "The quick brown fox jumps over the lazy dog and runs away quickly."
    chunks = [base, base + " Extra word."]
    report = analyzer.analyze(chunks, similarity_threshold=0.85)
    types = [i.issue_type for i in report.issues]
    assert ChunkIssueType.DUPLICATE_CONTENT in types


def test_duplicate_indices_are_correct(analyzer):
    """Duplicate issue must reference the second chunk, not the first."""
    base = " ".join(["word"] * 20)
    chunks = [base, base]
    report = analyzer.analyze(chunks, similarity_threshold=0.9)
    dup_issues = [i for i in report.issues if i.issue_type == ChunkIssueType.DUPLICATE_CONTENT]
    assert any(i.chunk_index == 1 for i in dup_issues)


def test_sparse_chunk_detected(analyzer):
    """A chunk with fewer tokens than min_tokens must be flagged as TOO_SPARSE."""
    chunks = ["Short chunk."]
    report = analyzer.analyze(chunks, min_tokens=50)
    types = [i.issue_type for i in report.issues]
    assert ChunkIssueType.TOO_SPARSE in types


def test_dense_chunk_detected(analyzer):
    """A chunk with more tokens than max_tokens must be flagged as TOO_DENSE."""
    long_chunk = " ".join(["word"] * 700)
    chunks = [long_chunk]
    report = analyzer.analyze(chunks, max_tokens=600)
    types = [i.issue_type for i in report.issues]
    assert ChunkIssueType.TOO_DENSE in types


def test_quality_score_decreases_with_issues(analyzer):
    """More issues must lower the quality score."""
    # Good chunks: varied, long enough, well-formed
    good_chunks = [
        f"This is well-formed sentence number {i} with enough content to pass all checks and ends properly."
        for i in range(10)
    ]
    # Bad chunks: all identical (duplicates) + broken context
    bad_chunks = ["incomplete fragment without punctuation"] * 10
    good_report = analyzer.analyze(good_chunks, min_tokens=5)
    bad_report = analyzer.analyze(bad_chunks, min_tokens=5, similarity_threshold=0.8)
    assert bad_report.quality_score < good_report.quality_score


def test_empty_chunks_returns_perfect_score(analyzer):
    """Empty input must return quality_score 1.0 without errors."""
    report = analyzer.analyze([])
    assert report.quality_score == 1.0
    assert report.total_chunks == 0
    assert report.issues == []
