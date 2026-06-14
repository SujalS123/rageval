# rageval/chunk_analyzer.py

from dataclasses import dataclass, field
from enum import Enum


class ChunkIssueType(str, Enum):
    BROKEN_CONTEXT = "broken_context"
    DUPLICATE_CONTENT = "duplicate_content"
    TOO_SPARSE = "too_sparse"
    TOO_DENSE = "too_dense"


@dataclass
class ChunkIssue:
    chunk_index: int
    issue_type: ChunkIssueType
    description: str
    severity: float  # 0.0 to 1.0
    suggestion: str


@dataclass
class ChunkQualityReport:
    total_chunks: int
    issues: list[ChunkIssue] = field(default_factory=list)
    quality_score: float = 1.0  # 0.0 to 1.0
    recommendations: list[str] = field(default_factory=list)

    def issues_by_type(self) -> dict[str, list[ChunkIssue]]:
        result: dict[str, list[ChunkIssue]] = {}
        for issue in self.issues:
            result.setdefault(issue.issue_type.value, []).append(issue)
        return result

    def summary(self) -> str:
        lines = [
            "Chunk Quality Report",
            f"Total chunks: {self.total_chunks}",
            f"Quality score: {self.quality_score:.2f}",
            f"Total issues: {len(self.issues)}",
        ]
        by_type = self.issues_by_type()
        for issue_type, items in sorted(by_type.items()):
            pct = len(items) / self.total_chunks * 100
            lines.append(f"  {issue_type}: {len(items)} ({pct:.1f}%)")
        if self.recommendations:
            lines.append("\nRecommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)


class ChunkQualityAnalyzer:
    """
    Evaluates the quality of document chunks BEFORE indexing.

    Works without any LLM calls — uses heuristics and optional embeddings.
    Free and fast to run on large document collections.

    Detection:
    - BROKEN_CONTEXT   : chunk starts with lowercase or ends without sentence-ending punctuation
    - DUPLICATE_CONTENT: cosine similarity above threshold between any two chunks
    - TOO_SPARSE       : fewer than min_tokens non-whitespace tokens
    - TOO_DENSE        : more than max_tokens tokens

    Usage:
        analyzer = ChunkQualityAnalyzer()
        report = analyzer.analyze(chunks, similarity_threshold=0.92)
        print(report.summary())
    """

    def __init__(self, embedding_judge=None):
        self.embedding_judge = embedding_judge

    def analyze(
        self,
        chunks: list[str],
        similarity_threshold: float = 0.92,
        min_tokens: int = 50,
        max_tokens: int = 600,
    ) -> ChunkQualityReport:
        if not chunks:
            return ChunkQualityReport(total_chunks=0, quality_score=1.0)

        issues: list[ChunkIssue] = []

        for i, chunk in enumerate(chunks):
            issues.extend(self._check_broken_context(i, chunk))
            issues.extend(self._check_token_count(i, chunk, min_tokens, max_tokens))

        # Duplicate detection — O(n²) but fine for typical chunk counts
        dup_issues = self._check_duplicates(chunks, similarity_threshold)
        issues.extend(dup_issues)

        quality_score = max(0.0, 1.0 - len(issues) / max(len(chunks), 1))
        quality_score = round(quality_score, 4)

        recommendations = self._build_recommendations(issues, chunks)

        return ChunkQualityReport(
            total_chunks=len(chunks),
            issues=issues,
            quality_score=quality_score,
            recommendations=recommendations,
        )

    # ── Detection methods ──────────────────────────────────────────────────

    def _check_broken_context(self, idx: int, chunk: str) -> list[ChunkIssue]:
        issues = []
        stripped = chunk.strip()
        if not stripped:
            return issues

        # Starts mid-sentence: first character is lowercase letter
        if stripped[0].islower():
            issues.append(ChunkIssue(
                chunk_index=idx,
                issue_type=ChunkIssueType.BROKEN_CONTEXT,
                description=f"Chunk starts with lowercase: '{stripped[:60]}...'",
                severity=0.6,
                suggestion="Increase chunk overlap so chunks begin at sentence boundaries.",
            ))

        # Ends mid-sentence: no sentence-ending punctuation
        if stripped and stripped[-1] not in ".!?\"'":
            issues.append(ChunkIssue(
                chunk_index=idx,
                issue_type=ChunkIssueType.BROKEN_CONTEXT,
                description=f"Chunk ends without punctuation: '...{stripped[-60:]}'",
                severity=0.5,
                suggestion="Adjust chunk size or overlap to end at sentence boundaries.",
            ))

        return issues

    def _check_token_count(self, idx: int, chunk: str, min_tokens: int, max_tokens: int) -> list[ChunkIssue]:
        # Simple whitespace tokenisation — no NLTK/spaCy dependency
        tokens = [t for t in chunk.split() if t.strip()]
        count = len(tokens)
        issues = []

        if count < min_tokens:
            severity = round(1.0 - count / min_tokens, 2)
            issues.append(ChunkIssue(
                chunk_index=idx,
                issue_type=ChunkIssueType.TOO_SPARSE,
                description=f"Only {count} tokens (minimum {min_tokens}): '{chunk.strip()[:80]}'",
                severity=severity,
                suggestion=f"Filter chunks with fewer than {min_tokens} tokens before indexing.",
            ))

        if count > max_tokens:
            severity = round(min(1.0, (count - max_tokens) / max_tokens), 2)
            issues.append(ChunkIssue(
                chunk_index=idx,
                issue_type=ChunkIssueType.TOO_DENSE,
                description=f"{count} tokens exceeds maximum {max_tokens}.",
                severity=severity,
                suggestion=f"Reduce chunk size to {max_tokens} tokens for better retrieval precision.",
            ))

        return issues

    def _check_duplicates(self, chunks: list[str], threshold: float) -> list[ChunkIssue]:
        issues = []

        if self.embedding_judge is not None:
            # Use embeddings for semantic similarity
            issues.extend(self._embedding_duplicates(chunks, threshold))
        else:
            # Fallback: simple token overlap (Jaccard similarity)
            issues.extend(self._jaccard_duplicates(chunks, threshold))

        return issues

    def _embedding_duplicates(self, chunks: list[str], threshold: float) -> list[ChunkIssue]:
        issues = []
        reported: set[int] = set()
        for i in range(len(chunks)):
            if i in reported:
                continue
            sims = self.embedding_judge.batch_similarity(chunks[i], chunks[i+1:])
            for offset, sim in enumerate(sims):
                j = i + 1 + offset
                if sim >= threshold and j not in reported:
                    reported.add(j)
                    issues.append(ChunkIssue(
                        chunk_index=j,
                        issue_type=ChunkIssueType.DUPLICATE_CONTENT,
                        description=f"Chunk {j} is {sim*100:.0f}% similar to chunk {i}.",
                        severity=round(sim, 2),
                        suggestion="Add a deduplication step to your indexing pipeline.",
                    ))
        return issues

    def _jaccard_duplicates(self, chunks: list[str], threshold: float) -> list[ChunkIssue]:
        """Token-overlap Jaccard similarity — no embedding model needed."""
        issues = []
        reported: set[int] = set()
        token_sets = [set(c.lower().split()) for c in chunks]

        for i in range(len(chunks)):
            if i in reported:
                continue
            for j in range(i + 1, len(chunks)):
                if j in reported:
                    continue
                a, b = token_sets[i], token_sets[j]
                union = a | b
                if not union:
                    continue
                jaccard = len(a & b) / len(union)
                if jaccard >= threshold:
                    reported.add(j)
                    issues.append(ChunkIssue(
                        chunk_index=j,
                        issue_type=ChunkIssueType.DUPLICATE_CONTENT,
                        description=f"Chunk {j} has {jaccard*100:.0f}% token overlap with chunk {i}.",
                        severity=round(jaccard, 2),
                        suggestion="Add a deduplication step to your indexing pipeline.",
                    ))
        return issues

    def _build_recommendations(self, issues: list[ChunkIssue], chunks: list[str]) -> list[str]:
        by_type: dict[str, int] = {}
        for issue in issues:
            by_type[issue.issue_type.value] = by_type.get(issue.issue_type.value, 0) + 1

        recs = []
        total = len(chunks)

        if by_type.get("broken_context", 0) > total * 0.05:
            recs.append("More than 5% of chunks have broken context — increase chunk overlap.")
        if by_type.get("duplicate_content", 0) > total * 0.03:
            recs.append("More than 3% of chunks are near-duplicates — add deduplication to indexing.")
        if by_type.get("too_sparse", 0) > total * 0.05:
            recs.append("More than 5% of chunks are too sparse — filter short chunks before indexing.")
        if by_type.get("too_dense", 0) > total * 0.05:
            recs.append("More than 5% of chunks are too dense — reduce chunk size.")

        return recs
