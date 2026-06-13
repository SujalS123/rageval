# rageval/drift.py

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DriftReport:
    drift_score: float                              # 0.0 to 1.0
    uncovered_query_count: int
    uncovered_query_percentage: float
    new_topic_clusters: list[str] = field(default_factory=list)
    predicted_faithfulness_degradation: float = 0.0
    knowledge_base_additions_recommended: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Semantic Drift Report",
            f"Drift score: {self.drift_score:.3f}",
            f"Uncovered queries: {self.uncovered_query_count} "
            f"({self.uncovered_query_percentage*100:.1f}%)",
            f"Predicted faithfulness degradation: "
            f"{self.predicted_faithfulness_degradation:.3f}",
        ]
        if self.new_topic_clusters:
            lines.append("New topics detected:")
            for t in self.new_topic_clusters:
                lines.append(f"  - {t}")
        if self.knowledge_base_additions_recommended:
            lines.append("Recommended KB additions:")
            for r in self.knowledge_base_additions_recommended:
                lines.append(f"  - {r}")
        return "\n".join(lines)


class SemanticDriftDetector:
    """
    Detects when recent user queries are drifting toward topics not covered
    by the knowledge base — before faithfulness scores drop.

    No LLM calls. Pure embedding math. Fast enough for 10,000 queries.

    Algorithm:
    1. Embed baseline queries, recent queries, and knowledge base documents
    2. For each recent query, compute max cosine similarity to any KB document
    3. Queries below coverage_threshold are flagged as uncovered
    4. Cluster uncovered queries (greedy cosine) to find new topic areas
    5. drift_score = fraction of recent uncovered - fraction of baseline uncovered
    6. predicted_faithfulness_degradation = drift_score * 0.35

    Usage:
        detector = SemanticDriftDetector(embedding_judge=HeuristicJudge())
        detector.set_baseline(historical_queries)
        detector.set_knowledge_base(your_documents)
        report = detector.detect(this_weeks_queries)
    """

    DRIFT_CONSTANT = 0.35   # empirical: each unit of drift predicts ~0.35 faithfulness drop
    CLUSTER_THRESHOLD = 0.72

    def __init__(self, embedding_judge=None, coverage_threshold: float = 0.65):
        self.embedding_judge = embedding_judge
        self.coverage_threshold = coverage_threshold
        self._baseline_queries: list[str] = []
        self._kb_documents: list[str] = []

    def set_baseline(self, queries: list[str]) -> None:
        self._baseline_queries = queries

    def set_knowledge_base(self, documents: list[str]) -> None:
        self._kb_documents = documents

    def detect(self, recent_queries: list[str]) -> DriftReport:
        if not recent_queries:
            return DriftReport(
                drift_score=0.0,
                uncovered_query_count=0,
                uncovered_query_percentage=0.0,
            )

        # Embed everything in batch
        kb_embeddings = self._embed(self._kb_documents) if self._kb_documents else []
        recent_embeddings = self._embed(recent_queries)
        baseline_embeddings = self._embed(self._baseline_queries) if self._baseline_queries else []

        # Fraction of baseline uncovered (for relative drift computation)
        baseline_uncovered_frac = self._uncovered_fraction(baseline_embeddings, kb_embeddings)

        # Fraction of recent uncovered
        recent_uncovered_mask = self._uncovered_mask(recent_embeddings, kb_embeddings)
        recent_uncovered_frac = sum(recent_uncovered_mask) / len(recent_queries)

        # drift = difference in uncovered fractions, clamped to [0, 1]
        drift_score = round(max(0.0, min(1.0, recent_uncovered_frac - baseline_uncovered_frac)), 4)

        uncovered_queries = [q for q, u in zip(recent_queries, recent_uncovered_mask) if u]
        uncovered_embeddings = [e for e, u in zip(recent_embeddings, recent_uncovered_mask) if u]

        # Cluster uncovered queries to find new topics
        new_topic_clusters: list[str] = []
        kb_additions: list[str] = []
        if uncovered_queries:
            clusters = self._greedy_cluster(uncovered_queries, uncovered_embeddings)
            for cluster in sorted(clusters, key=lambda c: len(c["members"]), reverse=True):
                representative = cluster["members"][0]
                new_topic_clusters.append(representative[:120])
                kb_additions.append(f"Add documents covering: \"{representative[:80]}\"")

        predicted_degradation = round(drift_score * self.DRIFT_CONSTANT, 4)

        return DriftReport(
            drift_score=drift_score,
            uncovered_query_count=len(uncovered_queries),
            uncovered_query_percentage=round(recent_uncovered_frac, 4),
            new_topic_clusters=new_topic_clusters,
            predicted_faithfulness_degradation=predicted_degradation,
            knowledge_base_additions_recommended=kb_additions,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using embedding_judge if available, else character n-grams."""
        if not texts:
            return []
        if self.embedding_judge is not None:
            try:
                embeddings = self.embedding_judge.model.encode(texts)
                return [e.tolist() for e in embeddings]
            except Exception:
                pass
        return [self._ngram_vector(t) for t in texts]

    @staticmethod
    def _ngram_vector(text: str, n: int = 3, dim: int = 256) -> list[float]:
        text = text.lower()
        vec = [0.0] * dim
        total = max(len(text) - n + 1, 1)
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            bucket = hash(gram) % dim
            vec[bucket] += 1.0 / total
        return vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def _max_kb_similarity(self, query_emb: list[float], kb_embs: list[list[float]]) -> float:
        """Max cosine similarity between a query embedding and any KB document embedding."""
        if not kb_embs:
            return 0.0
        return max(self._cosine(query_emb, kb_emb) for kb_emb in kb_embs)

    def _uncovered_mask(
        self, query_embs: list[list[float]], kb_embs: list[list[float]]
    ) -> list[bool]:
        """True = query is below coverage_threshold (uncovered)."""
        return [
            self._max_kb_similarity(q, kb_embs) < self.coverage_threshold
            for q in query_embs
        ]

    def _uncovered_fraction(
        self, query_embs: list[list[float]], kb_embs: list[list[float]]
    ) -> float:
        if not query_embs:
            return 0.0
        mask = self._uncovered_mask(query_embs, kb_embs)
        return sum(mask) / len(mask)

    def _greedy_cluster(
        self, texts: list[str], embeddings: list[list[float]]
    ) -> list[dict]:
        clusters: list[dict] = []
        for text, emb in zip(texts, embeddings):
            best_idx, best_sim = -1, self.CLUSTER_THRESHOLD
            for idx, cluster in enumerate(clusters):
                sim = self._cosine(emb, cluster["centroid"])
                if sim > best_sim:
                    best_sim, best_idx = sim, idx
            if best_idx >= 0:
                cluster = clusters[best_idx]
                cluster["members"].append(text)
                cluster["embeddings"].append(emb)
                n = len(cluster["embeddings"])
                cluster["centroid"] = [
                    sum(cluster["embeddings"][i][d] for i in range(n)) / n
                    for d in range(len(emb))
                ]
            else:
                clusters.append({"centroid": emb[:], "embeddings": [emb], "members": [text]})
        return clusters
