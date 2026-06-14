# rageval/taxonomy.py

import math
from dataclasses import dataclass, field

from rageval.judges.base import BaseJudge

CLUSTER_NAMING_PROMPT = """\
You are given a list of failure evidence strings from a RAG evaluation system.
All of these failures belong to the same cluster — they share a common pattern.

Your job is to characterise this cluster precisely.

Failure examples:
{examples}

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{
    "cluster_name": "short name for this failure pattern (max 5 words)",
    "pattern_description": "one sentence describing what all these failures have in common",
    "trigger_condition": "one sentence describing what causes this failure to occur",
    "fix_suggestion": "one concrete action to reduce this failure type"
}}
"""


@dataclass
class FailureCluster:
    name: str
    pattern: str
    trigger: str
    fix: str
    example_evidence: list[str]
    count: int
    percentage_of_failures: float


@dataclass
class FailureTaxonomy:
    total_failures: int
    clusters: list[FailureCluster] = field(default_factory=list)
    coverage: float = 0.0   # fraction of failures explained by clusters

    def summary(self) -> str:
        lines = [
            f"Failure Taxonomy — {self.total_failures} total failures",
            f"Coverage: {self.coverage*100:.1f}% of failures explained",
            "",
        ]
        for i, c in enumerate(self.clusters, 1):
            lines.append(
                f"{i}. {c.name} — {c.count} failures ({c.percentage_of_failures*100:.1f}%)"
            )
            lines.append(f"   Pattern: {c.pattern}")
            lines.append(f"   Fix: {c.fix}")
        return "\n".join(lines)


class FailureTaxonomyBuilder:
    """
    Clusters all failure evidence from a set of EvalResults into named patterns
    and generates fix suggestions for each cluster.

    Algorithm:
    1. Collect all evidence strings from failed MetricResults
    2. Embed them using embedding_judge (or fall back to character n-gram heuristic)
    3. Greedy clustering: assign each embedding to the nearest cluster centroid
       if similarity > 0.75, otherwise start a new cluster
    4. For each cluster, call judge to generate name, pattern, trigger, fix
    5. Sort by size descending

    Usage:
        builder = FailureTaxonomyBuilder(judge=judge, embedding_judge=HeuristicJudge())
        taxonomy = builder.build(eval_results)
        print(taxonomy.summary())
    """

    SIMILARITY_THRESHOLD = 0.75
    MAX_EXAMPLES_PER_CLUSTER = 3
    MAX_CLUSTERS_TO_NAME = 10  # cap LLM calls — only name the largest clusters

    def __init__(self, judge: BaseJudge, embedding_judge=None):
        self.judge = judge
        self.embedding_judge = embedding_judge

    def build(self, results: list) -> FailureTaxonomy:
        # Step 1: collect evidence from failed metric results
        all_evidence: list[str] = []
        for result in results:
            for mr in result.metric_results.values():
                if not mr.passed:
                    all_evidence.extend(mr.evidence)

        if not all_evidence:
            return FailureTaxonomy(total_failures=0, clusters=[], coverage=1.0)

        # Step 2: embed
        embeddings = self._embed(all_evidence)

        # Step 3: greedy clustering
        raw_clusters = self._greedy_cluster(all_evidence, embeddings)

        # Sort by size descending before naming
        raw_clusters.sort(key=lambda c: len(c["members"]), reverse=True)

        # Step 4: name each cluster (cap LLM calls)
        named_clusters: list[FailureCluster] = []
        total = len(all_evidence)

        for rc in raw_clusters[:self.MAX_CLUSTERS_TO_NAME]:
            count = len(rc["members"])
            pct = round(count / total, 4) if total > 0 else 0.0
            examples = rc["members"][:self.MAX_EXAMPLES_PER_CLUSTER]
            name, pattern, trigger, fix = self._name_cluster(examples)
            named_clusters.append(FailureCluster(
                name=name,
                pattern=pattern,
                trigger=trigger,
                fix=fix,
                example_evidence=examples,
                count=count,
                percentage_of_failures=pct,
            ))

        # Step 5: coverage = fraction of failures in named clusters
        covered = sum(c.count for c in named_clusters)
        coverage = round(covered / total, 4) if total > 0 else 1.0

        return FailureTaxonomy(
            total_failures=total,
            clusters=named_clusters,
            coverage=coverage,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors. Falls back to char n-gram if no model."""
        if self.embedding_judge is not None:
            try:
                all_texts = texts
                embeddings = self.embedding_judge.model.encode(all_texts)
                return [emb.tolist() for emb in embeddings]
            except Exception:
                pass
        # Fallback: simple character n-gram frequency vector (no dependencies)
        return [self._ngram_vector(t) for t in texts]

    @staticmethod
    def _ngram_vector(text: str, n: int = 3) -> list[float]:
        """Character trigram frequency vector — dependency-free embedding fallback."""
        text = text.lower()
        ngrams: dict[str, int] = {}
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            ngrams[gram] = ngrams.get(gram, 0) + 1
        total = sum(ngrams.values()) or 1
        # Return a fixed-length vector by hashing gram to a bucket
        dim = 256
        vec = [0.0] * dim
        for gram, cnt in ngrams.items():
            bucket = hash(gram) % dim
            vec[bucket] += cnt / total
        return vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def _greedy_cluster(
        self, texts: list[str], embeddings: list[list[float]]
    ) -> list[dict]:
        """
        Greedy clustering: assign each item to the nearest existing cluster centroid
        if similarity > SIMILARITY_THRESHOLD, else create a new cluster.
        Centroid is the mean of all member embeddings.
        """
        clusters: list[dict] = []  # each: {"centroid": [...], "members": [...]}

        for text, emb in zip(texts, embeddings):
            best_idx = -1
            best_sim = self.SIMILARITY_THRESHOLD  # must exceed threshold to merge

            for idx, cluster in enumerate(clusters):
                sim = self._cosine(emb, cluster["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_idx >= 0:
                # Merge into existing cluster and update centroid
                cluster = clusters[best_idx]
                cluster["members"].append(text)
                cluster["embeddings"].append(emb)
                n = len(cluster["embeddings"])
                cluster["centroid"] = [
                    sum(cluster["embeddings"][i][d] for i in range(n)) / n
                    for d in range(len(emb))
                ]
            else:
                clusters.append({
                    "centroid": emb[:],
                    "embeddings": [emb],
                    "members": [text],
                })

        return clusters

    def _name_cluster(self, examples: list[str]) -> tuple[str, str, str, str]:
        """Call LLM to generate name, pattern, trigger, fix for a cluster."""
        examples_text = "\n".join(f"- {e}" for e in examples)
        try:
            result = self.judge.complete_json(
                CLUSTER_NAMING_PROMPT.format(examples=examples_text)
            )
            return (
                result.get("cluster_name", "Unnamed Cluster"),
                result.get("pattern_description", ""),
                result.get("trigger_condition", ""),
                result.get("fix_suggestion", ""),
            )
        except Exception:
            return ("Unnamed Cluster", "", "", "")
