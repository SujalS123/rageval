# rageval/consistency.py

from dataclasses import dataclass, field
from typing import Callable, Optional

from rageval.judges.base import BaseJudge
from rageval.metrics.faithfulness import CLAIM_EXTRACTION_PROMPT

CROSS_ANSWER_CONTRADICTION_PROMPT = """\
You are given two answers to semantically equivalent questions. Your job is to find \
any claims that directly contradict each other across the two answers.

A contradiction exists when:
- Answer A states X and Answer B states not-X, or the opposite value.
- Answer A gives a specific number/date/name that Answer B gives differently.

Do NOT flag:
- Claims in one answer not mentioned in the other (that is inconsistency, not contradiction).
- Minor phrasing differences that convey the same meaning.

Answer A (query: {query_a}):
{answer_a}

Answer B (query: {query_b}):
{answer_b}

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{
    "contradictions": [
        {{
            "claim_a": "the claim from answer A",
            "claim_b": "the contradicting claim from answer B",
            "reason": "brief explanation"
        }}
    ],
    "inconsistencies": [
        {{
            "claim_a": "claim present in A but absent/different in B",
            "claim_b": "corresponding claim in B, or empty string if absent",
            "reason": "brief explanation of the difference"
        }}
    ]
}}
"""


@dataclass
class ConsistencyItem:
    query_a: str
    query_b: str
    claim_a: str
    claim_b: str
    type: str   # "contradiction" or "inconsistency"
    reason: str


@dataclass
class ConsistencyReport:
    consistency_score: float          # 0.0 to 1.0
    answer_similarity_scores: list[float] = field(default_factory=list)
    inconsistencies: list[ConsistencyItem] = field(default_factory=list)
    root_cause_hypothesis: str = ""
    fix_suggestion: str = ""

    def contradictions(self) -> list[ConsistencyItem]:
        return [i for i in self.inconsistencies if i.type == "contradiction"]

    def summary(self) -> str:
        lines = [
            f"Consistency Score: {self.consistency_score:.3f}",
            f"Total issues: {len(self.inconsistencies)} "
            f"({len(self.contradictions())} contradictions)",
        ]
        if self.root_cause_hypothesis:
            lines.append(f"Root cause: {self.root_cause_hypothesis}")
        if self.fix_suggestion:
            lines.append(f"Fix: {self.fix_suggestion}")
        return "\n".join(lines)


class ConsistencyAnalyzer:
    """
    Measures whether a RAG pipeline gives consistent answers across semantically
    equivalent queries (paraphrases of the same question).

    A pipeline that answers "The treaty was signed in 1847" for one phrasing and
    "The treaty was signed in 1850" for another is inconsistent — users will
    get different facts depending on how they phrase their question.

    Algorithm:
    1. Run pipeline_fn on the original query and all paraphrases
    2. Extract atomic claims from each answer (one LLM call per answer)
    3. Cross-compare every pair of answers for contradictions and inconsistencies
    4. consistency_score = 1.0 - (contradictions / total_claim_comparisons)
    5. Root cause: if embedding_judge shows paraphrases retrieved different docs,
       root cause is vocabulary mismatch in the embedding model

    Usage:
        analyzer = ConsistencyAnalyzer(judge=judge, embedding_judge=HeuristicJudge())

        def my_pipeline(query):
            docs = retriever.search(query)
            answer = llm.generate(query, docs)
            return docs, answer

        report = analyzer.analyze(
            query="What caused the 2008 crisis?",
            paraphrases=["Why did the 2008 financial crisis happen?",
                         "What were the causes of the 2008 recession?"],
            pipeline_fn=my_pipeline,
        )
    """

    def __init__(
        self,
        judge: BaseJudge,
        embedding_judge=None,
    ):
        self.judge = judge
        self.embedding_judge = embedding_judge

    def analyze(
        self,
        query: str,
        paraphrases: list[str],
        pipeline_fn: Callable[[str], tuple],
    ) -> ConsistencyReport:
        """
        Run pipeline_fn on query + all paraphrases and measure answer consistency.

        pipeline_fn signature: (query: str) -> (docs: list[str], answer: str)
        """
        all_queries = [query] + paraphrases

        # Step 1: run the pipeline on every query variant
        results: list[dict] = []
        for q in all_queries:
            try:
                docs, answer = pipeline_fn(q)
                results.append({"query": q, "docs": docs or [], "answer": answer or ""})
            except Exception as e:
                results.append({"query": q, "docs": [], "answer": "", "error": str(e)})

        # Filter to runs that actually produced answers
        valid = [r for r in results if r.get("answer", "").strip()]
        if len(valid) < 2:
            return ConsistencyReport(
                consistency_score=1.0,
                root_cause_hypothesis="Not enough valid answers to compare.",
                fix_suggestion="",
            )

        # Step 2: extract claims from each answer
        for r in valid:
            r["claims"] = self._extract_claims(r["answer"])

        # Step 3: cross-compare every pair
        all_inconsistencies: list[ConsistencyItem] = []
        total_comparisons = 0
        total_contradictions = 0

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i], valid[j]
                items = self._compare_answers(a, b)
                all_inconsistencies.extend(items)

                n_claims_a = max(len(a["claims"]), 1)
                n_claims_b = max(len(b["claims"]), 1)
                total_comparisons += max(n_claims_a, n_claims_b)
                total_contradictions += sum(1 for it in items if it.type == "contradiction")

        # Step 4: consistency score
        if total_comparisons == 0:
            consistency_score = 1.0
        else:
            consistency_score = max(
                0.0, 1.0 - total_contradictions / total_comparisons
            )
        consistency_score = round(consistency_score, 4)

        # Step 5: answer embedding similarities (if embedding_judge available)
        answer_similarities: list[float] = []
        if self.embedding_judge is not None and len(valid) >= 2:
            anchor = valid[0]["answer"]
            others = [r["answer"] for r in valid[1:]]
            try:
                answer_similarities = self.embedding_judge.batch_similarity(anchor, others)
            except Exception:
                answer_similarities = []

        # Step 6: root cause hypothesis via doc similarity
        root_cause, fix_suggestion = self._diagnose_root_cause(valid)

        return ConsistencyReport(
            consistency_score=consistency_score,
            answer_similarity_scores=answer_similarities,
            inconsistencies=all_inconsistencies,
            root_cause_hypothesis=root_cause,
            fix_suggestion=fix_suggestion,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _extract_claims(self, answer: str) -> list[str]:
        """Extract atomic claims from an answer using the Faithfulness prompt."""
        try:
            result = self.judge.complete_json(
                CLAIM_EXTRACTION_PROMPT.format(answer=answer)
            )
            return result.get("claims", [])
        except Exception:
            return []

    def _compare_answers(self, a: dict, b: dict) -> list[ConsistencyItem]:
        """Cross-compare two answer dicts for contradictions and inconsistencies."""
        try:
            result = self.judge.complete_json(
                CROSS_ANSWER_CONTRADICTION_PROMPT.format(
                    query_a=a["query"],
                    answer_a=a["answer"],
                    query_b=b["query"],
                    answer_b=b["answer"],
                )
            )
        except Exception:
            return []

        items: list[ConsistencyItem] = []

        for c in result.get("contradictions", []):
            items.append(ConsistencyItem(
                query_a=a["query"],
                query_b=b["query"],
                claim_a=c.get("claim_a", ""),
                claim_b=c.get("claim_b", ""),
                type="contradiction",
                reason=c.get("reason", ""),
            ))

        for inc in result.get("inconsistencies", []):
            items.append(ConsistencyItem(
                query_a=a["query"],
                query_b=b["query"],
                claim_a=inc.get("claim_a", ""),
                claim_b=inc.get("claim_b", ""),
                type="inconsistency",
                reason=inc.get("reason", ""),
            ))

        return items

    def _diagnose_root_cause(self, valid: list[dict]) -> tuple[str, str]:
        """
        Use embedding_judge to compare the docs retrieved per paraphrase.
        If paraphrases retrieve semantically different documents, the root cause
        is vocabulary mismatch in the embedding model.
        """
        if self.embedding_judge is None or len(valid) < 2:
            return (
                "Could not diagnose root cause — no embedding judge provided.",
                "Pass embedding_judge=HeuristicJudge() to enable root cause analysis.",
            )

        # Build one string per query from its retrieved docs
        doc_strings = []
        for r in valid:
            docs = r.get("docs", [])
            doc_strings.append(" ".join(docs[:3]) if docs else "")

        # Filter out runs with no docs
        non_empty = [d for d in doc_strings if d.strip()]
        if len(non_empty) < 2:
            return (
                "Could not diagnose root cause — no retrieved documents to compare.",
                "Ensure pipeline_fn returns retrieved documents.",
            )

        try:
            sims = self.embedding_judge.batch_similarity(non_empty[0], non_empty[1:])
            avg_doc_sim = sum(sims) / len(sims) if sims else 1.0
        except Exception:
            return ("Root cause analysis failed.", "")

        if avg_doc_sim < 0.7:
            return (
                f"Vocabulary mismatch — paraphrases retrieved semantically different "
                f"documents (avg doc similarity: {avg_doc_sim:.2f}). The embedding model "
                f"is sensitive to query phrasing.",
                "Add query expansion or use a retrieval model fine-tuned for your domain. "
                "Consider HyDE (hypothetical document embeddings) to normalise query representations.",
            )

        return (
            f"Retrieval is consistent (avg doc similarity: {avg_doc_sim:.2f}). "
            "Inconsistencies originate in the generation step — the LLM produces "
            "different answers from the same context.",
            "Add a stricter system prompt restricting the LLM to context only, "
            "and set temperature=0.",
        )
