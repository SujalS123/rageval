# rageval/query_classifier.py

from enum import Enum
from rageval.judges.base import BaseJudge

CLASSIFICATION_PROMPT = """\
Classify the following query into exactly one of these types:

factual        — asks for a specific verifiable fact (who, what, when, where)
comparison     — asks to compare two or more things
multi_hop      — requires combining information from multiple sources to answer
time_sensitive — asks about current state, recent events, or time-dependent information
negation       — asks what is NOT true or what does NOT apply
procedural     — asks how to do something, step-by-step
ambiguous      — unclear intent, could mean multiple things
unanswerable   — asks for information that is unlikely to exist in any document

Query: {query}

Respond ONLY with a JSON object. No explanation. No markdown fences.
{{"query_type": "factual"}}
"""


class QueryType(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    TIME_SENSITIVE = "time_sensitive"
    NEGATION = "negation"
    PROCEDURAL = "procedural"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


class QueryClassifier:
    """
    Classifies queries by type and groups evaluation performance by type.

    A team that knows "our pipeline fails on comparison queries" can build
    a targeted fix. A team that only knows "faithfulness is 0.71" is guessing.

    Usage:
        classifier = QueryClassifier(judge=judge)
        q_type = classifier.classify("What is the capital of France?")
        # QueryType.FACTUAL

        breakdown = classifier.classify_batch(eval_results)
        # {"factual": {"count": 34, "avg_faithfulness": 0.94}, ...}
    """

    def __init__(self, judge: BaseJudge):
        self.judge = judge

    def classify(self, query: str) -> QueryType:
        """Classify a single query. Falls back to AMBIGUOUS on any failure."""
        try:
            result = self.judge.complete_json(
                CLASSIFICATION_PROMPT.format(query=query)
            )
            raw = result.get("query_type", "ambiguous").strip().lower()
            return QueryType(raw)
        except (Exception, ValueError):
            return QueryType.AMBIGUOUS

    def classify_batch(self, results: list) -> dict:
        """
        Classify all queries in a list of EvalResults and group
        per-metric averages by query type.

        Returns:
            {
                "factual": {
                    "count": 34,
                    "pass_rate": 0.91,
                    "avg_scores": {"faithfulness": 0.94, ...},
                    "samples": [...]   # list of EvalResult
                },
                ...
            }
        """
        from collections import defaultdict

        groups: dict[str, dict] = defaultdict(lambda: {
            "count": 0,
            "passed": 0,
            "metric_scores": defaultdict(list),
            "samples": [],
        })

        for result in results:
            q_type = self.classify(result.sample.query)
            key = q_type.value
            groups[key]["count"] += 1
            groups[key]["passed"] += int(result.passed)
            groups[key]["samples"].append(result)
            for name, mr in result.metric_results.items():
                groups[key]["metric_scores"][name].append(mr.score)

        summary = {}
        for q_type, data in sorted(groups.items()):
            avg_scores = {
                name: round(sum(scores) / len(scores), 4)
                for name, scores in data["metric_scores"].items()
            }
            summary[q_type] = {
                "count": data["count"],
                "pass_rate": round(data["passed"] / data["count"], 4) if data["count"] else 0.0,
                "avg_scores": avg_scores,
                "samples": data["samples"],
            }

        return summary
