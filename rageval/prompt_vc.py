# rageval/prompt_vc.py

import difflib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PromptComparisonReport:
    version_a: str
    version_b: str
    prompt_diff: str                        # unified diff string
    metric_deltas: dict = field(default_factory=dict)
    # {metric: {delta, direction, significant, score_a, score_b}}
    hallucination_type_deltas: dict = field(default_factory=dict)
    recommendation: str = "inconclusive"    # "deploy_b" | "keep_a" | "inconclusive"
    recommendation_reason: str = ""


class PromptVersionControl:
    """
    Stores system prompt versions in the same SQLite database as RunTracker
    and compares their evaluation performance using a two-sample z-test.

    Usage:
        pvc = PromptVersionControl()
        pvc.register("v1", "Answer ONLY from context.")
        pvc.register("v2", "Answer ONLY from context. Never invent details.")

        report = pvc.compare("v1", "v2", results_v1, results_v2)
        print(report.recommendation)  # "deploy_b"
    """

    def __init__(self, db_path: str = ".rageval/runs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_name TEXT NOT NULL UNIQUE,
                    prompt_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def register(self, version_name: str, prompt_text: str) -> None:
        """Store a prompt version. Overwrites if version_name already exists."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO prompt_versions (version_name, prompt_text, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(version_name) DO UPDATE SET
                    prompt_text=excluded.prompt_text,
                    created_at=excluded.created_at
            """, (version_name, prompt_text, datetime.now(timezone.utc).isoformat()))

    def get(self, version_name: str) -> str:
        """Return prompt text for a version. Raises KeyError if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT prompt_text FROM prompt_versions WHERE version_name = ?",
                (version_name,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Prompt version '{version_name}' not found.")
        return row["prompt_text"]

    def list_versions(self) -> list[dict]:
        """Return all versions ordered by creation time descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version_name, created_at FROM prompt_versions ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def compare(
        self,
        version_a: str,
        version_b: str,
        results_a: list,
        results_b: list,
    ) -> PromptComparisonReport:
        """
        Compare two prompt versions using per-metric score distributions.

        Statistical significance is determined by a two-sample z-test (p < 0.05).
        Recommendation logic:
        - deploy_b : at least one metric significantly improved, none significantly degraded
        - keep_a   : at least one metric significantly degraded
        - inconclusive : no significant changes detected
        """
        text_a = self.get(version_a)
        text_b = self.get(version_b)

        prompt_diff = "\n".join(difflib.unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            fromfile=version_a,
            tofile=version_b,
            lineterm="",
        ))

        # Aggregate per-metric score lists
        def scores_by_metric(results: list) -> dict[str, list[float]]:
            out: dict[str, list[float]] = {}
            for r in results:
                for name, mr in r.metric_results.items():
                    out.setdefault(name, []).append(mr.score)
            return out

        scores_a = scores_by_metric(results_a)
        scores_b = scores_by_metric(results_b)
        all_metrics = sorted(set(scores_a) | set(scores_b))

        metric_deltas: dict = {}
        has_improvement = False
        has_degradation = False

        for name in all_metrics:
            sa = scores_a.get(name, [])
            sb = scores_b.get(name, [])
            if not sa or not sb:
                metric_deltas[name] = {
                    "score_a": None, "score_b": None,
                    "delta": None, "direction": "insufficient_data",
                    "significant": False,
                }
                continue

            mean_a = sum(sa) / len(sa)
            mean_b = sum(sb) / len(sb)
            delta = round(mean_b - mean_a, 4)

            significant = self._z_test_significant(sa, sb)

            if delta > 0.005:
                direction = "improved"
                if significant:
                    has_improvement = True
            elif delta < -0.005:
                direction = "degraded"
                if significant:
                    has_degradation = True
            else:
                direction = "unchanged"

            metric_deltas[name] = {
                "score_a": round(mean_a, 4),
                "score_b": round(mean_b, 4),
                "delta": delta,
                "direction": direction,
                "significant": significant,
            }

        # Hallucination type deltas
        def hall_counts(results: list) -> dict[str, int]:
            counts: dict[str, int] = {}
            for r in results:
                for mr in r.metric_results.values():
                    for h in mr.hallucinations:
                        k = h.type.value
                        counts[k] = counts.get(k, 0) + 1
            return counts

        h_a = hall_counts(results_a)
        h_b = hall_counts(results_b)
        all_types = sorted(set(h_a) | set(h_b))
        hallucination_type_deltas = {
            t: {"count_a": h_a.get(t, 0), "count_b": h_b.get(t, 0),
                "delta": h_b.get(t, 0) - h_a.get(t, 0)}
            for t in all_types
        }

        # Recommendation
        if has_degradation:
            recommendation = "keep_a"
            reason = (
                f"{version_b} significantly degraded one or more metrics. "
                f"Keep {version_a}."
            )
        elif has_improvement:
            recommendation = "deploy_b"
            reason = (
                f"{version_b} significantly improved one or more metrics "
                f"with no significant regressions. Deploy {version_b}."
            )
        else:
            recommendation = "inconclusive"
            reason = (
                "No statistically significant differences detected between "
                f"{version_a} and {version_b}. Collect more samples."
            )

        return PromptComparisonReport(
            version_a=version_a,
            version_b=version_b,
            prompt_diff=prompt_diff,
            metric_deltas=metric_deltas,
            hallucination_type_deltas=hallucination_type_deltas,
            recommendation=recommendation,
            recommendation_reason=reason,
        )

    # ── Statistics ─────────────────────────────────────────────────────────

    @staticmethod
    def _z_test_significant(a: list[float], b: list[float], alpha: float = 0.05) -> bool:
        """
        Two-sample z-test for difference in means.
        Returns True if the difference is statistically significant (p < alpha).
        Falls back to False if either sample has zero variance.
        """
        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            return False

        mean_a = sum(a) / n_a
        mean_b = sum(b) / n_b

        var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return False

        z = abs(mean_b - mean_a) / se
        # z > 1.96 corresponds to p < 0.05 (two-tailed)
        return z > 1.96
