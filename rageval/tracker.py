# rageval/tracker.py

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from rageval.core.result import EvalResult


class RunTracker:
    """
    Persists evaluation runs to a local SQLite database.

    Zero new dependencies — sqlite3 is in Python's standard library.
    One file on disk, no database server, no configuration.

    Once a team has 10 runs of history, they cannot remove rageval
    without losing that data. This is what turns users into permanent users.

    Usage:
        tracker = RunTracker()
        tracker.save_run("v2.3-deploy", results)
        tracker.list_runs()
        tracker.compare_runs("v2.2-deploy", "v2.3-deploy")
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
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_samples INTEGER NOT NULL,
                    overall_score REAL NOT NULL,
                    pass_rate REAL NOT NULL,
                    per_metric_json TEXT NOT NULL,
                    hallucination_types_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_run_name ON runs(run_name)
            """)

    def save_run(self, run_name: str, results: list[EvalResult]) -> None:
        """Persist a named run. Overwrites if run_name already exists."""
        if not results:
            raise ValueError("Cannot save an empty results list.")

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = passed / total
        overall_score = sum(r.overall_score for r in results) / total

        # Aggregate per-metric averages
        per_metric: dict[str, list[float]] = {}
        hallucination_counts: dict[str, int] = {}

        for result in results:
            for name, mr in result.metric_results.items():
                per_metric.setdefault(name, []).append(mr.score)
                # Count hallucination types from structured objects
                for h in mr.hallucinations:
                    key = h.type.value
                    hallucination_counts[key] = hallucination_counts.get(key, 0) + 1

        per_metric_summary = {
            name: round(sum(scores) / len(scores), 4)
            for name, scores in per_metric.items()
        }

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO runs
                    (run_name, timestamp, total_samples, overall_score, pass_rate,
                     per_metric_json, hallucination_types_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_name) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    total_samples=excluded.total_samples,
                    overall_score=excluded.overall_score,
                    pass_rate=excluded.pass_rate,
                    per_metric_json=excluded.per_metric_json,
                    hallucination_types_json=excluded.hallucination_types_json
            """, (
                run_name,
                datetime.now(timezone.utc).isoformat(),
                total,
                round(overall_score, 4),
                round(pass_rate, 4),
                json.dumps(per_metric_summary),
                json.dumps(hallucination_counts),
            ))

    def get_run(self, run_name: str) -> dict:
        """Return a single run by name. Raises KeyError if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_name = ?", (run_name,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Run '{run_name}' not found in tracker.")
        return self._row_to_dict(row)

    def list_runs(self) -> list[dict]:
        """Return all runs ordered by timestamp descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def compare_runs(self, run_a: str, run_b: str) -> dict:
        """
        Compare two runs. Returns per-metric deltas with direction and % change.

        run_a is the baseline, run_b is the newer run.
        direction: "improved" / "degraded" / "unchanged"
        """
        a = self.get_run(run_a)
        b = self.get_run(run_b)

        metrics_a = a["per_metric"]
        metrics_b = b["per_metric"]
        all_metrics = set(metrics_a) | set(metrics_b)

        metric_deltas = {}
        for name in sorted(all_metrics):
            score_a = metrics_a.get(name)
            score_b = metrics_b.get(name)
            if score_a is None or score_b is None:
                metric_deltas[name] = {"run_a": score_a, "run_b": score_b, "direction": "new_or_removed"}
                continue
            delta = round(score_b - score_a, 4)
            pct = round((delta / score_a) * 100, 1) if score_a != 0 else 0.0
            if delta > 0.005:
                direction = "improved"
            elif delta < -0.005:
                direction = "degraded"
            else:
                direction = "unchanged"
            metric_deltas[name] = {
                "run_a": score_a,
                "run_b": score_b,
                "delta": delta,
                "pct_change": pct,
                "direction": direction,
            }

        overall_delta = round(b["overall_score"] - a["overall_score"], 4)

        # Hallucination type changes
        h_a = a["hallucination_types"]
        h_b = b["hallucination_types"]
        all_types = set(h_a) | set(h_b)
        hallucination_deltas = {
            t: {"run_a": h_a.get(t, 0), "run_b": h_b.get(t, 0),
                "delta": h_b.get(t, 0) - h_a.get(t, 0)}
            for t in sorted(all_types)
        }

        return {
            "run_a": run_a,
            "run_b": run_b,
            "overall_delta": overall_delta,
            "metrics": metric_deltas,
            "hallucination_types": hallucination_deltas,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["per_metric"] = json.loads(d.pop("per_metric_json"))
        d["hallucination_types"] = json.loads(d.pop("hallucination_types_json"))
        return d
