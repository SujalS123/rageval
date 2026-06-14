# rageval/trace.py

import time
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rageval.core.result import MetricResult


@dataclass
class TraceStep:
    name: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    retrieval_docs: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    metric_results: dict[str, MetricResult] = field(default_factory=dict)


@dataclass
class RAGTrace:
    trace_id: str
    steps: list[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    total_latency_ms: float = 0.0


class _ActiveTrace:
    """
    The object yielded by RAGTracer.trace(). Users call step(), log_output(),
    and log_retrieval() on this object inside the with-block.
    """

    def __init__(self, trace_id: str):
        self._trace = RAGTrace(trace_id=trace_id)
        self._current_step: Optional[TraceStep] = None
        self._start_time = time.monotonic()
        self._step_start: float = 0.0

    def step(self, name: str) -> None:
        """Mark the start of a new pipeline step."""
        if self._current_step is not None:
            self._current_step.latency_ms = round(
                (time.monotonic() - self._step_start) * 1000, 1
            )
        self._current_step = TraceStep(name=name)
        self._trace.steps.append(self._current_step)
        self._step_start = time.monotonic()

    def log_output(self, text: str) -> None:
        """Log a text output from the current step."""
        if self._current_step is None:
            self.step("default")
        self._current_step.outputs["text"] = text
        self._trace.final_answer = text

    def log_retrieval(self, docs: list, scores: list[float] = None) -> None:
        """Log retrieved documents (list[str] or list[RetrievedDoc]) for the current step."""
        if self._current_step is None:
            self.step("retrieval")
        # Normalise to plain strings
        from rageval.core.retrieved_doc import RetrievedDoc
        texts = [d.content if isinstance(d, RetrievedDoc) else d for d in docs]
        self._current_step.retrieval_docs = texts
        self._current_step.retrieval_scores = scores or []

    def _finalise(self) -> RAGTrace:
        if self._current_step is not None:
            self._current_step.latency_ms = round(
                (time.monotonic() - self._step_start) * 1000, 1
            )
        self._trace.total_latency_ms = round(
            (time.monotonic() - self._start_time) * 1000, 1
        )
        return self._trace


class RAGTracer:
    """
    Step-level pipeline tracer.

    Wraps any RAG pipeline with a context manager. Captures inputs, outputs,
    and retrieved docs at each step. Evaluates each retrieval step independently
    and identifies the root cause — the earliest step where quality drops.

    Usage:
        tracer = RAGTracer(metrics=[ContextPrecision(judge), Faithfulness(judge)])

        with tracer.trace("query-123") as t:
            t.step("retrieval")
            docs = retriever.search(query)
            t.log_retrieval(docs, scores=[0.91, 0.72])

            t.step("generation")
            answer = llm.generate(query, docs)
            t.log_output(answer)

        result = tracer.evaluate_trace(t._trace)
    """

    def __init__(self, metrics: list = None, db_path: str = ".rageval/runs.db"):
        self.metrics = metrics or []
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
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    final_answer TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    root_cause TEXT
                )
            """)

    @contextmanager
    def trace(self, trace_id: str):
        """
        Context manager that captures all pipeline steps.

        Yields an _ActiveTrace object. After the with-block exits,
        the completed trace is returned via the yielded object's _trace attribute.
        """
        active = _ActiveTrace(trace_id)
        try:
            yield active
        finally:
            active._finalise()

    def evaluate_trace(self, trace: RAGTrace) -> dict:
        """
        Run metrics on each step that has retrieval_docs, and Faithfulness
        on the final answer. Returns per-step results and identifies root cause.
        """
        from rageval.core.sample import RAGSample
        from rageval.metrics.context_precision import ContextPrecision
        from rageval.metrics.faithfulness import Faithfulness

        step_results: dict[str, dict] = {}
        root_cause: Optional[str] = None

        # Separate metrics by type for routing
        precision_metrics = [m for m in self.metrics if isinstance(m, ContextPrecision)]
        faith_metrics = [m for m in self.metrics if isinstance(m, Faithfulness)]

        # Use first available query text from step inputs, fallback to empty
        query = ""
        for step in trace.steps:
            q = step.inputs.get("query", "")
            if q:
                query = q
                break

        for step in trace.steps:
            if not step.retrieval_docs:
                continue

            # We need a minimal RAGSample to run metrics on this step
            # Use the final answer as a proxy when query isn't per-step
            step_query = step.inputs.get("query", query) or "evaluation query"
            # Need a non-empty answer for RAGSample — use a placeholder if this is a retrieval-only step
            step_answer = step.outputs.get("text", "placeholder answer for retrieval evaluation")

            try:
                sample = RAGSample(
                    query=step_query,
                    retrieved_docs=step.retrieval_docs,
                    answer=step_answer,
                )
            except ValueError:
                continue

            step_metric_results: dict[str, MetricResult] = {}

            for metric in precision_metrics:
                try:
                    result = metric.score(sample)
                    step_metric_results[metric.name] = result
                    step.metric_results[metric.name] = result
                except Exception:
                    pass

            step_results[step.name] = {
                "latency_ms": step.latency_ms,
                "doc_count": len(step.retrieval_docs),
                "scores": {name: r.score for name, r in step_metric_results.items()},
                "passed": all(r.passed for r in step_metric_results.values()),
                "evidence": {name: r.evidence for name, r in step_metric_results.items()},
            }

            # Root cause = first step that fails
            if root_cause is None and not step_results[step.name]["passed"]:
                root_cause = step.name

        # Evaluate final answer faithfulness if we have docs and answer
        all_docs = []
        for step in trace.steps:
            all_docs.extend(step.retrieval_docs)

        if trace.final_answer and all_docs and faith_metrics:
            try:
                final_sample = RAGSample(
                    query=query or "evaluation query",
                    retrieved_docs=all_docs,
                    answer=trace.final_answer,
                )
                for metric in faith_metrics:
                    result = metric.score(final_sample)
                    step_results["__generation__"] = {
                        "latency_ms": 0.0,
                        "doc_count": len(all_docs),
                        "scores": {metric.name: result.score},
                        "passed": result.passed,
                        "evidence": {metric.name: result.evidence},
                    }
                    if root_cause is None and not result.passed:
                        root_cause = "generation"
            except Exception:
                pass

        self._save_trace(trace, root_cause)

        return {
            "trace_id": trace.trace_id,
            "total_latency_ms": trace.total_latency_ms,
            "steps": step_results,
            "root_cause": root_cause,
            "root_cause_message": _root_cause_message(root_cause),
        }

    def get_trace(self, trace_id: str) -> dict:
        """Retrieve a saved trace by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Trace '{trace_id}' not found.")
        return self._row_to_dict(row)

    def list_traces(self, limit: int = 20) -> list[dict]:
        """Return most recent traces."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT trace_id, timestamp, total_latency_ms, root_cause "
                "FROM traces ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _save_trace(self, trace: RAGTrace, root_cause: Optional[str]) -> None:
        steps_data = [
            {
                "name": s.name,
                "latency_ms": s.latency_ms,
                "doc_count": len(s.retrieval_docs),
                "retrieval_scores": s.retrieval_scores,
                "outputs": s.outputs,
            }
            for s in trace.steps
        ]
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO traces
                    (trace_id, timestamp, total_latency_ms, final_answer, steps_json, root_cause)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    total_latency_ms=excluded.total_latency_ms,
                    final_answer=excluded.final_answer,
                    steps_json=excluded.steps_json,
                    root_cause=excluded.root_cause
            """, (
                trace.trace_id,
                datetime.now(timezone.utc).isoformat(),
                trace.total_latency_ms,
                trace.final_answer,
                json.dumps(steps_data),
                root_cause,
            ))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["steps"] = json.loads(d.pop("steps_json"))
        return d


def _root_cause_message(root_cause: Optional[str]) -> str:
    if root_cause is None:
        return "All steps passed — no quality issues detected."
    if root_cause == "generation":
        return (
            "ROOT CAUSE: Generation step. Retrieval was clean. "
            "The LLM introduced hallucinations not present in context. "
            "Fix: tighten system prompt to restrict LLM to context only."
        )
    return (
        f"ROOT CAUSE: '{root_cause}' step. Quality dropped here first. "
        f"Inspect the retrieved documents at this step."
    )
