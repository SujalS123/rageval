# rageval/autoeval.py

import random
import logging
import functools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AutoEval:
    """
    Production monitoring decorator for RAG pipelines.

    Samples a fraction of production queries, evaluates them in a background
    thread (zero latency impact), saves results to the regression tracker,
    and fires an alert when rolling average drops below threshold.

    The decorator is non-intrusive: if evaluation fails for any reason,
    the original function result is returned unchanged. Evaluation errors
    are logged but never propagated.

    Usage:
        autoeval = AutoEval(
            metrics=[Faithfulness(judge), ContextPrecision(judge)],
            sample_rate=0.05,
            alert_threshold=0.75,
            alert_fn=send_slack_alert,
        )

        @autoeval.monitor
        def handle_query(query: str, docs: list[str], answer: str) -> str:
            return answer

    The monitored function must accept keyword arguments:
        query, retrieved_docs, answer
    OR return a plain string (treated as answer, with query/docs from kwargs).
    """

    def __init__(
        self,
        metrics: list,
        sample_rate: float = 0.05,
        alert_threshold: float = 0.75,
        alert_fn: Optional[Callable] = None,
        tracker=None,
        window_size: int = 100,
    ):
        self.metrics = metrics
        self.sample_rate = sample_rate
        self.alert_threshold = alert_threshold
        self.alert_fn = alert_fn
        self.tracker = tracker
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._scores: deque = deque(maxlen=window_size)
        self._evaluated_count = 0

    def monitor(self, fn: Callable) -> Callable:
        """
        Decorator that wraps a function and evaluates sampled calls.

        The wrapped function must accept query, retrieved_docs, answer
        as keyword arguments (or positional in that order).
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)

            # Sample at configured rate
            if random.random() < self.sample_rate:
                self._executor.submit(
                    self._evaluate_safely, args, kwargs, result
                )

            return result

        return wrapper

    def get_live_stats(self) -> dict:
        """Return rolling stats for the last window_size evaluated samples."""
        scores = list(self._scores)
        if not scores:
            return {
                "evaluated_count": 0,
                "avg_score": None,
                "min_score": None,
                "max_score": None,
                "below_threshold": 0,
            }
        return {
            "evaluated_count": self._evaluated_count,
            "avg_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "below_threshold": sum(1 for s in scores if s < self.alert_threshold),
        }

    def _evaluate_safely(self, args: tuple, kwargs: dict, fn_result) -> None:
        """Run evaluation in the background. Never raises."""
        try:
            from rageval.core.sample import RAGSample
            from rageval.core.pipeline import evaluate

            # Try to extract query, docs, answer from kwargs first, then args
            query = kwargs.get("query") or (args[0] if len(args) > 0 else None)
            docs = kwargs.get("retrieved_docs") or kwargs.get("docs") or (args[1] if len(args) > 1 else None)
            answer = kwargs.get("answer") or (fn_result if isinstance(fn_result, str) else None)

            if not query or not docs or not answer:
                return

            sample = RAGSample(query=str(query), retrieved_docs=docs, answer=str(answer))
            eval_result = evaluate(sample=sample, metrics=self.metrics)

            self._evaluated_count += 1
            self._scores.append(eval_result.overall_score)

            # Save to tracker if configured
            if self.tracker is not None:
                try:
                    self.tracker.save_run(
                        f"autoeval-{self._evaluated_count}",
                        [eval_result],
                    )
                except Exception:
                    pass

            # Fire alert if rolling average drops below threshold
            stats = self.get_live_stats()
            if (
                self.alert_fn is not None
                and stats["avg_score"] is not None
                and stats["avg_score"] < self.alert_threshold
            ):
                try:
                    self.alert_fn(stats)
                except Exception:
                    pass

        except Exception as e:
            logger.debug("AutoEval background evaluation failed: %s", e)
