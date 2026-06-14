# rageval/metrics/base.py

from abc import ABC, abstractmethod
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult
from rageval.judges.base import BaseJudge


class BaseMetric(ABC):
    """
    Abstract base class for all rageval metrics.

    Every metric in this library inherits from this class.
    The contract:
    - Declare which inputs you need in required_inputs
    - Implement score() and return a MetricResult
    - Call self.validate(sample) at the start of score()
    - Call self._make_result() to build the MetricResult

    Why ABC: if a subclass does not implement score(), Python raises
    TypeError the moment you try to instantiate it — at startup,
    not during an evaluation run when it is too late.
    """

    name: str = "base"
    required_inputs: list[str] = ["query", "retrieved_docs", "answer"]

    def __init__(self, judge: BaseJudge, threshold: float = 0.5):
        self.judge = judge
        self.threshold = threshold

    def validate(self, sample: RAGSample) -> None:
        """
        Check that all required inputs are present in the sample.
        Raises ValueError with a clear message if anything is missing.

        Call this at the start of every score() implementation.
        It catches missing ground_truth, empty doc lists, etc.
        before any LLM calls are made — saving tokens and time.
        """
        for field_name in self.required_inputs:
            value = getattr(sample, field_name, None)

            if value is None:
                raise ValueError(
                    f"Metric '{self.name}' requires '{field_name}' "
                    f"but RAGSample.{field_name} is None.\n"
                    f"This metric needs: {self.required_inputs}\n"
                    f"Pass '{field_name}' when constructing your RAGSample."
                )

            if isinstance(value, list) and len(value) == 0:
                raise ValueError(
                    f"Metric '{self.name}' requires '{field_name}' "
                    f"but it is an empty list.\n"
                    f"Make sure your retriever is returning documents."
                )

            if isinstance(value, str) and not value.strip():
                raise ValueError(
                    f"Metric '{self.name}' requires '{field_name}' "
                    f"but it is an empty string."
                )

    @abstractmethod
    def score(self, sample: RAGSample) -> MetricResult:
        """
        Compute the metric score for the given sample.

        Must:
        1. Call self.validate(sample) first
        2. Return a MetricResult using self._make_result()

        Must NOT:
        - Raise exceptions for LLM errors (catch them, return score 0.0)
        - Return None
        - Return a raw float
        """
        raise NotImplementedError

    async def ascore(self, sample: RAGSample) -> MetricResult:
        """
        Async version of score. 
        Default implementation runs the synchronous score() in a thread pool.
        Subclasses can override this for native async LLM calls.
        """
        import asyncio
        return await asyncio.to_thread(self.score, sample)

    def _make_result(
        self,
        score: float,
        reasoning: str,
        evidence: list[str],
        hallucinations: list = None,
    ) -> MetricResult:
        """
        Helper to build a MetricResult consistently.

        Every metric uses this instead of constructing MetricResult directly.
        This ensures threshold and metric_name are always set correctly,
        and score is always clamped to [0.0, 1.0].
        """
        return MetricResult(
            metric_name=self.name,
            score=round(max(0.0, min(1.0, float(score))), 4),
            passed=score >= self.threshold,
            reasoning=reasoning,
            evidence=evidence,
            threshold=self.threshold,
            hallucinations=hallucinations or [],
        )