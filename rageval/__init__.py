"""
rageval — Framework-agnostic RAG evaluation with debug-first scores.

Every metric returns a score AND the specific evidence that caused it.
Not just faithfulness: 0.43 — but which sentences hallucinated.

Quickstart:
    from rageval import evaluate, RAGSample
    from rageval.metrics.faithfulness import Faithfulness
    from rageval.judges.anthropic_judge import AnthropicJudge

    result = evaluate(
        sample=RAGSample(
            query="Who wrote Romeo and Juliet?",
            retrieved_docs=["Romeo and Juliet was written by William Shakespeare."],
            answer="Romeo and Juliet was written by Charles Dickens.",
        ),
        metrics=[Faithfulness(judge=AnthropicJudge(), threshold=0.8)],
    )
    print(result.summary())

Documentation: https://sujalsonawane.github.io/rageval
PyPI: https://pypi.org/project/rageval
"""

from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult, EvalResult
from rageval.core.pipeline import evaluate, batch_evaluate, summary
from rageval.tracker import RunTracker
from rageval.dataset import EvalDatasetGenerator
from rageval.autoeval import AutoEval
from rageval.query_classifier import QueryClassifier
from rageval.chunk_analyzer import ChunkQualityAnalyzer
from rageval.trace import RAGTracer
from rageval.consistency import ConsistencyAnalyzer
from rageval.taxonomy import FailureTaxonomyBuilder
from rageval.prompt_vc import PromptVersionControl
from rageval.drift import SemanticDriftDetector
from rageval.explainer import ExplainabilityReporter

__version__ = "0.5.0"

__all__ = [
    "RAGSample",
    "MetricResult",
    "EvalResult",
    "evaluate",
    "batch_evaluate",
    "summary",
    "RunTracker",
    "EvalDatasetGenerator",
    "AutoEval",
    "QueryClassifier",
    "ChunkQualityAnalyzer",
    "RAGTracer",
    "ConsistencyAnalyzer",
    "FailureTaxonomyBuilder",
    "PromptVersionControl",
    "SemanticDriftDetector",
    "ExplainabilityReporter",
    "__version__",
]
