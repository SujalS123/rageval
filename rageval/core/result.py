from dataclasses import dataclass , field
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from rageval.core.sample import RAGSample
    from rageval.core.hallucination import Hallucination

@dataclass
class MetricResult:
    """
    The ouput of running one metric on one RAGSample.
    
    Design decision: score alone is useless for debugging.
    reasoning tells tells you why in one sentence.
    evidence tells you WhICH specific sentences/claims caused the score.
    
    Every other RAG eval tool stops at score.
    rageval surfaces evidence - this is the entire differentiator.

    """
    metric_name : str
    score : float      #always 0.0 to 1.0
    passed: bool     #True if score >= threshold
    reasoning : str  #one sentence - why is the score this value
    evidence: list[str]  #specific claims/chunks that caused the score
    threshold : float = 0.5
    hallucinations: list = field(default_factory=list)  # list[Hallucination]

    def __post_init__(self):
        # Clamp score to valid range
        self.score = round(max(0.0 , min(1.0 , self.score)) , 4)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] {self.metric_name} : {self.score:.2f} (threshold={self.threshold:.2f})",
            f"Reasoning: {self.reasoning}",
        ]
        if self.evidence:
            lines.append(" Evidence:")
            for e in self.evidence:
                lines.append(f" - {e}")
        else:
            lines.append(" Evidence: none (all checks passed)")
        return "\n".join(lines)
    
@dataclass
class EvalResult:
    """
    The complete output of evaluate() for one RAGSample
    Aggregates all MetricResult and computes an overall score.
    """
    sample: "RAGSample"
    metric_results: dict[str , MetricResult] = field(default_factory = dict)
    overall_score : float = 0.0
    passed : bool = True
    latency_ms: float = 0.0

    def failed_metrics(self) -> list[MetricResult]:
        """Return only the metrics that did not pass their threshold."""
        return [r for r in self.metric_results.values() if not r.passed]
    
    def summary(self) -> str:
        """Human-readable summary for terminal output. """
        status = "PASSED" if self.passed else "FAILED"
        divider = "-" * 55
        lines = [
            f"\n{divider}",
            f"Query: {self.sample.query[:80]}{'...' if len(self.sample.query) > 80 else ''}",
            f"Overall: {self.overall_score:.2f} | {status} | {self.latency_ms:.0f}ms",
            divider,
        ]
        for result in self.metric_results.values():
            lines.append(str(result))
        lines.append(divider)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Serailize to plain dict for JSON export. """
        return {
            "query": self.sample.query , 
            "answer" : self.sample.answer , 
            "overall_score": self.overall_score ,
            "passed" : self.passed ,
            "latency_ms" : round(self.latency_ms , 1),
            "metrics" : {
                name : {
                    "score": r.score ,
                    "passed": r.passed ,
                    "reasoning":r.reasoning,
                    "evidence":r.evidence,
                    "threshold":r.threshold,
                }
                for name, r in self.metric_results.items()
            }
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
        

