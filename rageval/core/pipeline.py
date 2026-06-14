# rageval/core/pipeline.py

import time
from rageval.core.sample import RAGSample
from rageval.core.result import EvalResult, MetricResult
from rageval.metrics.base import BaseMetric

def evaluate(
        sample: RAGSample,
        metrics: list[BaseMetric],
        weights: dict[str,float] = None,
)-> EvalResult:
    """
    Run all metrics on one RAGSample. Never raises - errors stores in MetricResult.
    
    Args:
        sample: The RAG interaction to evaluate
        metrics: List of metric instances to run
        weights: Optional dicr of {metric_name:weight} for overall score.
                 Default is equal weights for all metrics.
    
    Returns:
    EvalResult with per-metric scores , overall score , and latency

    Examples:
        from rageval.core.pipeline import evaluate
        from rageval.core.sample import RAGSample
        from rageval.metrics.faithfulness import Faithfulness
        from rageval.juges.openai_judge import OpenAIJudge
        
        judge = OpenAIJudge()
        result = evaluate(
            sample = RAGSample(
                query="what is python?",
                retrieved_docs=['Python is a programming language..'],
                answer='Python is a high-level programming language.',
                ),
                metrics=[Faithfulness(judge=judge , threshold = 0.8)],

            )
            print(result.summary())
    """

    if not metrics:
        raise ValueError(
            "You must provide at least one metric. "
            "Example: metrivs=[Faithfulness(judge=judge)]"
        )
    
    start_time  = time.time()
    metric_results = {}

    for metric in metrics:
        try:
            result = metric.score(sample)
            metric_results[metric.name] = result
        except Exception as e:
            # One metric failing never stops the others from running
            # The error is stores in MetricResult so users can debug it

            metric_results[metric.name] = MetricResult(
                metric_name=metric.name,
                score=0.0,
                passed= False,
                reasoning=f"Metric crashed: {type(e).__name__}: {str(e)}",
                evidence=[],
                threshold=metric.threshold,
            )

    latency_ms = (time.time() - start_time) * 1000

    # Compute weighted overall score
    w = weights or {m.name: 1.0 for m in metrics}
    total_weight = sum(w.get(name , 1.0) for name in metric_results)
    
    if total_weight > 0:
        overall_score = sum(
            r.score * w.get(name , 1.0)
            for name , r in metric_results.items()      
        ) / total_weight
    else:
        overall_score = 0.0
    all_passed = all(r.passed for r in metric_results.values())

    return EvalResult(
        sample=sample,
        metric_results=metric_results,
        overall_score=round(overall_score, 4),
        passed=all_passed,
        latency_ms=round(latency_ms,1),
    )
    
async def aevaluate(
        sample: RAGSample,
        metrics: list[BaseMetric],
        weights: dict[str,float] = None,
) -> EvalResult:
    """
    Async version of evaluate(). Use in FastAPI, async pipelines.
    Runs all metrics concurrently using asyncio.gather.
    """
    import asyncio
    import time

    if not metrics:
        raise ValueError(
            "You must provide at least one metric. "
            "Example: metrics=[Faithfulness(judge=judge)]"
        )
    
    start_time = time.time()
    
    async def safe_ascore(metric):
        try:
            return metric.name, await metric.ascore(sample)
        except Exception as e:
            return metric.name, MetricResult(
                metric_name=metric.name,
                score=0.0,
                passed=False,
                reasoning=f"Metric crashed: {type(e).__name__}: {str(e)}",
                evidence=[],
                threshold=metric.threshold,
            )

    tasks = [safe_ascore(metric) for metric in metrics]
    results = await asyncio.gather(*tasks)
    
    metric_results = dict(results)
    latency_ms = (time.time() - start_time) * 1000

    # Compute weighted overall score
    w = weights or {m.name: 1.0 for m in metrics}
    total_weight = sum(w.get(name, 1.0) for name in metric_results)
    
    if total_weight > 0:
        overall_score = sum(
            r.score * w.get(name, 1.0)
            for name, r in metric_results.items()      
        ) / total_weight
    else:
        overall_score = 0.0
        
    all_passed = all(r.passed for r in metric_results.values())

    return EvalResult(
        sample=sample,
        metric_results=metric_results,
        overall_score=round(overall_score, 4),
        passed=all_passed,
        latency_ms=round(latency_ms, 1),
    )

def batch_evaluate(
        samples: list[RAGSample],
        metrics : list[BaseMetric],
        weights: dict[str,float] = None,
        max_workers: int = 4,
        show_progress: bool = True,
) -> list[EvalResult]:
    """
    Evaluate a list of RAGSample in parallel using a thread pool
    
    Args;
        samples: List of RAGSamples to evaluate
        metrics: Mteric instances to run every sample
        weights: Optional metric weights for overall score
        max_workers: Number of parallel threads (Default 4)
                     keep low to avoid LLM rate limits
        show_progress: Print progress to terminal
        
    Returns:
        List of EvalResult in the same order as input samples
    """

    from concurrent.futures import ThreadPoolExecutor , as_completed

    if not samples:
        return []
    
    results = [None] * len(samples)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(evaluate , sample , metrics , weights): i
            for i , sample in enumerate(samples)
        }

        completed = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception:
                # Should never happen since evaluate() never raises
                # But just incase , create a failes result
                results[i] = EvalResult(
                    sample=samples[i],
                    metric_results={},
                    overall_score=0.0,
                    passed=False,
                    latency_ms=0.0,
                )
            completed += 1
            if show_progress:
                print(f" Evaluated {completed}/{len(samples)} samples", end='\r')

    if show_progress:
        print(f" Evaluated {len(samples)}/{len(samples)} samples - done.")
    
    return results
    
def summary(results: list[EvalResult]) -> dict:
    """
    Compute aggregate statistics across a batch of results.
    
    Returns overall pass rate and per-metric averages.
    Use this to print final report or check CI threshols.
    """
    if not results:
        return {}
    
    n = len(results)
    metric_names = list(results[0].metric_results.keys())

    per_metric = {}
    for name in metric_names:
        scores = [
            r.metric_results[name].score
            for r in results
            if name in r.metric_results
        ]
        passed = [
            r.metric_results[name].passed
            for r in results
            if name in r.metric_results
        ]
        per_metric[name] = {
            "avg_score": round(sum(scores) / len(scores) , 4) if scores else 0.0,
            "pass_rate": round(sum(passed) / len(passed) , 4) if passed else 0.0,
            "samples": len(scores),
        }

    return{
        "total_samples" : n ,
        "overall_pass_rate": round(sum(r.passed for r in results) / n , 4),
        "avg_overall_score": round(sum(r.overall_score for r in results) /n , 4),
        "per_metric":per_metric,
    }
    