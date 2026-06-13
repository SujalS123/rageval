# rageval/reporters/json_csv.py

import json
import csv
from pathlib import Path
from rageval.core.result import EvalResult


def to_json(results: list[EvalResult], path: str) -> None:
    """
    Save evaluation results to a JSON file.
    Each result is serialized using EvalResult.to_dict().
    """
    data = [r.to_dict() for r in results]
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Results saved to {path} ({len(results)} samples)")


def to_csv(results: list[EvalResult], path: str) -> None:
    """
    Save evaluation results to a CSV file.
    One row per sample. One column per metric score and pass/fail.
    Useful for loading into Excel or pandas for analysis.
    """
    if not results:
        print("No results to save.")
        return

    metric_names = list(results[0].metric_results.keys())

    fieldnames = (
        ["query", "answer", "overall_score", "passed", "latency_ms"]
        + [f"{m}_score" for m in metric_names]
        + [f"{m}_passed" for m in metric_names]
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "query": r.sample.query,
                "answer": r.sample.answer[:200],
                "overall_score": r.overall_score,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
            }
            for m in metric_names:
                mr = r.metric_results.get(m)
                row[f"{m}_score"] = mr.score if mr else ""
                row[f"{m}_passed"] = mr.passed if mr else ""

            writer.writerow(row)

    print(f"CSV saved to {path} ({len(results)} samples)")


def print_summary(results: list[EvalResult]) -> None:
    """
    Print a clean summary table to the terminal.
    Shows per-metric averages and pass rates across all samples.
    """
    from rageval.core.pipeline import summary as compute_summary

    if not results:
        print("No results to summarize.")
        return

    stats = compute_summary(results)

    print("\n" + "=" * 55)
    print(f"  EVALUATION SUMMARY — {stats['total_samples']} samples")
    print("=" * 55)
    print(f"  Overall pass rate : {stats['overall_pass_rate']*100:.1f}%")
    print(f"  Avg overall score : {stats['avg_overall_score']:.3f}")
    print("-" * 55)

    for name, m in stats["per_metric"].items():
        status = "PASS" if m["pass_rate"] >= 0.8 else "FAIL"
        print(
            f"  {name:<22} "
            f"avg={m['avg_score']:.3f}  "
            f"pass={m['pass_rate']*100:.0f}%  "
            f"[{status}]"
        )

    print("=" * 55)