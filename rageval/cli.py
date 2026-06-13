# rageval/cli.py

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(
    name="rageval",
    help="rageval — framework-agnostic RAG pipeline evaluator",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    data: Path = typer.Argument(
        ...,
        help="Path to JSON file with evaluation samples",
    ),
    output: Path = typer.Option(
        "results.json",
        "--output", "-o",
        help="Path to save results JSON",
    ),
    judge: str = typer.Option(
        "openai",
        "--judge", "-j",
        help="Judge backend: openai, anthropic, gemini, ollama, cohere, groq",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model", "-m",
        help="Model name for the judge",
    ),
    metrics: str = typer.Option(
        "faithfulness,context_precision,answer_relevancy",
        "--metrics",
        help="Comma-separated list of metrics to run",
    ),
    threshold: float = typer.Option(
        0.7,
        "--threshold", "-t",
        help="Pass/fail threshold applied to all metrics",
    ),
    workers: int = typer.Option(
        4,
        "--workers", "-w",
        help="Number of parallel workers for batch evaluation",
    ),
    save_run: Optional[str] = typer.Option(
        None,
        "--save-run",
        help="Name this run and save it to the regression tracker (e.g. 'v2.3-deploy')",
    ),
):
    """
    Run evaluation on a JSON dataset.

    The JSON file must be a list of objects with these fields:
      - query (required)
      - retrieved_docs (required, list of strings)
      - answer (required)
      - ground_truth (optional)

    Example:
      rageval run eval_data.json --judge anthropic --model claude-haiku-4-5
    """
    from rageval.core.sample import RAGSample
    from rageval.core.pipeline import batch_evaluate
    from rageval.reporters.json_csv import to_json, print_summary

    # ── Load data ────────────────────────────────────────────────────────
    if not data.exists():
        console.print(f"[red]Error: file not found: {data}[/red]")
        raise typer.Exit(1)

    try:
        raw = json.loads(data.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: invalid JSON in {data}: {e}[/red]")
        raise typer.Exit(1)

    if not isinstance(raw, list):
        console.print("[red]Error: JSON file must contain a list of samples[/red]")
        raise typer.Exit(1)

    samples = []
    for i, s in enumerate(raw):
        try:
            samples.append(RAGSample(
                query=s["query"],
                retrieved_docs=s["retrieved_docs"],
                answer=s["answer"],
                ground_truth=s.get("ground_truth"),
            ))
        except (KeyError, ValueError) as e:
            console.print(f"[red]Error in sample {i}: {e}[/red]")
            raise typer.Exit(1)

    # ── Build judge ──────────────────────────────────────────────────────
    if judge == "openai":
        from rageval.judges.openai_judge import OpenAIJudge
        judge_obj = OpenAIJudge(model=model)
    elif judge == "anthropic":
        from rageval.judges.anthropic_judge import AnthropicJudge
        judge_obj = AnthropicJudge(model=model)
    elif judge == "gemini":
        from rageval.judges.gemini_judge import GeminiJudge
        judge_obj = GeminiJudge(model=model)
    elif judge == "ollama":
        from rageval.judges.ollama_judge import OllamaJudge
        judge_obj = OllamaJudge(model=model)
    elif judge == "cohere":
        from rageval.judges.cohere_judge import CohereJudge
        judge_obj = CohereJudge(model=model)
    elif judge == "groq":
        from rageval.judges.groq_judge import GroqJudge
        judge_obj = GroqJudge(model=model)
    else:
        console.print(f"[red]Unknown judge: {judge}. Use: openai, anthropic, gemini, ollama, cohere, groq[/red]")
        raise typer.Exit(1)

    # ── Build metrics ────────────────────────────────────────────────────
    from rageval.metrics.faithfulness import Faithfulness
    from rageval.metrics.context_precision import ContextPrecision
    from rageval.metrics.answer_relevancy import AnswerRelevancy
    from rageval.metrics.context_recall import ContextRecall
    from rageval.metrics.noise_sensitivity import NoiseSensitivity
    from rageval.metrics.answer_completeness import AnswerCompleteness
    from rageval.metrics.contradiction_detector import ContradictionDetector

    metric_registry = {
        "faithfulness": Faithfulness(judge_obj, threshold=threshold),
        "context_precision": ContextPrecision(judge_obj, threshold=threshold),
        "answer_relevancy": AnswerRelevancy(judge_obj, threshold=threshold),
        "context_recall": ContextRecall(judge_obj, threshold=threshold),
        "noise_sensitivity": NoiseSensitivity(judge_obj, noise_corpus=[], threshold=threshold),
        "answer_completeness": AnswerCompleteness(judge_obj, threshold=threshold),
        "contradiction_detector": ContradictionDetector(judge_obj, threshold=threshold),
    }

    selected_metrics = []
    for name in metrics.split(","):
        name = name.strip()
        if name not in metric_registry:
            console.print(f"[yellow]Warning: unknown metric '{name}', skipping[/yellow]")
            continue
        selected_metrics.append(metric_registry[name])

    if not selected_metrics:
        console.print("[red]Error: no valid metrics selected[/red]")
        raise typer.Exit(1)

    # ── Run evaluation ───────────────────────────────────────────────────
    console.print(
        f"\nRunning [bold]{len(selected_metrics)} metrics[/bold] "
        f"on [bold]{len(samples)} samples[/bold] "
        f"using [bold]{judge}/{model}[/bold]...\n"
    )

    results = batch_evaluate(
        samples=samples,
        metrics=selected_metrics,
        max_workers=workers,
        show_progress=True,
    )

    # ── Save results ─────────────────────────────────────────────────────
    to_json(results, str(output))

    # ── Print summary table ──────────────────────────────────────────────
    from rageval.core.pipeline import summary as compute_summary
    stats = compute_summary(results)

    table = Table(
        title="Evaluation Results",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Avg Score", justify="center")
    table.add_column("Pass Rate", justify="center")
    table.add_column("Status", justify="center")

    for name, m in stats["per_metric"].items():
        passed = m["pass_rate"] >= threshold
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(
            name,
            f"{m['avg_score']:.3f}",
            f"{m['pass_rate']*100:.0f}%",
            status,
        )

    console.print(table)
    console.print(f"Overall pass rate: {stats['overall_pass_rate']*100:.1f}%")
    console.print(f"Results saved to: {output}\n")

    # ── Save run to tracker ──────────────────────────────────────────────
    if save_run:
        from rageval.tracker import RunTracker
        tracker = RunTracker()
        tracker.save_run(save_run, results)
        console.print(f"[green]Run '{save_run}' saved to regression tracker.[/green]")

    # ── Exit code for CI ─────────────────────────────────────────────────
    if stats["overall_pass_rate"] < threshold:
        console.print("[red]Evaluation FAILED — scores below threshold[/red]")
        raise typer.Exit(1)

    console.print("[green]Evaluation PASSED[/green]")


@app.command()
def history(
    db: str = typer.Option(".rageval/runs.db", "--db", help="Path to tracker database"),
):
    """List all saved evaluation runs with per-metric averages."""
    from rageval.tracker import RunTracker
    tracker = RunTracker(db_path=db)
    runs = tracker.list_runs()

    if not runs:
        console.print("[yellow]No runs saved yet. Use --save-run when running evaluation.[/yellow]")
        return

    # Collect all metric names across all runs for column headers
    all_metrics: list[str] = []
    for r in runs:
        for m in r["per_metric"]:
            if m not in all_metrics:
                all_metrics.append(m)

    table = Table(title="Evaluation Run History", box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Run", style="bold")
    table.add_column("Date", style="dim")
    for m in all_metrics:
        table.add_column(m.replace("_", " ").title(), justify="center")
    table.add_column("Overall", justify="center")
    table.add_column("Pass Rate", justify="center")
    table.add_column("Samples", justify="right")

    for run in runs:
        date = run["timestamp"][:10]
        metric_cols = [f"{run['per_metric'].get(m, '-'):.3f}" if run["per_metric"].get(m) is not None else "-" for m in all_metrics]
        pass_color = "green" if run["pass_rate"] >= 0.8 else "red"
        table.add_row(
            run["run_name"],
            date,
            *metric_cols,
            f"{run['overall_score']:.3f}",
            f"[{pass_color}]{run['pass_rate']*100:.0f}%[/{pass_color}]",
            str(run["total_samples"]),
        )

    console.print(table)


@app.command()
def diff(
    run_a: str = typer.Argument(..., help="Baseline run name"),
    run_b: str = typer.Argument(..., help="Comparison run name"),
    db: str = typer.Option(".rageval/runs.db", "--db", help="Path to tracker database"),
):
    """Show metric changes between two saved runs."""
    from rageval.tracker import RunTracker
    tracker = RunTracker(db_path=db)

    try:
        comparison = tracker.compare_runs(run_a, run_b)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Changes from {run_a} to {run_b}[/bold]\n")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column(run_a, justify="center")
    table.add_column(run_b, justify="center")
    table.add_column("Delta", justify="center")
    table.add_column("Change %", justify="center")
    table.add_column("Status", justify="center")

    for name, m in comparison["metrics"].items():
        if m.get("direction") == "new_or_removed":
            table.add_row(name, str(m["run_a"]), str(m["run_b"]), "-", "-", "[yellow]N/A[/yellow]")
            continue
        delta_str = f"{m['delta']:+.4f}"
        pct_str = f"{m['pct_change']:+.1f}%"
        if m["direction"] == "improved":
            status = "[green]IMPROVED[/green]"
        elif m["direction"] == "degraded":
            status = "[red]DEGRADED[/red]"
        else:
            status = "[dim]unchanged[/dim]"
        table.add_row(name, f"{m['run_a']:.4f}", f"{m['run_b']:.4f}", delta_str, pct_str, status)

    console.print(table)
    overall_delta = comparison["overall_delta"]
    color = "green" if overall_delta >= 0 else "red"
    console.print(f"Overall score delta: [{color}]{overall_delta:+.4f}[/{color}]\n")

    if comparison["hallucination_types"]:
        console.print("[bold]Hallucination types:[/bold]")
        for t, h in comparison["hallucination_types"].items():
            delta = h["delta"]
            color = "green" if delta <= 0 else "red"
            console.print(f"  {t}: {h['run_a']} -> {h['run_b']}  [{color}]{delta:+d}[/{color}]")


@app.command(name="init-ci")
def init_ci(
    judge: str = typer.Option(
        "openai",
        "--judge", "-j",
        help="Judge backend to use in the workflow: openai, anthropic, groq, etc.",
    ),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="Pass/fail threshold"),
    eval_file: str = typer.Option("eval_data.json", "--eval-file", help="Eval data file path"),
):
    """Generate a GitHub Actions workflow file for CI/CD RAG evaluation."""
    import os

    secret_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "groq": "GROQ_API_KEY",
        "ollama": None,
    }
    secret = secret_map.get(judge, "OPENAI_API_KEY")
    env_block = ""
    if secret:
        env_block = f"\n        env:\n          {secret}: ${{{{ secrets.{secret} }}}}"

    workflow = f"""name: RAG Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install rageval
      - run: rageval run {eval_file} --judge {judge} --threshold {threshold}{env_block}
"""

    out = Path(".github/workflows")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "rag-eval.yml"
    target.write_text(workflow, encoding="utf-8")
    console.print(f"[green]Created {target}[/green]")
    console.print("Commit this file to enable RAG evaluation on every push.")


@app.command(name="traces")
def list_traces(
    db: str = typer.Option(".rageval/runs.db", "--db", help="Path to tracker database"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent traces to show"),
):
    """List recent pipeline traces."""
    from rageval.trace import RAGTracer
    tracer = RAGTracer(db_path=db)
    rows = tracer.list_traces(limit=limit)

    if not rows:
        console.print("[yellow]No traces saved yet.[/yellow]")
        return

    table = Table(title="Recent Traces", box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Trace ID", style="bold")
    table.add_column("Date", style="dim")
    table.add_column("Latency", justify="right")
    table.add_column("Root Cause", justify="left")

    for row in rows:
        root = row.get("root_cause") or "[green]none[/green]"
        table.add_row(
            row["trace_id"],
            row["timestamp"][:10],
            f"{row['total_latency_ms']:.0f}ms",
            root,
        )
    console.print(table)


@app.command(name="trace")
def show_trace(
    trace_id: str = typer.Argument(..., help="Trace ID to inspect"),
    db: str = typer.Option(".rageval/runs.db", "--db", help="Path to tracker database"),
):
    """Show the full step breakdown for a saved trace."""
    from rageval.trace import RAGTracer
    tracer = RAGTracer(db_path=db)

    try:
        t = tracer.get_trace(trace_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Trace: {t['trace_id']}[/bold]")
    console.print(f"Latency: {t['total_latency_ms']:.0f}ms | Root cause: {t.get('root_cause') or 'none'}\n")

    for i, step in enumerate(t["steps"], 1):
        console.print(f"  Step {i}: [bold]{step['name']}[/bold] ({step['latency_ms']:.0f}ms)")
        if step["doc_count"]:
            console.print(f"    Docs retrieved: {step['doc_count']}")
        if step.get("retrieval_scores"):
            scores_str = ", ".join(f"{s:.2f}" for s in step["retrieval_scores"])
            console.print(f"    Scores: [{scores_str}]")


@app.command(name="serve")
def serve():
    """Start the rageval MCP server (requires pip install rageval[mcp])."""
    import asyncio
    try:
        from rageval.mcp_server import create_server
        from mcp.server.stdio import stdio_server
    except ImportError:
        console.print("[red]MCP server requires: pip install rageval[mcp][/red]")
        raise typer.Exit(1)

    server = create_server()
    console.print("[green]rageval MCP server running on stdio[/green]")

    async def _run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


prompts_app = typer.Typer(help="Prompt version control commands.")
app.add_typer(prompts_app, name="prompts")


@prompts_app.command(name="list")
def prompts_list(
    db: str = typer.Option(".rageval/runs.db", "--db", help="Path to tracker database"),
):
    """List all registered prompt versions."""
    from rageval.prompt_vc import PromptVersionControl
    pvc = PromptVersionControl(db_path=db)
    versions = pvc.list_versions()
    if not versions:
        console.print("[yellow]No prompt versions registered yet.[/yellow]")
        return
    table = Table(title="Prompt Versions", box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Version", style="bold")
    table.add_column("Created", style="dim")
    for v in versions:
        table.add_row(v["version_name"], v["created_at"][:10])
    console.print(table)


@prompts_app.command(name="register")
def prompts_register(
    name: str = typer.Argument(..., help="Version name, e.g. 'v2'"),
    db: str = typer.Option(".rageval/runs.db", "--db", help="Path to tracker database"),
):
    """Register a prompt version from stdin."""
    import sys
    from rageval.prompt_vc import PromptVersionControl
    console.print("Paste prompt text, then press Ctrl+D (Unix) or Ctrl+Z (Windows):")
    prompt_text = sys.stdin.read()
    pvc = PromptVersionControl(db_path=db)
    pvc.register(name, prompt_text)
    console.print(f"[green]Registered prompt version '{name}'.[/green]")


@prompts_app.command(name="compare")
def prompts_compare(
    version_a: str = typer.Argument(..., help="Baseline prompt version"),
    version_b: str = typer.Argument(..., help="Comparison prompt version"),
    results_a: Path = typer.Option(..., "--results-a", help="JSON results file for version A"),
    results_b: Path = typer.Option(..., "--results-b", help="JSON results file for version B"),
    db: str = typer.Option(".rageval/runs.db", "--db", help="Path to tracker database"),
):
    """Compare two prompt versions using evaluation results."""
    from rageval.prompt_vc import PromptVersionControl
    from rageval.reporters.json_csv import load_results
    pvc = PromptVersionControl(db_path=db)
    try:
        ra = load_results(str(results_a))
        rb = load_results(str(results_b))
        report = pvc.compare(version_a, version_b, ra, rb)
    except (KeyError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Prompt comparison: {version_a} vs {version_b}[/bold]")
    color = {"deploy_b": "green", "keep_a": "red"}.get(report.recommendation, "yellow")
    console.print(f"Recommendation: [{color}]{report.recommendation.upper()}[/{color}]")
    console.print(f"Reason: {report.recommendation_reason}\n")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column(version_a, justify="center")
    table.add_column(version_b, justify="center")
    table.add_column("Delta", justify="center")
    table.add_column("Significant", justify="center")
    for name, m in report.metric_deltas.items():
        if m.get("score_a") is None:
            continue
        sig = "[green]YES[/green]" if m["significant"] else "[dim]no[/dim]"
        table.add_row(name, f"{m['score_a']:.3f}", f"{m['score_b']:.3f}",
                      f"{m['delta']:+.4f}", sig)
    console.print(table)

    if report.prompt_diff:
        console.print("\n[bold]Prompt diff:[/bold]")
        console.print(report.prompt_diff)


@app.command(name="init")
def init_project():
    """Create a starter eval_data.json and rageval.md in the current directory."""
    starter = [
        {
            "query": "What is the capital of France?",
            "retrieved_docs": ["France is a country in Western Europe. Its capital city is Paris."],
            "answer": "The capital of France is Paris.",
            "ground_truth": "Paris is the capital of France."
        },
        {
            "query": "What does DNA stand for?",
            "retrieved_docs": ["DNA stands for Deoxyribonucleic Acid. It carries genetic information."],
            "answer": "DNA stands for Deoxyribonucleic Acid.",
            "ground_truth": "DNA stands for Deoxyribonucleic Acid."
        }
    ]
    eval_path = Path("eval_data.json")
    if eval_path.exists():
        console.print("[yellow]eval_data.json already exists, skipping.[/yellow]")
    else:
        eval_path.write_text(json.dumps(starter, indent=2), encoding="utf-8")
        console.print("[green]Created eval_data.json with 2 starter samples.[/green]")

    readme = Path("rageval.md")
    if not readme.exists():
        readme.write_text(
            "# rageval evaluation data\n\n"
            "Add samples to `eval_data.json`. Each sample needs:\n"
            "- `query` (string)\n"
            "- `retrieved_docs` (list of strings)\n"
            "- `answer` (string)\n"
            "- `ground_truth` (string, optional — required for context_recall)\n\n"
            "Run evaluation:\n"
            "```bash\n"
            "rageval run eval_data.json --judge openai --threshold 0.8\n"
            "```\n",
            encoding="utf-8",
        )
        console.print("[green]Created rageval.md[/green]")


if __name__ == "__main__":
    app()