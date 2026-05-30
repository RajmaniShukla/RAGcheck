"""
ragcheck CLI — entry point for `ragcheck eval` and related commands.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

app = typer.Typer(
    name="ragcheck",
    help="🔍 RAGcheck — Measure, debug, and improve your RAG pipeline quality.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


@app.command("eval")
def eval_cmd(
    input: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Input file path (.json or .csv). See docs for schema.",
    ),
    metrics: str = typer.Option(
        "context_relevance,faithfulness,answer_relevance",
        "--metrics", "-m",
        help='Comma-separated list of metrics, or "all". '
             "Options: context_relevance, faithfulness, answer_relevance, "
             "context_recall, noise_sensitivity, chunk_utilization",
    ),
    judge: str = typer.Option(
        "gpt-4o-mini",
        "--judge", "-j",
        help="Judge model. Examples: gpt-4o-mini, anthropic/claude-3-haiku-20240307, ollama/llama3",
    ),
    provider: str = typer.Option(
        "litellm",
        "--provider", "-p",
        help="Judge provider: litellm | openai | anthropic | local",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key (falls back to env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)",
        envvar="RAGCHECK_API_KEY",
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        help="Custom API base URL (for Ollama / vLLM local deployments).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (.html or .json). Defaults to terminal output only.",
    ),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Max concurrent judge calls."),
    fail_threshold: Optional[float] = typer.Option(
        None,
        "--fail-threshold",
        help="Exit code 1 if overall score < threshold (for CI use).",
        min=0.0,
        max=1.0,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-sample breakdown."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset / experiment name."),
) -> None:
    """
    [bold cyan]Evaluate a RAG pipeline from a JSON or CSV file.[/bold cyan]

    [dim]Example:[/dim]
        ragcheck eval --input data.json --metrics all --judge gpt-4o --output report.html
    """
    from ragcheck import evaluate_dataset, _resolve_metrics
    from ragcheck.connectors.custom import load as load_dataset
    from ragcheck.core.schema import EvalConfig, JudgeConfig, JudgeProvider
    from ragcheck.reporters.terminal import print_report
    from ragcheck.reporters.json_export import save_json
    from ragcheck.reporters.html_report import save_html

    if not input.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input}")
        raise typer.Exit(1)

    # Parse metrics
    metric_list = ["all"] if metrics.strip().lower() == "all" else [m.strip() for m in metrics.split(",")]

    try:
        resolved_metrics = _resolve_metrics(metric_list)
    except ValueError as exc:
        console.print(f"[red]Invalid metric:[/red] {exc}")
        raise typer.Exit(1)

    # Load dataset
    try:
        dataset = load_dataset(input, name=name)
    except Exception as exc:
        console.print(f"[red]Failed to load input:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"\n[dim]Loaded [bold white]{len(dataset.samples)}[/bold white] samples from "
        f"[bold white]{input.name}[/bold white][/dim]"
    )
    console.print(
        f"[dim]Metrics: [bold white]{', '.join(m.value for m in resolved_metrics)}[/bold white][/dim]"
    )
    console.print(
        f"[dim]Judge: [bold white]{judge}[/bold white] (provider: {provider})[/dim]\n"
    )

    config = EvalConfig(
        metrics=resolved_metrics,
        judge=JudgeConfig(
            provider=JudgeProvider(provider),
            model=judge,
            api_key=api_key,
            api_base=api_base,
        ),
        concurrency=concurrency,
        fail_threshold=fail_threshold,
    )

    # Run with progress bar
    completed_count = [0]
    total = len(dataset.samples)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Evaluating...", total=total)

        def on_progress(done: int, _total: int) -> None:
            progress.update(task, completed=done)

        from ragcheck.core.pipeline import Pipeline
        pipeline = Pipeline(config, progress_callback=on_progress)

        try:
            report = asyncio.run(pipeline.run(dataset))
        except Exception as exc:
            console.print(f"\n[red]Evaluation failed:[/red] {exc}")
            raise typer.Exit(1)

    # Print report
    print_report(report, verbose=verbose)

    # Export
    if output:
        ext = output.suffix.lower()
        try:
            if ext == ".html":
                save_html(report, output)
                console.print(f"[green]HTML report saved:[/green] {output}")
            elif ext == ".json":
                save_json(report, output)
                console.print(f"[green]JSON report saved:[/green] {output}")
            else:
                console.print(f"[yellow]Warning:[/yellow] Unknown output format '{ext}', skipping export.")
        except Exception as exc:
            console.print(f"[red]Failed to save output:[/red] {exc}")

    # CI exit code
    if report.passed is False:
        console.print(
            f"\n[red bold]FAILED:[/red bold] Overall score {report.overall_score:.3f} "
            f"< threshold {fail_threshold:.2f}"
        )
        raise typer.Exit(1)


@app.command("version")
def version_cmd() -> None:
    """Show ragcheck version."""
    from ragcheck import __version__
    console.print(f"ragcheck [bold cyan]{__version__}[/bold cyan]")


@app.command("metrics")
def metrics_cmd() -> None:
    """List all available evaluation metrics."""
    from rich.table import Table
    from rich import box

    table = Table(box=box.ROUNDED, title="[bold]Available Metrics[/bold]", padding=(0, 1))
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Needs ground_truth?", justify="center")

    rows = [
        ("context_relevance", "Are retrieved chunks relevant to the question?", "No"),
        ("faithfulness", "Is the answer grounded in context? (no hallucination)", "No"),
        ("answer_relevance", "Does the answer address the question?", "No"),
        ("context_recall", "Did retrieval cover all facts needed to answer?", "[bold yellow]Yes[/bold yellow]"),
        ("noise_sensitivity", "How much does quality degrade with irrelevant chunks?", "No"),
        ("chunk_utilization", "Which retrieved chunks were actually used by the LLM?", "No"),
    ]
    for name, desc, gt in rows:
        table.add_row(name, desc, gt)

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
