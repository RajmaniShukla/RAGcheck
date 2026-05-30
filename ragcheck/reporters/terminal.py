"""
Terminal reporter — beautiful Rich CLI output.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from ragcheck.core.schema import EvalReport, MetricName

console = Console()


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    elif score >= 0.4:
        return "orange3"
    else:
        return "red"


def _score_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def print_report(report: EvalReport, verbose: bool = False) -> None:
    """Print a full evaluation report to the terminal."""
    console.print()

    # Header
    title = f"[bold cyan]RAGcheck Evaluation Report[/bold cyan]"
    if report.dataset_name:
        title += f"  [dim]·[/dim]  [white]{report.dataset_name}[/white]"
    console.print(Panel(title, box=box.ROUNDED, padding=(0, 2)))
    console.print()

    # Summary stats
    summary_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value", style="bold white")
    summary_table.add_row("Samples evaluated", str(len(report.results)))
    summary_table.add_row("Metrics", str(len(report.aggregate_stats)))
    summary_table.add_row("Judge model", report.config.judge.model)

    score_text = Text(f"{report.overall_score:.3f}", style=_score_color(report.overall_score))
    score_text.append(f"  {_score_bar(report.overall_score)}", style=_score_color(report.overall_score))
    summary_table.add_row("Overall score", score_text)

    if report.passed is not None:
        threshold = report.config.fail_threshold
        status = "[green]✅ PASSED[/green]" if report.passed else "[red]❌ FAILED[/red]"
        summary_table.add_row("Threshold check", f"{status}  [dim](threshold: {threshold})[/dim]")

    console.print(summary_table)
    console.print()

    # Per-metric aggregate table
    metric_table = Table(
        title="[bold]Per-Metric Results[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
    )
    metric_table.add_column("Metric", style="bold", min_width=22)
    metric_table.add_column("Mean", justify="right", min_width=7)
    metric_table.add_column("Min", justify="right", min_width=7)
    metric_table.add_column("Max", justify="right", min_width=7)
    metric_table.add_column("Std", justify="right", min_width=7)
    metric_table.add_column("Score Bar", min_width=22)
    metric_table.add_column("Errors", justify="right", min_width=6)

    for stat in report.aggregate_stats:
        color = _score_color(stat.mean)
        metric_table.add_row(
            stat.metric.value,
            f"[{color}]{stat.mean:.3f}[/{color}]",
            f"{stat.min:.3f}",
            f"{stat.max:.3f}",
            f"{stat.std:.3f}",
            f"[{color}]{_score_bar(stat.mean)}[/{color}]",
            str(stat.failed_samples) if stat.failed_samples else "[dim]0[/dim]",
        )

    console.print(metric_table)

    # Per-sample breakdown (verbose)
    if verbose:
        console.print()
        console.print("[bold]Per-Sample Breakdown[/bold]")
        console.print()
        for i, result in enumerate(report.results):
            q_short = result.sample.question[:80] + ("…" if len(result.sample.question) > 80 else "")
            console.print(f"[dim]Sample {i + 1}:[/dim] [white]{q_short}[/white]")
            for score in result.scores:
                color = _score_color(score.score)
                bar = _score_bar(score.score, width=15)
                error_hint = f"  [red dim]({score.error})[/red dim]" if score.error else ""
                console.print(
                    f"  [dim]{score.metric.value:<22}[/dim] "
                    f"[{color}]{score.score:.3f}  {bar}[/{color}]{error_hint}"
                )
            console.print()

    console.print()
