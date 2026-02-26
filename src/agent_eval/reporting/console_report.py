"""Rich console report formatter for evaluation results."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from agent_eval.core.report import EvalReport


class ConsoleReportFormatter:
    """Renders an EvalReport to the terminal using Rich.

    Parameters
    ----------
    console:
        Rich Console instance. Creates a new one if not provided.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def render(self, report: EvalReport) -> None:
        """Print the report to the console."""
        status = "[green]PASS[/green]" if report.all_passed else "[red]FAIL[/red]"
        pass_rate = round(report.pass_rate * 100, 1) if report.total_runs > 0 else 0.0

        # Header panel
        header = (
            f"[bold]{report.suite_name}[/bold]\n"
            f"Agent: {report.agent_name}\n"
            f"Status: {status} | Pass Rate: {pass_rate}%\n"
            f"Cases: {report.total_cases} | Runs: {report.total_runs} | "
            f"Passed: {report.passed_count} | Failed: {report.failed_count}"
        )
        self._console.print(Panel(header, title="Evaluation Report"))

        # Dimension summary table
        if report.dimension_means:
            dim_table = Table(title="Dimension Scores")
            dim_table.add_column("Dimension", style="bold")
            dim_table.add_column("Mean Score", justify="right")
            dim_table.add_column("Bar", min_width=20)

            for dim, score in report.dimension_means.items():
                score_pct = round(score * 100, 1)
                bar_len = int(score * 20)
                bar = "[green]" + "█" * bar_len + "[/green]" + "░" * (20 - bar_len)
                color = "green" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
                dim_table.add_row(
                    dim.value.title(),
                    f"[{color}]{score_pct}%[/{color}]",
                    bar,
                )

            self._console.print(dim_table)

        # Results table (limited for large suites)
        results_table = Table(title="Results")
        results_table.add_column("Case", style="dim")
        results_table.add_column("Run", justify="center")
        results_table.add_column("Score", justify="right")
        results_table.add_column("Status", justify="center")

        display_results = report.results[:50]
        for r in display_results:
            score_pct = round(r.overall_score * 100, 1)
            if r.passed:
                status_str = "[green]PASS[/green]"
            else:
                status_str = "[red]FAIL[/red]"
            results_table.add_row(
                r.case_id,
                str(r.run_index),
                f"{score_pct}%",
                status_str,
            )

        if len(report.results) > 50:
            results_table.add_row(
                f"... and {len(report.results) - 50} more",
                "",
                "",
                "",
            )

        self._console.print(results_table)
