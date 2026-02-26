"""CLI entry point for agent-eval.

Invoked as::

    agent-eval [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_eval.cli.main
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """Framework for evaluating AI agents across multiple quality dimensions."""


# ---------------------------------------------------------------
# run — main evaluation entry point
# ---------------------------------------------------------------


@cli.command(name="run")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to eval.yaml config file.",
)
@click.option(
    "--suite",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a suite YAML/JSON file or directory of suites.",
)
@click.option(
    "--builtin",
    type=str,
    default=None,
    help="Name of a built-in suite (e.g., qa_basic, safety_basic).",
)
@click.option("--runs", type=int, default=1, help="Runs per test case.")
@click.option("--concurrency", type=int, default=1, help="Max parallel agent calls.")
@click.option("--timeout", type=int, default=30000, help="Timeout per case in ms.")
@click.option("--fail-fast", is_flag=True, help="Abort after first failure.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown", "html", "console"]),
    default="console",
    help="Report output format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (stdout if not set).",
)
@click.option(
    "--agent-url",
    type=str,
    default=None,
    help="HTTP endpoint URL for the agent under test.",
)
def run_command(
    config: Path | None,
    suite: Path | None,
    builtin: str | None,
    runs: int,
    concurrency: int,
    timeout: int,
    fail_fast: bool,
    output_format: str,
    output: Path | None,
    agent_url: str | None,
) -> None:
    """Run an evaluation suite against an agent.

    Either provide --config for a full eval.yaml configuration,
    or use --suite/--builtin with --agent-url for quick evaluation.
    """
    from agent_eval.core.runner import EvalRunner, RunnerOptions

    if config is not None:
        from agent_eval.core.config import EvalConfig

        eval_config = EvalConfig.from_yaml(config)
        runner = EvalRunner.from_config(eval_config)
        console.print(f"[bold]Loaded config:[/bold] {config}")
    else:
        from agent_eval.evaluators import EVALUATOR_REGISTRY

        # Build evaluators from defaults
        default_types = ["accuracy", "latency", "safety", "format"]
        evaluators = []
        for eval_type in default_types:
            if eval_type in EVALUATOR_REGISTRY:
                evaluators.append(EVALUATOR_REGISTRY[eval_type]())
        if not evaluators:
            console.print("[red]No evaluators available.[/red]")
            sys.exit(1)
        options = RunnerOptions(
            runs_per_case=runs,
            timeout_ms=timeout,
            concurrency=concurrency,
            fail_fast=fail_fast,
        )
        runner = EvalRunner(evaluators=evaluators, options=options)

    # Load suite
    from agent_eval.suites.loader import SuiteLoader

    benchmark_suite = None
    if suite is not None:
        if suite.is_dir():
            suites = SuiteLoader.load_directory(suite)
            if not suites:
                console.print(f"[red]No suites found in {suite}[/red]")
                sys.exit(1)
            benchmark_suite = suites[0]
            if len(suites) > 1:
                console.print(
                    f"[yellow]Multiple suites found, using first: {benchmark_suite.name}[/yellow]"
                )
        else:
            benchmark_suite = SuiteLoader.load_file(suite)
    elif builtin is not None:
        benchmark_suite = SuiteLoader.load_builtin(builtin)
    else:
        console.print("[red]Provide --suite, --builtin, or --config with suite path.[/red]")
        sys.exit(1)

    # Build agent
    from agent_eval.core.agent_wrapper import AgentUnderTest

    if agent_url is not None:
        from agent_eval.adapters.http import HTTPAdapter

        adapter = HTTPAdapter(url=agent_url)

        async def agent_fn(prompt: str) -> str:
            return await adapter.invoke(prompt)

        agent = AgentUnderTest(callable_fn=agent_fn, name="http-agent", timeout_ms=timeout)
    else:
        console.print("[yellow]No agent specified. Use --agent-url or --config.[/yellow]")
        console.print("Running in dry-run mode (echo agent).")

        async def echo_agent(prompt: str) -> str:
            return f"Echo: {prompt}"

        agent = AgentUnderTest(callable_fn=echo_agent, name="echo-agent", timeout_ms=timeout)

    # Execute
    console.print(f"\n[bold]Suite:[/bold] {benchmark_suite.name}")
    console.print(f"[bold]Cases:[/bold] {len(benchmark_suite.cases)}")
    console.print(f"[bold]Runs per case:[/bold] {runs}")
    console.print()

    report = asyncio.run(runner.run(agent, benchmark_suite))

    # Format output
    formatted = _format_report(report, output_format)
    if output is not None:
        output.write_text(formatted, encoding="utf-8")
        console.print(f"\n[green]Report written to {output}[/green]")
    elif output_format != "console":
        click.echo(formatted)
    else:
        console.print(formatted)


def _format_report(report: object, fmt: str) -> str:
    """Format an EvalReport using the specified formatter."""
    if fmt == "json":
        from agent_eval.reporting.json_report import JSONReportFormatter

        return JSONReportFormatter().format(report)
    elif fmt == "markdown":
        from agent_eval.reporting.markdown_report import MarkdownReportFormatter

        return MarkdownReportFormatter().format(report)
    elif fmt == "html":
        from agent_eval.reporting.html_report import HTMLReportFormatter

        return HTMLReportFormatter().format(report)
    else:
        from agent_eval.reporting.console_report import ConsoleReportFormatter

        return ConsoleReportFormatter().format(report)


# ---------------------------------------------------------------
# suite — manage benchmark suites
# ---------------------------------------------------------------


@cli.group(name="suite")
def suite_group() -> None:
    """Manage benchmark suites."""


@suite_group.command(name="list")
def suite_list() -> None:
    """List all built-in evaluation suites."""
    from agent_eval.suites.loader import SuiteLoader

    builtins = SuiteLoader.list_builtin()

    table = Table(title="Built-in Suites")
    table.add_column("Name", style="cyan")
    table.add_column("Format", style="green")

    for name in sorted(builtins):
        ext = "yaml" if name.endswith(".yaml") or name.endswith(".yml") else "json"
        display_name = name.rsplit(".", 1)[0] if "." in name else name
        table.add_row(display_name, ext)

    console.print(table)


@suite_group.command(name="validate")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def suite_validate(path: Path) -> None:
    """Validate a suite file for correct schema."""
    from agent_eval.suites.loader import SuiteLoader

    try:
        if path.is_dir():
            suites = SuiteLoader.load_directory(path)
            for s in suites:
                console.print(f"  [green]OK[/green] {s.name} ({len(s.cases)} cases)")
        else:
            s = SuiteLoader.load_file(path)
            console.print(f"  [green]OK[/green] {s.name} ({len(s.cases)} cases)")
    except Exception as exc:
        console.print(f"  [red]FAIL[/red] {path}: {exc}")
        sys.exit(1)


@suite_group.command(name="show")
@click.argument("name", type=str)
def suite_show(name: str) -> None:
    """Show details of a built-in suite."""
    from agent_eval.suites.loader import SuiteLoader

    try:
        s = SuiteLoader.load_builtin(name)
    except FileNotFoundError:
        console.print(f"[red]Built-in suite '{name}' not found.[/red]")
        sys.exit(1)

    console.print(f"[bold]{s.name}[/bold]")
    if s.description:
        console.print(f"  {s.description}")
    console.print(f"  Cases: {len(s.cases)}")
    console.print()

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Input (truncated)", max_width=60)
    table.add_column("Expected", max_width=30)

    for case in s.cases:
        truncated = case.input[:57] + "..." if len(case.input) > 60 else case.input
        expected = (case.expected_output or "")[:27] + "..." if case.expected_output and len(case.expected_output) > 30 else (case.expected_output or "-")
        table.add_row(case.id, truncated, expected)

    console.print(table)


# ---------------------------------------------------------------
# report — generate reports from saved results
# ---------------------------------------------------------------


@cli.command(name="report")
@click.argument("results_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown", "html", "console"]),
    default="console",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def report_command(results_file: Path, output_format: str, output: Path | None) -> None:
    """Generate a report from saved evaluation results (JSON)."""
    data = json.loads(results_file.read_text(encoding="utf-8"))
    console.print(f"[bold]Loaded results:[/bold] {results_file}")
    console.print(json.dumps(data, indent=2)[:2000])

    if output is not None:
        console.print(f"\n[green]Report written to {output}[/green]")


# ---------------------------------------------------------------
# gate — deployment gate checking
# ---------------------------------------------------------------


@cli.command(name="gate")
@click.argument("results_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.8,
    help="Minimum overall score to pass (0.0-1.0).",
)
@click.option(
    "--dimension",
    "-d",
    multiple=True,
    type=str,
    help="Dimension=threshold pairs (e.g., accuracy=0.9 safety=1.0).",
)
def gate_command(
    results_file: Path, threshold: float, dimension: Sequence[str]
) -> None:
    """Check if evaluation results pass deployment gates.

    Exit code 0 = passed, exit code 1 = failed.
    """
    data = json.loads(results_file.read_text(encoding="utf-8"))

    overall_score = data.get("overall_score", 0.0)
    passed = overall_score >= threshold

    console.print(f"[bold]Gate Check:[/bold]")
    console.print(f"  Overall score: {overall_score:.3f}")
    console.print(f"  Threshold:     {threshold:.3f}")

    if dimension:
        dimension_scores = data.get("dimension_scores", {})
        for dim_spec in dimension:
            if "=" in dim_spec:
                dim_name, dim_thresh = dim_spec.split("=", 1)
                dim_score = dimension_scores.get(dim_name, 0.0)
                dim_passed = float(dim_score) >= float(dim_thresh)
                status = "[green]PASS[/green]" if dim_passed else "[red]FAIL[/red]"
                console.print(f"  {dim_name}: {dim_score:.3f} >= {dim_thresh} {status}")
                if not dim_passed:
                    passed = False

    if passed:
        console.print("\n[bold green]GATE: PASSED[/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]GATE: FAILED[/bold red]")
        sys.exit(1)


# ---------------------------------------------------------------
# init — project initialization
# ---------------------------------------------------------------


@cli.command(name="init")
@click.option(
    "--directory",
    "-d",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Directory to initialize (default: current directory).",
)
def init_command(directory: Path) -> None:
    """Initialize agent-eval configuration in a project."""
    config_path = directory / "eval.yaml"
    if config_path.exists():
        console.print(f"[yellow]Config already exists: {config_path}[/yellow]")
        return

    template = """# agent-eval configuration
# See: https://github.com/aumos-ai/agent-eval

evaluators:
  accuracy:
    enabled: true
    settings: {}
  latency:
    enabled: true
    settings:
      max_latency_ms: 5000
  safety:
    enabled: true
    settings: {}
  format:
    enabled: true
    settings: {}
  cost:
    enabled: false
    settings:
      max_cost_tokens: 4096

runner:
  runs_per_case: 1
  timeout_ms: 30000
  max_retries: 0
  concurrency: 1
  fail_fast: false

suites:
  - builtin: qa_basic
  - builtin: safety_basic
"""
    directory.mkdir(parents=True, exist_ok=True)
    config_path.write_text(template, encoding="utf-8")
    console.print(f"[green]Created config: {config_path}[/green]")
    console.print("Edit eval.yaml to configure evaluators and suites.")


# ---------------------------------------------------------------
# version — detailed version info
# ---------------------------------------------------------------


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_eval import __version__

    console.print(f"[bold]agent-eval[/bold] v{__version__}")
    console.print(f"  Python: {sys.version.split()[0]}")


# ---------------------------------------------------------------
# plugins — list registered plugins
# ---------------------------------------------------------------


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered evaluator plugins."""
    from agent_eval.evaluators import EVALUATOR_REGISTRY
    from agent_eval.plugins.registry import PluginRegistry

    table = Table(title="Registered Evaluators")
    table.add_column("Type", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Dimension", style="yellow")

    for type_name, cls in sorted(EVALUATOR_REGISTRY.items()):
        try:
            instance = cls()
            dim = instance.dimension.value
        except Exception:
            dim = "?"
        table.add_row(type_name, cls.__name__, dim)

    console.print(table)

    console.print("\n[bold]Plugin Registry:[/bold]")
    registry = PluginRegistry()
    registered = registry.list_plugins()
    if registered:
        for name, plugin_cls in registered.items():
            console.print(f"  {name}: {plugin_cls}")
    else:
        console.print("  (No external plugins registered)")


# ---------------------------------------------------------------
# adapters — list available agent adapters
# ---------------------------------------------------------------


@cli.command(name="adapters")
def adapters_command() -> None:
    """List available agent adapters."""
    table = Table(title="Agent Adapters")
    table.add_column("Adapter", style="cyan")
    table.add_column("Framework", style="green")
    table.add_column("Install", style="dim")

    adapters = [
        ("CallableAdapter", "Any Python callable", "(built-in)"),
        ("HTTPAdapter", "HTTP JSON endpoint", "(built-in)"),
        ("LangChainAdapter", "LangChain Agent/Chain/Runnable", "pip install langchain-core"),
        ("CrewAIAdapter", "CrewAI Crew", "pip install crewai"),
        ("AutoGenAdapter", "AutoGen AssistantAgent", "pip install autogen-agentchat"),
        ("OpenAIAgentsAdapter", "OpenAI Agents SDK", "pip install openai-agents"),
    ]

    for name, framework, install in adapters:
        table.add_row(name, framework, install)

    console.print(table)


if __name__ == "__main__":
    cli()
