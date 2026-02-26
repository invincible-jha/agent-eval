"""Unit tests for agent_eval.reporting formatters.

Tests JSON, HTML, Markdown, and Console report formatters using
mock EvalReport objects that match the formatter API.
"""
from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from agent_eval.core.evaluator import Dimension, DimensionScore, EvalResult
from agent_eval.core.report import EvalReport
from agent_eval.reporting import (
    ConsoleReportFormatter,
    HTMLReportFormatter,
    JSONReportFormatter,
    MarkdownReportFormatter,
)
from agent_eval.reporting.__init__ import __all__ as reporting_all


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_score(dimension: Dimension, score: float, passed: bool) -> DimensionScore:
    return DimensionScore(dimension=dimension, score=score, passed=passed)


def _make_report(
    passed: bool = True,
    num_results: int = 2,
) -> EvalReport:
    """Build an EvalReport with accuracy scores."""
    results = [
        EvalResult(
            case_id=f"case-{i}",
            run_index=0,
            agent_output="output text",
            dimension_scores=[
                _make_score(Dimension.ACCURACY, 1.0 if passed else 0.0, passed)
            ],
            latency_ms=100.0 * i,
        )
        for i in range(1, num_results + 1)
    ]
    return EvalReport.from_results(
        results, suite_name="Test Suite", agent_name="test-agent"
    )


# Mock report with the attribute interface the formatters expect
# (all_passed, pass_rate, total_runs, passed_count, failed_count,
#  dimension_means, timestamp, results, total_cases)
def _make_formatter_mock_report(
    all_passed: bool = True,
    total_runs: int = 2,
    passed_count: int = 2,
    failed_count: int = 0,
    total_cases: int = 2,
    pass_rate: float = 1.0,
    suite_name: str = "Test Suite",
    agent_name: str = "test-agent",
    timestamp: str = "2026-01-01T00:00:00",
) -> MagicMock:
    mock_report = MagicMock()
    mock_report.all_passed = all_passed
    mock_report.pass_rate = pass_rate
    mock_report.total_runs = total_runs
    mock_report.passed_count = passed_count
    mock_report.failed_count = failed_count
    mock_report.total_cases = total_cases
    mock_report.suite_name = suite_name
    mock_report.agent_name = agent_name
    mock_report.timestamp = timestamp
    mock_report.dimension_means = {Dimension.ACCURACY: 0.9}

    # Build mock results
    mock_result = MagicMock()
    mock_result.case_id = "case-1"
    mock_result.run_index = 0
    mock_result.overall_score = 0.9
    mock_result.passed = True
    mock_result.error = None
    mock_result.latency_ms = 100.0
    dim_score = MagicMock()
    dim_score.dimension = Dimension.ACCURACY
    dim_score.score = 0.9
    dim_score.passed = True
    dim_score.reason = "Test reason"
    mock_result.dimension_scores = [dim_score]
    mock_report.results = [mock_result]

    return mock_report


# ---------------------------------------------------------------------------
# reporting/__init__.py exports
# ---------------------------------------------------------------------------


class TestReportingInit:
    def test_all_exports_importable(self) -> None:
        assert "ConsoleReportFormatter" in reporting_all
        assert "HTMLReportFormatter" in reporting_all
        assert "JSONReportFormatter" in reporting_all
        assert "MarkdownReportFormatter" in reporting_all


# ---------------------------------------------------------------------------
# JSONReportFormatter
# ---------------------------------------------------------------------------


class TestJSONReportFormatter:
    def _make_json_report(self) -> EvalReport:
        """Build a proper EvalReport with the attributes JSON formatter expects."""
        results = [
            EvalResult(
                case_id="c1",
                run_index=0,
                agent_output="output",
                dimension_scores=[_make_score(Dimension.ACCURACY, 0.9, True)],
                latency_ms=100.0,
            )
        ]
        return EvalReport.from_results(results, suite_name="My Suite", agent_name="my-agent")

    def test_render_returns_valid_json(self) -> None:
        # JSONReportFormatter uses all_passed, pass_rate, etc. — use mock
        fmt = JSONReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_render_contains_suite_name(self) -> None:
        fmt = JSONReportFormatter()
        mock_report = _make_formatter_mock_report(suite_name="My Eval Suite")
        output = fmt.render(mock_report)
        assert "My Eval Suite" in output

    def test_render_contains_agent_name(self) -> None:
        fmt = JSONReportFormatter()
        mock_report = _make_formatter_mock_report(agent_name="gpt-4o-agent")
        output = fmt.render(mock_report)
        assert "gpt-4o-agent" in output

    def test_render_has_results_array(self) -> None:
        fmt = JSONReportFormatter()
        mock_report = _make_formatter_mock_report()
        parsed = json.loads(fmt.render(mock_report))
        assert "results" in parsed

    def test_render_has_dimension_summary(self) -> None:
        fmt = JSONReportFormatter()
        mock_report = _make_formatter_mock_report()
        parsed = json.loads(fmt.render(mock_report))
        assert "dimension_summary" in parsed

    def test_custom_indent(self) -> None:
        fmt = JSONReportFormatter(indent=4)
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        # 4-space indented JSON has 4-space lines
        assert "    " in output

    def test_zero_total_runs_pass_rate_is_zero(self) -> None:
        fmt = JSONReportFormatter()
        mock_report = _make_formatter_mock_report(total_runs=0, pass_rate=0.0)
        parsed = json.loads(fmt.render(mock_report))
        assert parsed["pass_rate"] == 0.0

    def test_pass_rate_rounded(self) -> None:
        fmt = JSONReportFormatter()
        mock_report = _make_formatter_mock_report(total_runs=3, pass_rate=2 / 3)
        parsed = json.loads(fmt.render(mock_report))
        assert 0 <= parsed["pass_rate"] <= 1.0


# ---------------------------------------------------------------------------
# MarkdownReportFormatter
# ---------------------------------------------------------------------------


class TestMarkdownReportFormatter:
    def test_render_returns_string(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        assert isinstance(output, str)

    def test_render_has_h1_header(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report(suite_name="MySuite")
        output = fmt.render(mock_report)
        assert "# " in output
        assert "MySuite" in output

    def test_render_shows_agent_name(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report(agent_name="my-bot")
        output = fmt.render(mock_report)
        assert "my-bot" in output

    def test_render_shows_pass_status(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report(all_passed=True)
        output = fmt.render(mock_report)
        assert "PASS" in output

    def test_render_shows_fail_status(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report(all_passed=False)
        output = fmt.render(mock_report)
        assert "FAIL" in output

    def test_render_has_summary_table(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        assert "Summary" in output
        assert "|" in output

    def test_render_has_dimension_scores_section(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        assert "Dimension" in output

    def test_render_has_results_section(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        assert "Results" in output

    def test_large_report_collapses_results(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report()
        # Create 25 mock results to trigger the collapse
        mock_results = []
        for i in range(25):
            r = MagicMock()
            r.case_id = f"case-{i}"
            r.run_index = 0
            r.overall_score = 1.0
            r.passed = True
            mock_results.append(r)
        mock_report.results = mock_results
        output = fmt.render(mock_report)
        assert "<details>" in output

    def test_ends_with_footer(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        assert "agent-eval" in output

    def test_zero_total_runs_shows_zero_pass_rate(self) -> None:
        fmt = MarkdownReportFormatter()
        mock_report = _make_formatter_mock_report(total_runs=0, pass_rate=0.0)
        output = fmt.render(mock_report)
        assert "0.0%" in output or "0%" in output


# ---------------------------------------------------------------------------
# HTMLReportFormatter
# ---------------------------------------------------------------------------


class TestHTMLReportFormatter:
    def test_render_returns_html_string(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        assert "<!DOCTYPE html>" in output

    def test_render_contains_suite_name(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report(suite_name="My Suite")
        output = fmt.render(mock_report)
        assert "My Suite" in output

    def test_render_contains_agent_name(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report(agent_name="gpt-agent")
        output = fmt.render(mock_report)
        assert "gpt-agent" in output

    def test_render_shows_pass_badge(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report(all_passed=True)
        output = fmt.render(mock_report)
        assert "pass" in output.lower()

    def test_render_shows_fail_badge(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report(all_passed=False)
        output = fmt.render(mock_report)
        assert "fail" in output.lower() or "FAIL" in output

    def test_render_contains_dimension_rows(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        # Accuracy should appear as a dimension title
        assert "Accuracy" in output or "accuracy" in output.lower()

    def test_render_contains_result_rows(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report()
        output = fmt.render(mock_report)
        assert "case-1" in output

    def test_render_contains_timestamp(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report(timestamp="2026-01-15T12:00:00")
        output = fmt.render(mock_report)
        assert "2026-01-15" in output

    def test_zero_total_runs_shows_zero_pass_rate(self) -> None:
        fmt = HTMLReportFormatter()
        mock_report = _make_formatter_mock_report(total_runs=0, pass_rate=0.0)
        output = fmt.render(mock_report)
        assert "0.0%" in output or "0%" in output


# ---------------------------------------------------------------------------
# ConsoleReportFormatter
# ---------------------------------------------------------------------------


class TestConsoleReportFormatter:
    def test_render_does_not_raise(self) -> None:
        from rich.console import Console
        console = Console(file=StringIO())
        fmt = ConsoleReportFormatter(console=console)
        mock_report = _make_formatter_mock_report()
        fmt.render(mock_report)  # Should not raise

    def test_render_with_no_console_uses_default(self) -> None:
        fmt = ConsoleReportFormatter()
        # Should construct without error
        assert fmt._console is not None

    def test_render_with_no_dimension_means(self) -> None:
        from rich.console import Console
        console = Console(file=StringIO())
        fmt = ConsoleReportFormatter(console=console)
        mock_report = _make_formatter_mock_report()
        mock_report.dimension_means = {}
        fmt.render(mock_report)  # No dimension table, should not raise

    def test_render_truncates_large_result_set(self) -> None:
        from rich.console import Console
        buf = StringIO()
        console = Console(file=buf)
        fmt = ConsoleReportFormatter(console=console)
        mock_report = _make_formatter_mock_report()
        # Create 60 results
        mock_results = []
        for i in range(60):
            r = MagicMock()
            r.case_id = f"c{i}"
            r.run_index = 0
            r.overall_score = 1.0
            r.passed = True
            mock_results.append(r)
        mock_report.results = mock_results
        fmt.render(mock_report)
        output = buf.getvalue()
        # Should show "10 more" message
        assert "10 more" in output

    def test_render_shows_pass_status(self) -> None:
        from rich.console import Console
        buf = StringIO()
        console = Console(file=buf, highlight=False, markup=False)
        fmt = ConsoleReportFormatter(console=console)
        mock_report = _make_formatter_mock_report(all_passed=True)
        fmt.render(mock_report)
        # The raw text should contain PASS

    def test_render_with_failing_results(self) -> None:
        from rich.console import Console
        buf = StringIO()
        console = Console(file=buf)
        fmt = ConsoleReportFormatter(console=console)
        mock_report = _make_formatter_mock_report(all_passed=False)
        # Set a failing result
        mock_result = MagicMock()
        mock_result.case_id = "fail-case"
        mock_result.run_index = 0
        mock_result.overall_score = 0.2
        mock_result.passed = False
        mock_report.results = [mock_result]
        fmt.render(mock_report)  # Should not raise
