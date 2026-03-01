"""Aggregated evaluation session report for pytest-agent-eval.

Collects per-test evaluation results across a full pytest session and
renders them as JSON or Markdown. The :class:`EvalReport` is populated
by the ``pytest_sessionfinish`` hook in :mod:`agent_eval.pytest_plugin.plugin`.

Usage example
-------------
::

    from agent_eval.pytest_plugin.report import EvalReport

    report = EvalReport()
    report.add_result(
        test_name="test_capital_agent",
        passed=True,
        scores={"accuracy": 0.9, "safety": 1.0},
        assertions=[("accuracy", True, "score 0.9 >= 0.8")],
    )
    print(report.to_markdown())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EvalReport:
    """Aggregated evaluation session report.

    Accumulates per-test results via :meth:`add_result` and renders them
    to JSON or Markdown via :meth:`to_json` / :meth:`to_markdown`.

    Parameters
    ----------
    timestamp:
        ISO-8601 UTC timestamp of report creation. Auto-populated on init.
    total_tests:
        Running count of all tests added.
    passed_tests:
        Running count of tests that passed all assertions.
    failed_tests:
        Running count of tests that failed at least one assertion.
    test_results:
        Ordered list of per-test result dicts.
    """

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    test_results: list[dict[str, object]] = field(default_factory=list)

    def add_result(
        self,
        test_name: str,
        passed: bool,
        scores: dict[str, float],
        assertions: list[tuple[str, bool, str]],
    ) -> None:
        """Record the outcome of a single test.

        Parameters
        ----------
        test_name:
            The pytest node ID or human-readable test name.
        passed:
            True when all assertions in the test passed.
        scores:
            Dimension scores produced during the test.
        assertions:
            List of ``(dimension, passed, reason)`` tuples from the test.
        """
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

        self.test_results.append(
            {
                "test_name": test_name,
                "passed": passed,
                "scores": dict(scores),
                "assertions": [(a[0], a[1], a[2]) for a in assertions],
            }
        )

    def to_json(self) -> str:
        """Render the report as a JSON string.

        Returns
        -------
        str
            Pretty-printed JSON representation of the full report.
        """
        return json.dumps(self.__dict__, indent=2, default=str)

    def to_markdown(self) -> str:
        """Render the report as a Markdown table.

        Returns
        -------
        str
            Multi-line Markdown string including a summary header and a
            results table with one row per test.
        """
        lines: list[str] = [
            "# Agent Evaluation Report",
            "",
            f"**Date:** {self.timestamp}",
            (
                f"**Total:** {self.total_tests} | "
                f"**Passed:** {self.passed_tests} | "
                f"**Failed:** {self.failed_tests}"
            ),
            "",
            "| Test | Status | Accuracy | Safety | Cost | Latency |",
            "|------|--------|----------|--------|------|---------|",
        ]

        for result in self.test_results:
            status = "PASS" if result["passed"] else "FAIL"
            scores: dict[str, object] = result.get("scores", {})  # type: ignore[assignment]
            if not isinstance(scores, dict):
                scores = {}

            def _fmt(key: str) -> str:
                val = scores.get(key)
                if isinstance(val, float):
                    return f"{val:.4f}"
                return "-"

            lines.append(
                f"| {result['test_name']} | {status} "
                f"| {_fmt('accuracy')} | {_fmt('safety')} "
                f"| {_fmt('cost')} | {_fmt('latency')} |"
            )

        return "\n".join(lines)
