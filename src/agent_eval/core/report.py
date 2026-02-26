"""EvalReport — aggregated results from an evaluation run.

An EvalReport is produced by EvalRunner after it has collected EvalResult
objects for all test cases and runs. It provides summary statistics,
per-dimension pass rates, and serialization to multiple output formats.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_eval.core.evaluator import Dimension, DimensionScore, EvalResult


@dataclass
class DimensionSummary:
    """Aggregated statistics for one evaluation dimension.

    Parameters
    ----------
    dimension:
        Which quality dimension this summary covers.
    mean_score:
        Arithmetic mean of all scores for this dimension.
    min_score:
        Minimum score observed.
    max_score:
        Maximum score observed.
    pass_rate:
        Fraction of cases that passed (score >= threshold). Range [0.0, 1.0].
    total_cases:
        Number of EvalResult objects included in this summary.
    """

    dimension: Dimension
    mean_score: float
    min_score: float
    max_score: float
    pass_rate: float
    total_cases: int
    score_stddev: float = 0.0


@dataclass
class EvalReport:
    """Aggregated report from a full evaluation run.

    Parameters
    ----------
    suite_name:
        Name of the BenchmarkSuite that was evaluated.
    agent_name:
        Name of the AgentUnderTest that was evaluated.
    results:
        All individual EvalResult objects from the run.
    dimension_summaries:
        Per-dimension aggregated statistics.
    created_at:
        UTC timestamp when this report was generated.
    run_config:
        Key-value snapshot of the runner configuration used.
    """

    suite_name: str
    agent_name: str
    results: list[EvalResult] = field(default_factory=list)
    dimension_summaries: list[DimensionSummary] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_config: dict[str, str | int | float | bool] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_results(
        cls,
        results: list[EvalResult],
        suite_name: str = "",
        agent_name: str = "",
        run_config: dict[str, str | int | float | bool] | None = None,
    ) -> "EvalReport":
        """Build an EvalReport by aggregating a list of EvalResult objects.

        Parameters
        ----------
        results:
            All EvalResult objects from the run. May include error results.
        suite_name:
            Name of the benchmark suite.
        agent_name:
            Name of the agent under test.
        run_config:
            Runner configuration snapshot for provenance.

        Returns
        -------
        EvalReport
        """
        dimension_summaries = cls._aggregate_dimensions(results)
        return cls(
            suite_name=suite_name,
            agent_name=agent_name,
            results=results,
            dimension_summaries=dimension_summaries,
            run_config=run_config or {},
        )

    # ------------------------------------------------------------------
    # Summary properties
    # ------------------------------------------------------------------

    @property
    def total_cases(self) -> int:
        """Total number of EvalResult objects in this report."""
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        """Number of EvalResult objects where all dimensions passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_cases(self) -> int:
        """Number of EvalResult objects where at least one dimension failed."""
        return sum(1 for r in self.results if not r.passed and r.error is None)

    @property
    def error_cases(self) -> int:
        """Number of EvalResult objects where the agent raised an error."""
        return sum(1 for r in self.results if r.error is not None)

    @property
    def overall_pass_rate(self) -> float:
        """Fraction of non-error cases that passed all dimensions."""
        non_error = [r for r in self.results if r.error is None]
        if not non_error:
            return 0.0
        return sum(1 for r in non_error if r.passed) / len(non_error)

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency in milliseconds across all results."""
        latencies = [r.latency_ms for r in self.results if r.latency_ms > 0]
        return statistics.mean(latencies) if latencies else 0.0

    def summary_for(self, dimension: Dimension) -> DimensionSummary | None:
        """Return the DimensionSummary for a specific dimension, or None."""
        for ds in self.dimension_summaries:
            if ds.dimension == dimension:
                return ds
        return None

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-paragraph text summary of the evaluation.

        Returns
        -------
        str
            Human-readable summary suitable for CI logs.
        """
        lines = [
            f"Evaluation Report: {self.suite_name} / {self.agent_name}",
            f"Run at: {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Total cases: {self.total_cases}",
            f"Passed: {self.passed_cases}  Failed: {self.failed_cases}  Errors: {self.error_cases}",
            f"Overall pass rate: {self.overall_pass_rate:.1%}",
            f"Mean latency: {self.mean_latency_ms:.1f}ms",
        ]
        if self.dimension_summaries:
            lines.append("")
            lines.append("Dimension scores:")
            for ds in self.dimension_summaries:
                lines.append(
                    f"  {ds.dimension.value:<12} mean={ds.mean_score:.3f}  "
                    f"pass_rate={ds.pass_rate:.1%}  "
                    f"(min={ds.min_score:.3f}, max={ds.max_score:.3f})"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Convert report to a plain dictionary.

        Returns
        -------
        dict[str, object]
            JSON-serializable representation of the report.
        """
        return {
            "suite_name": self.suite_name,
            "agent_name": self.agent_name,
            "created_at": self.created_at.isoformat(),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "error_cases": self.error_cases,
            "overall_pass_rate": round(self.overall_pass_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "run_config": self.run_config,
            "dimension_summaries": [
                {
                    "dimension": ds.dimension.value,
                    "mean_score": round(ds.mean_score, 4),
                    "min_score": round(ds.min_score, 4),
                    "max_score": round(ds.max_score, 4),
                    "score_stddev": round(ds.score_stddev, 4),
                    "pass_rate": round(ds.pass_rate, 4),
                    "total_cases": ds.total_cases,
                }
                for ds in self.dimension_summaries
            ],
            "results": [
                {
                    "case_id": r.case_id,
                    "run_index": r.run_index,
                    "passed": r.passed,
                    "overall_score": round(r.overall_score, 4),
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                    "agent_output": r.agent_output,
                    "dimension_scores": [
                        {
                            "dimension": ds.dimension.value,
                            "score": round(ds.score, 4),
                            "passed": ds.passed,
                            "reason": ds.reason,
                            "raw_value": ds.raw_value,
                        }
                        for ds in r.dimension_scores
                    ],
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string.

        Parameters
        ----------
        indent:
            JSON indentation level.

        Returns
        -------
        str
            Formatted JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render report as a Markdown document.

        Suitable for posting to GitHub PR comments.

        Returns
        -------
        str
            Markdown-formatted report.
        """
        from agent_eval.reporting.markdown_report import MarkdownReporter
        return MarkdownReporter().render(self)

    def to_html(self) -> str:
        """Render report as an HTML document.

        Returns
        -------
        str
            Self-contained HTML page with basic CSS styling.
        """
        from agent_eval.reporting.html_report import HtmlReporter
        return HtmlReporter().render(self)

    # ------------------------------------------------------------------
    # Internal aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_dimensions(results: list[EvalResult]) -> list[DimensionSummary]:
        """Compute per-dimension statistics from a list of EvalResult objects."""
        dimension_scores: dict[Dimension, list[DimensionScore]] = {}

        for result in results:
            for ds in result.dimension_scores:
                dimension_scores.setdefault(ds.dimension, []).append(ds)

        summaries: list[DimensionSummary] = []
        for dimension, scores in sorted(dimension_scores.items(), key=lambda x: x[0].value):
            score_values = [s.score for s in scores]
            pass_count = sum(1 for s in scores if s.passed)
            stddev = statistics.stdev(score_values) if len(score_values) > 1 else 0.0
            summaries.append(
                DimensionSummary(
                    dimension=dimension,
                    mean_score=statistics.mean(score_values),
                    min_score=min(score_values),
                    max_score=max(score_values),
                    pass_rate=pass_count / len(scores),
                    total_cases=len(scores),
                    score_stddev=stddev,
                )
            )

        return summaries
