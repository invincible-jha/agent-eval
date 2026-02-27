"""Statistical report generation for multi-run evaluations.

StatisticalReport wraps a StatisticalResult and provides formatted output
in text, dict, and JSON forms. This is the human-facing surface for
reliability reporting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval.statistical.runner import StatisticalResult


@dataclass
class StatisticalReport:
    """Formatted report summarizing multi-run statistical evaluation results.

    Produced by calling ``StatisticalReport.from_result()``. Provides
    pass@1, pass@3, pass@5, confidence intervals, and variance analysis.

    Parameters
    ----------
    n_runs:
        Total number of evaluation runs executed.
    n_passed:
        Number of runs that produced a passing result.
    n_failed:
        Number of runs that produced a failing result (not errors).
    n_errors:
        Number of runs where the evaluation function raised an exception.
    pass_at_1:
        Estimated probability of passing on a single attempt.
    pass_at_3:
        Estimated probability of passing on at least one of 3 attempts.
    pass_at_5:
        Estimated probability of passing on at least one of 5 attempts.
    ci_lower:
        Lower bound of the 95% Wilson score confidence interval on pass rate.
    ci_upper:
        Upper bound of the 95% Wilson score confidence interval on pass rate.
    mean_score:
        Mean overall score across all non-error runs.
    score_stddev:
        Sample standard deviation of scores across all non-error runs.
    score_variance:
        Sample variance of scores across all non-error runs.
    created_at:
        UTC timestamp when this report was generated.
    label:
        Optional human-readable label for the evaluation (e.g., test case name).
    """

    n_runs: int
    n_passed: int
    n_failed: int
    n_errors: int
    pass_at_1: float
    pass_at_3: float
    pass_at_5: float
    ci_lower: float
    ci_upper: float
    mean_score: float
    score_stddev: float
    score_variance: float
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    label: str = ""

    @classmethod
    def from_result(
        cls,
        result: "StatisticalResult",
        label: str = "",
    ) -> "StatisticalReport":
        """Build a StatisticalReport from a StatisticalResult.

        Parameters
        ----------
        result:
            The raw statistical result from StatisticalRunner.run().
        label:
            Optional label for the evaluation context.

        Returns
        -------
        StatisticalReport
        """
        pass_at_1_val = 0.0
        pass_at_3_val = 0.0
        pass_at_5_val = 0.0

        for pak in result.pass_at_k_values:
            if pak.k == 1:
                pass_at_1_val = pak.value
            elif pak.k == 3:
                pass_at_3_val = pak.value
            elif pak.k == 5:
                pass_at_5_val = pak.value

        ci_lower = 0.0
        ci_upper = 1.0
        if result.ci_95 is not None:
            ci_lower = result.ci_95.lower
            ci_upper = result.ci_95.upper

        return cls(
            n_runs=result.n_runs,
            n_passed=result.n_passed,
            n_failed=result.n_failed,
            n_errors=result.n_errors,
            pass_at_1=pass_at_1_val,
            pass_at_3=pass_at_3_val,
            pass_at_5=pass_at_5_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            mean_score=result.mean_score,
            score_stddev=result.score_std,
            score_variance=result.score_var,
            label=label,
        )

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Convert this report to a plain dictionary.

        Returns
        -------
        dict[str, object]
            JSON-serializable representation of the report.
        """
        return {
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "runs": {
                "total": self.n_runs,
                "passed": self.n_passed,
                "failed": self.n_failed,
                "errors": self.n_errors,
                "pass_rate": round(
                    self.n_passed / self.n_runs if self.n_runs > 0 else 0.0, 4
                ),
            },
            "pass_at_k": {
                "pass_at_1": round(self.pass_at_1, 4),
                "pass_at_3": round(self.pass_at_3, 4),
                "pass_at_5": round(self.pass_at_5, 4),
            },
            "confidence_interval_95": {
                "lower": round(self.ci_lower, 4),
                "upper": round(self.ci_upper, 4),
            },
            "score_statistics": {
                "mean": round(self.mean_score, 4),
                "stddev": round(self.score_stddev, 4),
                "variance": round(self.score_variance, 4),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize this report to a JSON string.

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

    def to_text(self) -> str:
        """Render this report as human-readable plain text.

        Suitable for display in CI logs or terminal output.

        Returns
        -------
        str
            Multi-line plain text summary.
        """
        pass_rate = (
            self.n_passed / self.n_runs if self.n_runs > 0 else 0.0
        )
        lines: list[str] = []

        header = "Statistical Reliability Report"
        if self.label:
            header += f": {self.label}"
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(
            f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        lines.append("")

        lines.append("Run Summary")
        lines.append(
            f"  Total runs : {self.n_runs}"
        )
        lines.append(
            f"  Passed     : {self.n_passed}  ({pass_rate:.1%})"
        )
        lines.append(
            f"  Failed     : {self.n_failed}"
        )
        if self.n_errors:
            lines.append(
                f"  Errors     : {self.n_errors}  (eval_fn raised)"
            )
        lines.append("")

        lines.append("Pass@k Reliability Estimates")
        lines.append(f"  pass@1 : {self.pass_at_1:.4f}  ({self.pass_at_1:.1%})")
        lines.append(f"  pass@3 : {self.pass_at_3:.4f}  ({self.pass_at_3:.1%})")
        lines.append(f"  pass@5 : {self.pass_at_5:.4f}  ({self.pass_at_5:.1%})")
        lines.append("")

        lines.append("95% Confidence Interval (Wilson score, on pass rate)")
        lines.append(
            f"  [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
            f"  (point estimate: {pass_rate:.4f})"
        )
        lines.append("")

        lines.append("Score Statistics")
        lines.append(f"  Mean score : {self.mean_score:.4f}")
        lines.append(f"  Std dev    : {self.score_stddev:.4f}")
        lines.append(f"  Variance   : {self.score_variance:.4f}")

        return "\n".join(lines)
