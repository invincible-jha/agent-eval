"""JSON report formatter for evaluation results."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval.core.report import EvalReport


class JSONReportFormatter:
    """Renders an EvalReport as JSON.

    Parameters
    ----------
    indent:
        JSON indentation level. Defaults to 2.
    """

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def render(self, report: EvalReport) -> str:
        """Render the report as a JSON string.

        Parameters
        ----------
        report:
            The evaluation report to format.

        Returns
        -------
        str
            Pretty-printed JSON string.
        """
        data = {
            "suite_name": report.suite_name,
            "agent_name": report.agent_name,
            "timestamp": report.timestamp,
            "total_cases": report.total_cases,
            "total_runs": report.total_runs,
            "passed": report.passed_count,
            "failed": report.failed_count,
            "pass_rate": round(report.pass_rate, 4) if report.total_runs > 0 else 0.0,
            "dimension_summary": {
                dim.value: {
                    "mean_score": round(score, 4),
                }
                for dim, score in report.dimension_means.items()
            },
            "results": [
                {
                    "case_id": r.case_id,
                    "run_index": r.run_index,
                    "passed": r.passed,
                    "overall_score": round(r.overall_score, 4),
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                    "dimensions": [
                        {
                            "dimension": ds.dimension.value,
                            "score": round(ds.score, 4),
                            "passed": ds.passed,
                            "reason": ds.reason,
                        }
                        for ds in r.dimension_scores
                    ],
                }
                for r in report.results
            ],
        }
        return json.dumps(data, indent=self._indent, default=str)
