"""Cascade failure report generation.

CascadeReport wraps a CascadeAnalysis and renders it in human-readable
text, dict, and JSON formats. It emphasizes root causes over cascade noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval.cascade.analyzer import CascadeAnalysis


@dataclass
class CascadeReport:
    """Human-readable report of a cascade failure analysis.

    Produced by calling ``CascadeReport.from_analysis()``.

    Parameters
    ----------
    total_steps:
        Total number of steps evaluated.
    n_passed:
        Number of steps that passed.
    n_root_causes:
        Number of root-cause failures.
    n_cascade:
        Number of cascade failures.
    n_independent:
        Number of independent failures (not in dependency graph).
    cascade_amplification:
        Ratio of total failures to actual (root) failures.
    root_cause_ids:
        Step IDs classified as root causes.
    cascade_ids:
        Step IDs classified as cascade failures.
    independent_ids:
        Step IDs that failed independently.
    cascade_chains:
        List of human-readable cascade chain descriptions.
    created_at:
        UTC timestamp when this report was generated.
    label:
        Optional label for the evaluation context.
    """

    total_steps: int
    n_passed: int
    n_root_causes: int
    n_cascade: int
    n_independent: int
    cascade_amplification: float
    root_cause_ids: list[str] = field(default_factory=list)
    cascade_ids: list[str] = field(default_factory=list)
    independent_ids: list[str] = field(default_factory=list)
    cascade_chains: list[str] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    label: str = ""

    @classmethod
    def from_analysis(
        cls,
        analysis: "CascadeAnalysis",
        label: str = "",
    ) -> "CascadeReport":
        """Build a CascadeReport from a CascadeAnalysis.

        Parameters
        ----------
        analysis:
            The cascade analysis produced by CascadeAnalyzer.
        label:
            Optional label for the evaluation context.

        Returns
        -------
        CascadeReport
        """
        cascade_chains: list[str] = []
        for step_analysis in analysis.step_analyses:
            from agent_eval.cascade.analyzer import FailureClassification

            if (
                step_analysis.classification == FailureClassification.ROOT_CAUSE
                and step_analysis.caused_failures
            ):
                caused = ", ".join(step_analysis.caused_failures)
                cascade_chains.append(
                    f"Step {step_analysis.step_id!r} failed "
                    f"-> caused {len(step_analysis.caused_failures)} cascade(s): "
                    f"{caused}"
                )

        return cls(
            total_steps=analysis.total_steps,
            n_passed=len(analysis.passed_steps),
            n_root_causes=len(analysis.root_causes),
            n_cascade=len(analysis.cascade_failures),
            n_independent=len(analysis.independent_failures),
            cascade_amplification=analysis.cascade_amplification,
            root_cause_ids=list(analysis.root_causes),
            cascade_ids=list(analysis.cascade_failures),
            independent_ids=list(analysis.independent_failures),
            cascade_chains=cascade_chains,
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
            JSON-serializable representation.
        """
        total_failed = self.n_root_causes + self.n_cascade + self.n_independent
        return {
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "summary": {
                "total_steps": self.total_steps,
                "passed": self.n_passed,
                "failed_total": total_failed,
                "root_causes": self.n_root_causes,
                "cascade_failures": self.n_cascade,
                "independent_failures": self.n_independent,
                "actual_reliability": (
                    f"{self.total_steps - total_failed + (total_failed - self.n_cascade)}"
                    f"/{self.total_steps}"
                ),
                "cascade_amplification": round(self.cascade_amplification, 2),
            },
            "root_cause_ids": self.root_cause_ids,
            "cascade_ids": self.cascade_ids,
            "independent_ids": self.independent_ids,
            "cascade_chains": self.cascade_chains,
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

        Root causes are listed first (highest priority). Cascade chains
        show downstream impact. Actual reliability discounts cascade noise.

        Returns
        -------
        str
            Multi-line plain text summary.
        """
        total_failed = self.n_root_causes + self.n_cascade + self.n_independent
        actual_failures = self.n_root_causes + self.n_independent

        lines: list[str] = []

        header = "Cascade Failure Analysis"
        if self.label:
            header += f": {self.label}"
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(
            f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        lines.append("")

        lines.append("Run Summary")
        lines.append(f"  Total steps   : {self.total_steps}")
        lines.append(f"  Passed        : {self.n_passed}")
        lines.append(f"  Failed (total): {total_failed}")
        lines.append("")

        lines.append("Failure Classification")
        lines.append(f"  Root causes   : {self.n_root_causes}  {self.root_cause_ids}")
        lines.append(f"  Cascades      : {self.n_cascade}  {self.cascade_ids}")
        lines.append(f"  Independent   : {self.n_independent}  {self.independent_ids}")
        lines.append("")

        if actual_failures < total_failed:
            lines.append(
                f"Actual Reliability: {actual_failures} independent failure(s) "
                f"caused {self.n_cascade} cascade failure(s)"
            )
            lines.append(
                f"  Cascade amplification: {self.cascade_amplification:.1f}x "
                f"(1 root failure -> {self.cascade_amplification:.1f} reported failures on average)"
            )
        else:
            lines.append(
                f"Actual Reliability: {actual_failures} independent failure(s), "
                f"no cascade amplification"
            )
        lines.append("")

        if self.cascade_chains:
            lines.append("Cascade Chains")
            for chain in self.cascade_chains:
                lines.append(f"  {chain}")
        elif self.n_root_causes > 0:
            lines.append("Cascade Chains")
            lines.append("  (no downstream cascades from root causes)")

        return "\n".join(lines)
