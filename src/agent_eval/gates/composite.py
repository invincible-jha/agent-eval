"""Composite deployment gate for combining multiple gates.

CompositeGate wraps multiple sub-gates with AND/OR logic, enabling
tiered deployment criteria (e.g., staging vs. production thresholds).
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from agent_eval.core.gate import DeploymentGate, GateResult

if TYPE_CHECKING:
    from agent_eval.core.report import EvalReport


class CompositeMode(str, Enum):
    """How sub-gates are combined."""

    ALL_PASS = "all_pass"
    """All sub-gates must pass (logical AND)."""

    ANY_PASS = "any_pass"
    """At least one sub-gate must pass (logical OR)."""


class CompositeGate(DeploymentGate):
    """Combines multiple deployment gates with AND/OR logic.

    Parameters
    ----------
    gates:
        Sub-gates to evaluate.
    mode:
        ``ALL_PASS`` requires all gates to pass; ``ANY_PASS`` requires
        at least one to pass.
    gate_name:
        Human-readable identifier for this composite gate.

    Examples
    --------
    ::

        staging_gate = BasicThresholdGate(thresholds={Dimension.ACCURACY: 0.7})
        prod_gate = BasicThresholdGate(thresholds={Dimension.ACCURACY: 0.9})

        # Both must pass for production
        composite = CompositeGate(
            gates=[staging_gate, prod_gate],
            mode=CompositeMode.ALL_PASS,
        )
        result = composite(report)
    """

    def __init__(
        self,
        gates: list[DeploymentGate],
        mode: CompositeMode = CompositeMode.ALL_PASS,
        gate_name: str = "composite",
    ) -> None:
        self._gates = gates
        self._mode = mode
        self._name = gate_name

    @property
    def name(self) -> str:
        """Human-readable gate identifier."""
        return self._name

    def evaluate(self, report: "EvalReport") -> GateResult:
        """Evaluate all sub-gates and combine results.

        Parameters
        ----------
        report:
            The evaluation report to assess.

        Returns
        -------
        GateResult
            Combined result from all sub-gates.
        """
        sub_results: list[GateResult] = []
        for gate in self._gates:
            sub_results.append(gate.evaluate(report))

        passed_count = sum(1 for r in sub_results if r.passed)
        total = len(sub_results)

        details: dict[str, str | float | bool] = {}
        for result in sub_results:
            details[f"{result.gate_name}_passed"] = result.passed
            details.update(result.details)

        if self._mode == CompositeMode.ALL_PASS:
            passed = all(r.passed for r in sub_results)
            reason = (
                f"All {total} gates passed"
                if passed
                else f"{passed_count}/{total} gates passed (ALL required)"
            )
        else:  # ANY_PASS
            passed = any(r.passed for r in sub_results)
            reason = (
                f"{passed_count}/{total} gates passed (at least 1 required)"
                if passed
                else f"No gates passed out of {total} (at least 1 required)"
            )

        return GateResult(
            gate_name=self.name,
            passed=passed,
            reason=reason,
            details=details,
        )
