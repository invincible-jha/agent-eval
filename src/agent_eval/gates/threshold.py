"""Basic threshold gate for agent-eval.

Blocks deployment when any (or all) dimension scores fall below
their configured thresholds.

NOTE: This is a commodity threshold gate. It performs static threshold
comparison. It is NOT a regression detection gate, does NOT compare
against baseline runs, and does NOT detect gradual score degradation.
Gates with regression detection and drift monitoring are available via
the plugin system.
"""
from __future__ import annotations

from typing import Literal

from agent_eval.core.evaluator import Dimension
from agent_eval.core.exceptions import GateError
from agent_eval.core.gate import DeploymentGate, GateResult
from agent_eval.core.report import EvalReport


class BasicThresholdGate(DeploymentGate):
    """Gate that blocks deployment when dimension scores fall below thresholds.

    Checks per-dimension pass rates (fraction of cases passing that dimension)
    against configured minimum thresholds. Can operate in two modes:
    - "all": ALL configured dimensions must pass their thresholds
    - "any": AT LEAST ONE configured dimension must pass its threshold

    NOTE: This is NOT a regression detection gate. It does not compare
    against previous runs or detect gradual quality degradation.
    Regression gates are available via the plugin system.

    Parameters
    ----------
    thresholds:
        Mapping of dimension name to minimum acceptable pass rate.
        Keys must be valid Dimension enum values (e.g., "accuracy", "safety").
        Values must be in [0.0, 1.0].
    mode:
        "all" (default) — all configured dimensions must pass.
        "any" — at least one configured dimension must pass.
    gate_name:
        Human-readable name for this gate instance.

    Example
    -------
    ::

        gate = BasicThresholdGate(
            thresholds={"accuracy": 0.8, "safety": 1.0, "latency": 0.9},
            mode="all",
        )
        result = gate.evaluate(report)
    """

    def __init__(
        self,
        thresholds: dict[str, float],
        mode: Literal["all", "any"] = "all",
        gate_name: str = "BasicThresholdGate",
    ) -> None:
        if not thresholds:
            raise GateError(gate_name, "At least one threshold must be configured")

        # Validate dimension names and threshold values
        validated: dict[Dimension, float] = {}
        for dim_name, threshold in thresholds.items():
            try:
                dimension = Dimension(dim_name)
            except ValueError:
                valid = [d.value for d in Dimension]
                raise GateError(
                    gate_name,
                    f"Unknown dimension {dim_name!r}. Valid dimensions: {valid}",
                ) from None
            if not (0.0 <= threshold <= 1.0):
                raise GateError(
                    gate_name,
                    f"Threshold for {dim_name!r} must be in [0.0, 1.0], got {threshold}",
                )
            validated[dimension] = threshold

        self._thresholds = validated
        self._mode = mode
        self._gate_name = gate_name

    @property
    def name(self) -> str:
        return self._gate_name

    def evaluate(self, report: EvalReport) -> GateResult:
        """Check EvalReport dimension pass rates against thresholds.

        Parameters
        ----------
        report:
            The completed evaluation report.

        Returns
        -------
        GateResult
            Passed if thresholds are met according to mode setting.
        """
        if not report.results:
            return GateResult(
                gate_name=self.name,
                passed=False,
                reason="Empty report: no results to evaluate",
            )

        dimension_checks: dict[str, str | float | bool] = {}
        dimension_results: list[bool] = []

        for dimension, threshold in self._thresholds.items():
            summary = report.summary_for(dimension)
            if summary is None:
                # Dimension not evaluated: treat as failure
                dimension_checks[dimension.value] = False
                dimension_checks[f"{dimension.value}_pass_rate"] = 0.0
                dimension_checks[f"{dimension.value}_threshold"] = threshold
                dimension_results.append(False)
                continue

            passes = summary.pass_rate >= threshold
            dimension_checks[dimension.value] = passes
            dimension_checks[f"{dimension.value}_pass_rate"] = round(summary.pass_rate, 4)
            dimension_checks[f"{dimension.value}_threshold"] = threshold
            dimension_results.append(passes)

        if not dimension_results:
            overall_passed = False
            reason = "No dimension results available"
        elif self._mode == "all":
            overall_passed = all(dimension_results)
            if overall_passed:
                reason = f"All {len(dimension_results)} dimensions meet thresholds"
            else:
                failing = [
                    f"{dim.value} ({dimension_checks.get(f'{dim.value}_pass_rate', 0.0):.1%} "
                    f"< {threshold:.1%})"
                    for dim, threshold in self._thresholds.items()
                    if not dimension_checks.get(dim.value, False)
                ]
                reason = f"Failing dimensions: {', '.join(failing)}"
        else:  # mode == "any"
            overall_passed = any(dimension_results)
            if overall_passed:
                passing = [
                    dim.value
                    for dim, _ in self._thresholds.items()
                    if dimension_checks.get(dim.value, False)
                ]
                reason = f"Passing dimensions: {', '.join(passing)}"
            else:
                reason = "No dimensions meet their thresholds"

        return GateResult(
            gate_name=self.name,
            passed=overall_passed,
            reason=reason,
            details=dimension_checks,
        )
