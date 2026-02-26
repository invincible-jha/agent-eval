"""Deployment gate abstractions for agent-eval.

A DeploymentGate examines an EvalReport and decides whether the agent
is ready to deploy. Gates are used in CI/CD pipelines to block
deployments that fail quality thresholds.

NOTE: This is the plugin boundary for gates. The ABC lives here.
Commodity implementations (BasicThresholdGate, CompositeGate) live in
agent_eval.gates.* Additional implementations register via the plugin system.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval.core.report import EvalReport


@dataclass(frozen=True)
class GateResult:
    """Result of a deployment gate evaluation.

    Parameters
    ----------
    gate_name:
        The name of the gate that produced this result.
    passed:
        True if the gate approves the deployment; False if it blocks.
    reason:
        Human-readable explanation of the decision.
    details:
        Per-dimension breakdown of which thresholds passed or failed.
        Keys are dimension names; values are pass/fail booleans or scores.
    """

    gate_name: str
    passed: bool
    reason: str
    details: dict[str, str | float | bool] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"Gate {self.gate_name!r}: {status} — {self.reason}"


class DeploymentGate(ABC):
    """Abstract base class for deployment gates.

    A gate receives an EvalReport and returns a GateResult.
    Gates are composable: CompositeGate wraps multiple gates with
    AND/OR logic.

    NOTE: Additional gate types (e.g., regression detection, drift monitoring)
    are not included in this package. They can be added via the
    plugin system by subclassing this ABC.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable gate identifier."""

    @abstractmethod
    def evaluate(self, report: "EvalReport") -> GateResult:
        """Evaluate an EvalReport and return a gate decision.

        Parameters
        ----------
        report:
            The completed evaluation report to assess.

        Returns
        -------
        GateResult
            Decision with pass/fail status and explanation.
        """

    def __call__(self, report: "EvalReport") -> GateResult:
        """Allow gates to be called directly: ``gate(report)``."""
        return self.evaluate(report)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
