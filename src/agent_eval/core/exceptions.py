"""Custom exceptions for agent-eval.

All exceptions inherit from AgentEvalError so callers can catch
the entire hierarchy with a single except clause when desired.
"""
from __future__ import annotations


class AgentEvalError(Exception):
    """Base exception for all agent-eval errors."""


class EvaluatorError(AgentEvalError):
    """Raised when an evaluator fails to produce a score.

    This covers configuration errors, unexpected output formats,
    and runtime failures during evaluation.
    """

    def __init__(self, evaluator_name: str, message: str) -> None:
        self.evaluator_name = evaluator_name
        super().__init__(f"[{evaluator_name}] {message}")


class SuiteError(AgentEvalError):
    """Raised when a test suite cannot be loaded or validated.

    Covers malformed YAML/JSON, missing required fields, and
    constraint violations in test cases.
    """

    def __init__(self, suite_name: str, message: str) -> None:
        self.suite_name = suite_name
        super().__init__(f"[suite:{suite_name}] {message}")


class RunnerError(AgentEvalError):
    """Raised when the evaluation runner encounters an unrecoverable error.

    Distinct from per-case failures, which are captured in EvalResult.
    RunnerError indicates a systemic problem (e.g., no evaluators configured).
    """


class GateError(AgentEvalError):
    """Raised when a deployment gate cannot evaluate a report.

    This is a configuration or structural error, not a gate failure.
    A gate that rules 'do not deploy' does not raise GateError; it
    returns a GateResult with passed=False.
    """

    def __init__(self, gate_name: str, message: str) -> None:
        self.gate_name = gate_name
        super().__init__(f"[gate:{gate_name}] {message}")


class ConfigError(AgentEvalError):
    """Raised when eval.yaml or programmatic config is invalid.

    Wraps Pydantic ValidationError messages into a user-friendly form.
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"Configuration error: {message}")


class AgentTimeoutError(AgentEvalError):
    """Raised when an agent under test exceeds its allotted time budget.

    Parameters
    ----------
    timeout_ms:
        The timeout value that was exceeded, in milliseconds.
    case_id:
        The test case identifier that triggered the timeout.
    """

    def __init__(self, timeout_ms: int, case_id: str) -> None:
        self.timeout_ms = timeout_ms
        self.case_id = case_id
        super().__init__(
            f"Agent timed out after {timeout_ms}ms on case {case_id!r}. "
            "Increase max_latency_ms in the test case or suite config."
        )
