"""Unit tests for agent_eval.core.exceptions.

Covers all custom exception classes: hierarchy, message formatting,
and attribute presence.
"""
from __future__ import annotations

import pytest

from agent_eval.core.exceptions import (
    AgentEvalError,
    AgentTimeoutError,
    ConfigError,
    EvaluatorError,
    GateError,
    RunnerError,
    SuiteError,
)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_evaluator_error_is_agent_eval_error(self) -> None:
        exc = EvaluatorError("MyEval", "bad config")
        assert isinstance(exc, AgentEvalError)

    def test_suite_error_is_agent_eval_error(self) -> None:
        exc = SuiteError("my_suite", "missing field")
        assert isinstance(exc, AgentEvalError)

    def test_runner_error_is_agent_eval_error(self) -> None:
        exc = RunnerError("no evaluators")
        assert isinstance(exc, AgentEvalError)

    def test_gate_error_is_agent_eval_error(self) -> None:
        exc = GateError("prod_gate", "threshold out of range")
        assert isinstance(exc, AgentEvalError)

    def test_config_error_is_agent_eval_error(self) -> None:
        exc = ConfigError("missing runner section")
        assert isinstance(exc, AgentEvalError)

    def test_agent_timeout_error_is_agent_eval_error(self) -> None:
        exc = AgentTimeoutError(timeout_ms=5000, case_id="case_001")
        assert isinstance(exc, AgentEvalError)

    def test_all_can_be_caught_as_base(self) -> None:
        exceptions = [
            EvaluatorError("e", "m"),
            SuiteError("s", "m"),
            RunnerError("m"),
            GateError("g", "m"),
            ConfigError("m"),
            AgentTimeoutError(1000, "c"),
        ]
        for exc in exceptions:
            with pytest.raises(AgentEvalError):
                raise exc


# ---------------------------------------------------------------------------
# EvaluatorError
# ---------------------------------------------------------------------------


class TestEvaluatorError:
    def test_message_format_includes_evaluator_name(self) -> None:
        exc = EvaluatorError("AccuracyEvaluator", "score out of range")
        assert "[AccuracyEvaluator]" in str(exc)
        assert "score out of range" in str(exc)

    def test_evaluator_name_attribute(self) -> None:
        exc = EvaluatorError("LatencyEvaluator", "negative max_ms")
        assert exc.evaluator_name == "LatencyEvaluator"

    def test_empty_message_still_formats(self) -> None:
        exc = EvaluatorError("X", "")
        assert "[X]" in str(exc)


# ---------------------------------------------------------------------------
# SuiteError
# ---------------------------------------------------------------------------


class TestSuiteError:
    def test_message_format_includes_suite_name(self) -> None:
        exc = SuiteError("qa_basic", "missing 'cases' key")
        assert "suite:qa_basic" in str(exc)
        assert "missing 'cases' key" in str(exc)

    def test_suite_name_attribute(self) -> None:
        exc = SuiteError("safety_v2", "invalid YAML")
        assert exc.suite_name == "safety_v2"


# ---------------------------------------------------------------------------
# GateError
# ---------------------------------------------------------------------------


class TestGateError:
    def test_message_format_includes_gate_name(self) -> None:
        exc = GateError("prod_gate", "threshold must be in [0.0, 1.0]")
        assert "gate:prod_gate" in str(exc)
        assert "threshold must be in [0.0, 1.0]" in str(exc)

    def test_gate_name_attribute(self) -> None:
        exc = GateError("staging_gate", "empty thresholds")
        assert exc.gate_name == "staging_gate"


# ---------------------------------------------------------------------------
# ConfigError
# ---------------------------------------------------------------------------


class TestConfigError:
    def test_message_format_has_configuration_prefix(self) -> None:
        exc = ConfigError("runs_per_case must be >= 1")
        assert "Configuration error" in str(exc)
        assert "runs_per_case must be >= 1" in str(exc)


# ---------------------------------------------------------------------------
# AgentTimeoutError
# ---------------------------------------------------------------------------


class TestAgentTimeoutError:
    def test_attributes_stored(self) -> None:
        exc = AgentTimeoutError(timeout_ms=3000, case_id="case_42")
        assert exc.timeout_ms == 3000
        assert exc.case_id == "case_42"

    def test_message_contains_timeout_and_case(self) -> None:
        exc = AgentTimeoutError(timeout_ms=1500, case_id="q1")
        msg = str(exc)
        assert "1500ms" in msg
        assert "q1" in msg

    @pytest.mark.parametrize("timeout_ms,case_id", [
        (100, "first"),
        (60000, "long_running"),
        (0, "zero_timeout"),
    ])
    def test_various_timeout_values(self, timeout_ms: int, case_id: str) -> None:
        exc = AgentTimeoutError(timeout_ms=timeout_ms, case_id=case_id)
        assert exc.timeout_ms == timeout_ms
        assert exc.case_id == case_id


# ---------------------------------------------------------------------------
# RunnerError
# ---------------------------------------------------------------------------


class TestRunnerError:
    def test_is_plain_message(self) -> None:
        exc = RunnerError("No evaluators configured")
        assert "No evaluators configured" in str(exc)
