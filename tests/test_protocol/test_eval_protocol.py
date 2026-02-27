"""Tests for agent_eval.protocol.eval_protocol."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_eval.protocol.eval_protocol import (
    CURRENT_PROTOCOL_VERSION,
    EvalMetric,
    EvalRequest,
    EvalResponse,
    MetricStatus,
    validate_eval_request,
    validate_eval_response,
)


class TestEvalMetric:
    def test_valid_pass_metric(self) -> None:
        metric = EvalMetric(name="faithfulness", score=0.9, status=MetricStatus.PASS)
        assert metric.score == 0.9
        assert metric.status == MetricStatus.PASS

    def test_valid_fail_metric(self) -> None:
        metric = EvalMetric(name="safety", score=0.3, status=MetricStatus.FAIL)
        assert metric.status == MetricStatus.FAIL

    def test_pass_status_requires_score(self) -> None:
        with pytest.raises(ValidationError, match="score"):
            EvalMetric(name="test", score=None, status=MetricStatus.PASS)

    def test_fail_status_requires_score(self) -> None:
        with pytest.raises(ValidationError, match="score"):
            EvalMetric(name="test", score=None, status=MetricStatus.FAIL)

    def test_error_status_no_score_ok(self) -> None:
        metric = EvalMetric(name="test", score=None, status=MetricStatus.ERROR)
        assert metric.score is None

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvalMetric(name="test", score=1.5, status=MetricStatus.PASS)

    def test_frozen_immutable(self) -> None:
        metric = EvalMetric(name="test", score=0.5, status=MetricStatus.PASS)
        with pytest.raises((ValidationError, TypeError)):
            metric.score = 0.9  # type: ignore[misc]

    def test_threshold_optional(self) -> None:
        metric = EvalMetric(name="test", score=0.8, status=MetricStatus.PASS)
        assert metric.threshold is None

    def test_reason_field(self) -> None:
        metric = EvalMetric(
            name="test",
            score=0.7,
            status=MetricStatus.PASS,
            reason="Looks good",
        )
        assert metric.reason == "Looks good"


class TestEvalRequest:
    def test_minimal_valid_request(self) -> None:
        req = EvalRequest(agent_id="agent_1", input="Hello world")
        assert req.agent_id == "agent_1"
        assert req.input == "Hello world"
        assert req.protocol_version == CURRENT_PROTOCOL_VERSION

    def test_request_id_auto_generated(self) -> None:
        req1 = EvalRequest(agent_id="a", input="q")
        req2 = EvalRequest(agent_id="a", input="q")
        assert req1.request_id != req2.request_id

    def test_invalid_protocol_version_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unsupported protocol version"):
            EvalRequest(
                agent_id="a",
                input="q",
                protocol_version="99.0",
            )

    def test_empty_agent_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvalRequest(agent_id="", input="q")

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvalRequest(agent_id="a", input="")

    def test_timeout_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EvalRequest(agent_id="a", input="q", timeout_seconds=0)
        with pytest.raises(ValidationError):
            EvalRequest(agent_id="a", input="q", timeout_seconds=601)

    def test_optional_fields_defaults(self) -> None:
        req = EvalRequest(agent_id="a", input="q")
        assert req.expected_output is None
        assert req.context == {}
        assert req.metrics == []
        assert req.tags == []

    def test_frozen(self) -> None:
        req = EvalRequest(agent_id="a", input="q")
        with pytest.raises((ValidationError, TypeError)):
            req.agent_id = "b"  # type: ignore[misc]


class TestEvalResponse:
    def _make_response(self, **kwargs: object) -> EvalResponse:
        defaults: dict[str, object] = {
            "request_id": "req-123",
            "agent_output": "The answer is 42.",
        }
        defaults.update(kwargs)
        return EvalResponse(**defaults)  # type: ignore[arg-type]

    def test_minimal_valid_response(self) -> None:
        resp = self._make_response()
        assert resp.agent_output == "The answer is 42."

    def test_passed_metrics_property(self) -> None:
        metrics = [
            EvalMetric(name="m1", score=0.9, status=MetricStatus.PASS),
            EvalMetric(name="m2", score=0.3, status=MetricStatus.FAIL),
        ]
        resp = self._make_response(metrics=metrics)
        assert len(resp.passed_metrics) == 1

    def test_failed_metrics_property(self) -> None:
        metrics = [
            EvalMetric(name="m1", score=0.9, status=MetricStatus.PASS),
            EvalMetric(name="m2", score=0.3, status=MetricStatus.FAIL),
        ]
        resp = self._make_response(metrics=metrics)
        assert len(resp.failed_metrics) == 1

    def test_overall_passed_all_pass(self) -> None:
        metrics = [
            EvalMetric(name="m1", score=0.9, status=MetricStatus.PASS),
            EvalMetric(name="m2", score=0.8, status=MetricStatus.PASS),
        ]
        resp = self._make_response(metrics=metrics)
        assert resp.overall_passed is True

    def test_overall_passed_any_fail(self) -> None:
        metrics = [
            EvalMetric(name="m1", score=0.9, status=MetricStatus.PASS),
            EvalMetric(name="m2", score=0.3, status=MetricStatus.FAIL),
        ]
        resp = self._make_response(metrics=metrics)
        assert resp.overall_passed is False

    def test_pass_rate_computed(self) -> None:
        metrics = [
            EvalMetric(name="m1", score=0.9, status=MetricStatus.PASS),
            EvalMetric(name="m2", score=0.9, status=MetricStatus.PASS),
            EvalMetric(name="m3", score=0.3, status=MetricStatus.FAIL),
        ]
        resp = self._make_response(metrics=metrics)
        assert abs(resp.pass_rate - 2 / 3) < 0.01

    def test_metric_by_name_found(self) -> None:
        metrics = [EvalMetric(name="safety", score=0.9, status=MetricStatus.PASS)]
        resp = self._make_response(metrics=metrics)
        m = resp.metric_by_name("safety")
        assert m is not None
        assert m.name == "safety"

    def test_metric_by_name_not_found(self) -> None:
        resp = self._make_response()
        assert resp.metric_by_name("nonexistent") is None

    def test_no_metrics_overall_passed_true(self) -> None:
        resp = self._make_response()
        assert resp.overall_passed is True
        assert resp.pass_rate == 1.0


class TestValidationHelpers:
    def test_validate_eval_request_dict(self) -> None:
        data = {"agent_id": "agent1", "input": "test question"}
        req = validate_eval_request(data)
        assert isinstance(req, EvalRequest)

    def test_validate_eval_request_invalid(self) -> None:
        with pytest.raises(ValidationError):
            validate_eval_request({"agent_id": "a"})  # missing input

    def test_validate_eval_response_dict(self) -> None:
        data = {"request_id": "r1", "agent_output": "answer"}
        resp = validate_eval_response(data)
        assert isinstance(resp, EvalResponse)

    def test_validate_eval_response_invalid(self) -> None:
        with pytest.raises(ValidationError):
            validate_eval_response({"request_id": "r1"})  # missing agent_output
