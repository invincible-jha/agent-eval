"""Unit tests for agent_eval.evaluators.latency.

Tests BasicLatencyEvaluator threshold comparison, metadata overrides,
score scaling, and edge cases.
"""
from __future__ import annotations

import pytest

from agent_eval.core.evaluator import Dimension
from agent_eval.core.exceptions import EvaluatorError
from agent_eval.evaluators.latency import BasicLatencyEvaluator


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestBasicLatencyEvaluatorConstruction:
    def test_default_max_ms(self) -> None:
        ev = BasicLatencyEvaluator()
        assert ev.max_ms == 5000

    def test_custom_max_ms(self) -> None:
        ev = BasicLatencyEvaluator(max_ms=2000)
        assert ev.max_ms == 2000

    def test_zero_max_ms_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="max_ms must be positive"):
            BasicLatencyEvaluator(max_ms=0)

    def test_negative_max_ms_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="max_ms must be positive"):
            BasicLatencyEvaluator(max_ms=-100)

    def test_dimension_property(self) -> None:
        ev = BasicLatencyEvaluator()
        assert ev.dimension == Dimension.LATENCY

    def test_name_property(self) -> None:
        ev = BasicLatencyEvaluator()
        assert ev.name == "BasicLatencyEvaluator"


# ---------------------------------------------------------------------------
# evaluate — within threshold
# ---------------------------------------------------------------------------


class TestLatencyWithinThreshold:
    @pytest.fixture
    def ev(self) -> BasicLatencyEvaluator:
        return BasicLatencyEvaluator(max_ms=5000)

    def test_zero_latency_passes_with_high_score(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 0.0})
        assert result.passed is True
        assert result.score >= 0.9

    def test_latency_at_threshold_passes(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 5000.0})
        assert result.passed is True

    def test_latency_just_under_threshold_passes(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 4999.0})
        assert result.passed is True

    def test_score_between_0_9_and_1_0_within_threshold(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 1000.0})
        assert 0.9 <= result.score <= 1.0

    def test_reason_contains_latency_and_threshold(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 100.0})
        assert "100" in result.reason or "ms" in result.reason.lower()

    def test_raw_value_is_latency(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 250.0})
        assert result.raw_value == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# evaluate — exceeds threshold
# ---------------------------------------------------------------------------


class TestLatencyExceedsThreshold:
    @pytest.fixture
    def ev(self) -> BasicLatencyEvaluator:
        return BasicLatencyEvaluator(max_ms=1000)

    def test_latency_over_threshold_fails(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 2000.0})
        assert result.passed is False

    def test_score_zero_at_double_threshold(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 2000.0})
        assert result.score == 0.0

    def test_score_zero_beyond_double_threshold(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 5000.0})
        assert result.score == 0.0

    def test_reason_contains_overage(self, ev: BasicLatencyEvaluator) -> None:
        result = ev.evaluate("c1", "output", None, {"latency_ms": 1500.0})
        assert "exceeds" in result.reason.lower() or "+" in result.reason


# ---------------------------------------------------------------------------
# evaluate — metadata threshold override
# ---------------------------------------------------------------------------


class TestLatencyMetadataOverride:
    def test_max_latency_ms_from_metadata_overrides_constructor(self) -> None:
        ev = BasicLatencyEvaluator(max_ms=1000)
        # Case provides a 10-second threshold via metadata
        result = ev.evaluate(
            "c1", "output", None,
            {"latency_ms": 8000.0, "max_latency_ms": 10000}
        )
        assert result.passed is True

    def test_zero_threshold_in_metadata_falls_back_to_constructor(self) -> None:
        ev = BasicLatencyEvaluator(max_ms=5000)
        result = ev.evaluate(
            "c1", "output", None,
            {"latency_ms": 100.0, "max_latency_ms": 0}
        )
        # Should fall back to constructor's 5000ms
        assert result.passed is True

    def test_missing_latency_ms_defaults_to_zero(self) -> None:
        ev = BasicLatencyEvaluator(max_ms=5000)
        result = ev.evaluate("c1", "output", None, {})
        assert result.passed is True

    def test_non_numeric_latency_defaults_to_zero(self) -> None:
        ev = BasicLatencyEvaluator(max_ms=5000)
        result = ev.evaluate("c1", "output", None, {"latency_ms": "fast"})  # type: ignore[dict-item]
        assert result.passed is True
