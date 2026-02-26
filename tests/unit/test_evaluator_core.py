"""Unit tests for agent_eval.core.evaluator.

Tests Dimension enum, DimensionScore dataclass, EvalResult dataclass,
Evaluator ABC behaviour, and measure_latency utility.
"""
from __future__ import annotations

import time
from typing import ClassVar

import pytest

from agent_eval.core.evaluator import (
    Dimension,
    DimensionScore,
    EvalResult,
    Evaluator,
    measure_latency,
)


# ---------------------------------------------------------------------------
# Dimension enum
# ---------------------------------------------------------------------------


class TestDimension:
    ALL_VALUES: ClassVar[list[str]] = [
        "accuracy", "latency", "cost", "safety", "format", "custom"
    ]

    def test_all_expected_members_exist(self) -> None:
        for value in self.ALL_VALUES:
            assert Dimension(value) is not None

    @pytest.mark.parametrize("value", ["accuracy", "latency", "cost", "safety", "format", "custom"])
    def test_from_string_roundtrip(self, value: str) -> None:
        dim = Dimension(value)
        assert dim.value == value

    def test_is_str_subclass(self) -> None:
        assert isinstance(Dimension.ACCURACY, str)

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            Dimension("nonexistent")


# ---------------------------------------------------------------------------
# DimensionScore
# ---------------------------------------------------------------------------


class TestDimensionScore:
    def test_valid_score_at_zero(self) -> None:
        ds = DimensionScore(dimension=Dimension.ACCURACY, score=0.0, passed=False)
        assert ds.score == 0.0

    def test_valid_score_at_one(self) -> None:
        ds = DimensionScore(dimension=Dimension.SAFETY, score=1.0, passed=True)
        assert ds.score == 1.0

    def test_valid_mid_range_score(self) -> None:
        ds = DimensionScore(dimension=Dimension.COST, score=0.75, passed=True)
        assert ds.score == 0.75

    @pytest.mark.parametrize("score", [-0.01, 1.001, 2.0, -1.0])
    def test_score_out_of_range_raises(self, score: float) -> None:
        with pytest.raises(ValueError, match="must be in"):
            DimensionScore(dimension=Dimension.LATENCY, score=score, passed=False)

    def test_frozen_prevents_mutation(self) -> None:
        ds = DimensionScore(dimension=Dimension.FORMAT, score=0.5, passed=True)
        with pytest.raises((AttributeError, TypeError)):
            ds.score = 0.9  # type: ignore[misc]

    def test_reason_defaults_empty_string(self) -> None:
        ds = DimensionScore(dimension=Dimension.ACCURACY, score=1.0, passed=True)
        assert ds.reason == ""

    def test_raw_value_defaults_none(self) -> None:
        ds = DimensionScore(dimension=Dimension.LATENCY, score=0.8, passed=True)
        assert ds.raw_value is None

    def test_raw_value_set(self) -> None:
        ds = DimensionScore(
            dimension=Dimension.LATENCY, score=0.8, passed=True,
            reason="within threshold", raw_value=1234.5
        )
        assert ds.raw_value == 1234.5


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def _make_score(self, passed: bool, score: float = 1.0) -> DimensionScore:
        return DimensionScore(
            dimension=Dimension.ACCURACY,
            score=score,
            passed=passed,
        )

    def test_passed_when_all_scores_pass(self) -> None:
        result = EvalResult(
            case_id="c1",
            run_index=0,
            agent_output="yes",
            dimension_scores=[
                DimensionScore(Dimension.ACCURACY, 1.0, True),
                DimensionScore(Dimension.SAFETY, 1.0, True),
            ],
        )
        assert result.passed is True

    def test_not_passed_when_any_score_fails(self) -> None:
        result = EvalResult(
            case_id="c2",
            run_index=0,
            agent_output="yes",
            dimension_scores=[
                DimensionScore(Dimension.ACCURACY, 1.0, True),
                DimensionScore(Dimension.SAFETY, 0.0, False),
            ],
        )
        assert result.passed is False

    def test_not_passed_with_no_dimension_scores(self) -> None:
        result = EvalResult(case_id="c3", run_index=0, agent_output="")
        assert result.passed is False

    def test_overall_score_mean(self) -> None:
        result = EvalResult(
            case_id="c4",
            run_index=0,
            agent_output="ans",
            dimension_scores=[
                DimensionScore(Dimension.ACCURACY, 0.8, True),
                DimensionScore(Dimension.LATENCY, 0.6, True),
            ],
        )
        assert result.overall_score == pytest.approx(0.7)

    def test_overall_score_zero_when_no_scores(self) -> None:
        result = EvalResult(case_id="c5", run_index=0, agent_output="")
        assert result.overall_score == 0.0

    def test_score_for_returns_correct_dimension(self) -> None:
        acc_score = DimensionScore(Dimension.ACCURACY, 0.9, True)
        lat_score = DimensionScore(Dimension.LATENCY, 0.7, True)
        result = EvalResult(
            case_id="c6", run_index=0, agent_output="x",
            dimension_scores=[acc_score, lat_score],
        )
        assert result.score_for(Dimension.ACCURACY) is acc_score
        assert result.score_for(Dimension.LATENCY) is lat_score

    def test_score_for_returns_none_for_missing_dimension(self) -> None:
        result = EvalResult(
            case_id="c7", run_index=0, agent_output="x",
            dimension_scores=[DimensionScore(Dimension.ACCURACY, 1.0, True)],
        )
        assert result.score_for(Dimension.COST) is None

    def test_error_field_defaults_none(self) -> None:
        result = EvalResult(case_id="c8", run_index=0, agent_output="ok")
        assert result.error is None

    def test_metadata_defaults_empty_dict(self) -> None:
        result = EvalResult(case_id="c9", run_index=0, agent_output="ok")
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Evaluator ABC concrete stub
# ---------------------------------------------------------------------------


class _StubEvaluator(Evaluator):
    """Minimal concrete Evaluator for testing ABC behaviour."""

    @property
    def dimension(self) -> Dimension:
        return Dimension.CUSTOM

    def evaluate(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        return DimensionScore(dimension=self.dimension, score=1.0, passed=True)


class TestEvaluatorABC:
    def test_name_defaults_to_class_name(self) -> None:
        evaluator = _StubEvaluator()
        assert evaluator.name == "_StubEvaluator"

    def test_repr_contains_dimension(self) -> None:
        evaluator = _StubEvaluator()
        assert "custom" in repr(evaluator)

    def test_evaluate_batch_calls_evaluate_sequentially(self) -> None:
        evaluator = _StubEvaluator()
        cases = [
            ("c1", "output1", "expected1", {}),
            ("c2", "output2", None, {"key": "val"}),
        ]
        scores = evaluator.evaluate_batch(cases)
        assert len(scores) == 2
        for score in scores:
            assert isinstance(score, DimensionScore)

    def test_evaluate_batch_preserves_order(self) -> None:
        call_order: list[str] = []

        class OrderTracker(Evaluator):
            @property
            def dimension(self) -> Dimension:
                return Dimension.CUSTOM

            def evaluate(
                self,
                case_id: str,
                agent_output: str,
                expected_output: str | None,
                metadata: dict[str, str | int | float | bool],
            ) -> DimensionScore:
                call_order.append(case_id)
                return DimensionScore(Dimension.CUSTOM, 1.0, True)

        tracker = OrderTracker()
        tracker.evaluate_batch([("a", "", None, {}), ("b", "", None, {}), ("c", "", None, {})])
        assert call_order == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# measure_latency
# ---------------------------------------------------------------------------


class TestMeasureLatency:
    def test_returns_positive_milliseconds(self) -> None:
        start = time.perf_counter()
        time.sleep(0.01)
        elapsed = measure_latency(start)
        assert elapsed >= 10.0

    def test_immediate_call_returns_near_zero(self) -> None:
        start = time.perf_counter()
        elapsed = measure_latency(start)
        # Should be under 100ms for an immediate call
        assert elapsed < 100.0

    def test_returns_float(self) -> None:
        start = time.perf_counter()
        result = measure_latency(start)
        assert isinstance(result, float)
