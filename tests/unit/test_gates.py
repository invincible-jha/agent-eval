"""Unit tests for agent_eval.gates.threshold and agent_eval.gates.composite.

Tests BasicThresholdGate construction, mode=all/any logic, empty reports,
and CompositeGate with ALL_PASS/ANY_PASS modes.
"""
from __future__ import annotations

import pytest

from agent_eval.core.evaluator import Dimension, DimensionScore, EvalResult
from agent_eval.core.exceptions import GateError
from agent_eval.core.gate import GateResult
from agent_eval.core.report import EvalReport
from agent_eval.gates.composite import CompositeGate, CompositeMode
from agent_eval.gates.threshold import BasicThresholdGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(dimension: Dimension, score: float, passed: bool) -> DimensionScore:
    return DimensionScore(dimension=dimension, score=score, passed=passed)


def _make_report_with_dimension(
    dimension: Dimension,
    scores: list[tuple[float, bool]],
) -> EvalReport:
    """Build an EvalReport with results for a single dimension."""
    results = [
        EvalResult(
            case_id=f"c{i}",
            run_index=0,
            agent_output="output",
            dimension_scores=[_make_score(dimension, score, passed)],
        )
        for i, (score, passed) in enumerate(scores)
    ]
    return EvalReport.from_results(results)


# ---------------------------------------------------------------------------
# BasicThresholdGate — construction
# ---------------------------------------------------------------------------


class TestBasicThresholdGateConstruction:
    def test_empty_thresholds_raises_gate_error(self) -> None:
        with pytest.raises(GateError, match="At least one threshold"):
            BasicThresholdGate(thresholds={})

    def test_invalid_dimension_name_raises_gate_error(self) -> None:
        with pytest.raises(GateError, match="Unknown dimension"):
            BasicThresholdGate(thresholds={"not_a_dimension": 0.8})

    def test_threshold_above_one_raises_gate_error(self) -> None:
        with pytest.raises(GateError, match="Threshold for"):
            BasicThresholdGate(thresholds={"accuracy": 1.5})

    def test_threshold_below_zero_raises_gate_error(self) -> None:
        with pytest.raises(GateError, match="Threshold for"):
            BasicThresholdGate(thresholds={"accuracy": -0.1})

    def test_valid_thresholds_accepted(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.8, "safety": 1.0})
        assert gate.name == "BasicThresholdGate"

    def test_custom_name(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.8}, gate_name="MyGate")
        assert gate.name == "MyGate"

    def test_all_valid_dimensions_accepted(self) -> None:
        for dim in Dimension:
            gate = BasicThresholdGate(thresholds={dim.value: 0.5})
            assert gate is not None


# ---------------------------------------------------------------------------
# BasicThresholdGate — evaluate (mode=all)
# ---------------------------------------------------------------------------


class TestBasicThresholdGateModeAll:
    def test_empty_report_fails(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.8})
        report = EvalReport.from_results([])
        result = gate.evaluate(report)
        assert result.passed is False
        assert "Empty report" in result.reason

    def test_all_dimensions_meet_threshold_passes(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.8}, mode="all")
        # 100% pass rate
        report = _make_report_with_dimension(
            Dimension.ACCURACY,
            [(1.0, True), (1.0, True)],
        )
        result = gate.evaluate(report)
        assert result.passed is True

    def test_one_dimension_below_threshold_fails(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.9}, mode="all")
        # 50% pass rate, threshold requires 90%
        report = _make_report_with_dimension(
            Dimension.ACCURACY,
            [(1.0, True), (0.0, False)],
        )
        result = gate.evaluate(report)
        assert result.passed is False

    def test_missing_dimension_treated_as_failure(self) -> None:
        gate = BasicThresholdGate(thresholds={"safety": 0.8}, mode="all")
        # Report only has accuracy dimension
        report = _make_report_with_dimension(
            Dimension.ACCURACY,
            [(1.0, True)],
        )
        result = gate.evaluate(report)
        assert result.passed is False

    def test_failing_reason_mentions_dimension(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.9}, mode="all")
        report = _make_report_with_dimension(
            Dimension.ACCURACY,
            [(0.0, False)],
        )
        result = gate.evaluate(report)
        assert "accuracy" in result.reason.lower()

    def test_passing_reason_mentions_dimension_count(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.5}, mode="all")
        report = _make_report_with_dimension(
            Dimension.ACCURACY,
            [(1.0, True)],
        )
        result = gate.evaluate(report)
        assert "1" in result.reason

    def test_details_contains_pass_rate(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.8}, mode="all")
        report = _make_report_with_dimension(
            Dimension.ACCURACY,
            [(1.0, True), (1.0, True)],
        )
        result = gate.evaluate(report)
        assert "accuracy_pass_rate" in result.details

    def test_gate_callable_directly(self) -> None:
        gate = BasicThresholdGate(thresholds={"accuracy": 0.8})
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        # Gates support __call__ via DeploymentGate base
        result = gate(report)
        assert result.passed is True


# ---------------------------------------------------------------------------
# BasicThresholdGate — evaluate (mode=any)
# ---------------------------------------------------------------------------


class TestBasicThresholdGateModeAny:
    def test_at_least_one_passing_passes(self) -> None:
        gate = BasicThresholdGate(
            thresholds={"accuracy": 0.9, "safety": 0.5},
            mode="any",
        )
        # Safety passes (100% rate), accuracy fails (0% rate)
        results = [
            EvalResult(
                case_id="c1",
                run_index=0,
                agent_output="output",
                dimension_scores=[
                    _make_score(Dimension.ACCURACY, 0.0, False),
                    _make_score(Dimension.SAFETY, 1.0, True),
                ],
            )
        ]
        report = EvalReport.from_results(results)
        result = gate.evaluate(report)
        assert result.passed is True

    def test_all_dimensions_failing_fails(self) -> None:
        gate = BasicThresholdGate(
            thresholds={"accuracy": 0.9, "safety": 0.9},
            mode="any",
        )
        results = [
            EvalResult(
                case_id="c1",
                run_index=0,
                agent_output="output",
                dimension_scores=[
                    _make_score(Dimension.ACCURACY, 0.0, False),
                    _make_score(Dimension.SAFETY, 0.0, False),
                ],
            )
        ]
        report = EvalReport.from_results(results)
        result = gate.evaluate(report)
        assert result.passed is False

    def test_any_mode_passing_reason_lists_passing_dimensions(self) -> None:
        gate = BasicThresholdGate(
            thresholds={"accuracy": 0.5},
            mode="any",
        )
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        result = gate.evaluate(report)
        assert "accuracy" in result.reason.lower()

    def test_any_mode_failing_reason_appropriate(self) -> None:
        gate = BasicThresholdGate(
            thresholds={"safety": 0.9},
            mode="any",
        )
        report = _make_report_with_dimension(Dimension.SAFETY, [(0.0, False)])
        result = gate.evaluate(report)
        assert "no" in result.reason.lower() or "fail" in result.reason.lower()


# ---------------------------------------------------------------------------
# CompositeGate — construction
# ---------------------------------------------------------------------------


class TestCompositeGateConstruction:
    def test_default_name(self) -> None:
        gate = CompositeGate(gates=[])
        assert gate.name == "composite"

    def test_custom_name(self) -> None:
        gate = CompositeGate(gates=[], gate_name="my-composite")
        assert gate.name == "my-composite"

    def test_default_mode_is_all_pass(self) -> None:
        gate = CompositeGate(gates=[])
        assert gate._mode == CompositeMode.ALL_PASS


# ---------------------------------------------------------------------------
# CompositeGate — ALL_PASS mode
# ---------------------------------------------------------------------------


class TestCompositeGateAllPass:
    def _make_gate(self, passes: bool, name: str = "sub") -> BasicThresholdGate:
        threshold = 0.5 if passes else 0.9
        return BasicThresholdGate(thresholds={"accuracy": threshold}, gate_name=name)

    def test_all_sub_gates_pass_returns_pass(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        gate = CompositeGate(
            gates=[self._make_gate(True, "g1"), self._make_gate(True, "g2")],
            mode=CompositeMode.ALL_PASS,
        )
        result = gate.evaluate(report)
        assert result.passed is True

    def test_one_sub_gate_fails_returns_fail(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(0.3, False)])
        gate = CompositeGate(
            gates=[self._make_gate(True, "g1"), self._make_gate(False, "g2")],
            mode=CompositeMode.ALL_PASS,
        )
        result = gate.evaluate(report)
        assert result.passed is False

    def test_all_pass_reason_mentions_all_gates(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        gate = CompositeGate(
            gates=[self._make_gate(True, "g1"), self._make_gate(True, "g2")],
            mode=CompositeMode.ALL_PASS,
        )
        result = gate.evaluate(report)
        assert "2" in result.reason

    def test_empty_gates_list_passes_all(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        gate = CompositeGate(gates=[], mode=CompositeMode.ALL_PASS)
        result = gate.evaluate(report)
        # all() of empty list is True
        assert result.passed is True

    def test_details_contains_sub_gate_results(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        gate = CompositeGate(
            gates=[self._make_gate(True, "g1")],
            mode=CompositeMode.ALL_PASS,
        )
        result = gate.evaluate(report)
        assert any("g1" in k for k in result.details.keys())


# ---------------------------------------------------------------------------
# CompositeGate — ANY_PASS mode
# ---------------------------------------------------------------------------


class TestCompositeGateAnyPass:
    def _make_gate(self, passes: bool, name: str = "sub") -> BasicThresholdGate:
        threshold = 0.5 if passes else 0.9
        return BasicThresholdGate(thresholds={"accuracy": threshold}, gate_name=name)

    def test_at_least_one_pass_returns_pass(self) -> None:
        # Report with 100% accuracy pass rate - g1 (threshold 0.5) will pass
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        gate = CompositeGate(
            gates=[self._make_gate(True, "g1"), self._make_gate(False, "g2")],
            mode=CompositeMode.ANY_PASS,
        )
        result = gate.evaluate(report)
        assert result.passed is True

    def test_all_sub_gates_fail_returns_fail(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(0.1, False)])
        gate = CompositeGate(
            gates=[self._make_gate(False, "g1"), self._make_gate(False, "g2")],
            mode=CompositeMode.ANY_PASS,
        )
        result = gate.evaluate(report)
        assert result.passed is False

    def test_any_pass_reason_mentions_count(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        gate = CompositeGate(
            gates=[self._make_gate(True, "g1")],
            mode=CompositeMode.ANY_PASS,
        )
        result = gate.evaluate(report)
        assert "1" in result.reason

    def test_callable_interface_works(self) -> None:
        report = _make_report_with_dimension(Dimension.ACCURACY, [(1.0, True)])
        gate = CompositeGate(
            gates=[self._make_gate(True, "g1")],
            mode=CompositeMode.ANY_PASS,
        )
        result = gate(report)
        assert result.passed is True


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    def test_str_representation_pass(self) -> None:
        result = GateResult(gate_name="my-gate", passed=True, reason="All good")
        assert "PASS" in str(result)
        assert "my-gate" in str(result)

    def test_str_representation_fail(self) -> None:
        result = GateResult(gate_name="my-gate", passed=False, reason="Failed")
        assert "FAIL" in str(result)

    def test_details_default_empty(self) -> None:
        result = GateResult(gate_name="g", passed=True, reason="ok")
        assert result.details == {}
