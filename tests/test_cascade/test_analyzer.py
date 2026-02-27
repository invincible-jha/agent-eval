"""Tests for agent_eval.cascade.analyzer.

Tests cascade failure classification across various graph topologies.
"""
from __future__ import annotations

import pytest

from agent_eval.cascade.analyzer import (
    CascadeAnalysis,
    CascadeAnalyzer,
    FailureClassification,
    StepResult,
)
from agent_eval.cascade.dependency_graph import DependencyGraph


def make_pass(step_id: str) -> StepResult:
    """Create a passing StepResult."""
    return StepResult(step_id=step_id, passed=True, score=1.0)


def make_fail(step_id: str, error: str | None = None) -> StepResult:
    """Create a failing StepResult."""
    return StepResult(step_id=step_id, passed=False, score=0.0, error=error)


class TestCascadeAnalyzerNoFailures:
    """All steps pass — no cascade analysis needed."""

    def test_all_pass_no_root_causes(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])

        results = {"step_1": make_pass("step_1"), "step_2": make_pass("step_2")}
        analyzer = CascadeAnalyzer()
        analysis = analyzer.analyze(graph, results)

        assert analysis.root_causes == []
        assert analysis.cascade_failures == []
        assert analysis.independent_failures == []

    def test_all_pass_total_steps_correct(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])

        results = {"step_1": make_pass("step_1"), "step_2": make_pass("step_2")}
        analyzer = CascadeAnalyzer()
        analysis = analyzer.analyze(graph, results)

        assert analysis.total_steps == 2


class TestCascadeAnalyzerLinearChain:
    """Root at step_2 cascades to steps 3, 4, 5 in a linear chain."""

    def setup_method(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_2"])
        graph.add_step("step_4", depends_on=["step_3"])
        graph.add_step("step_5", depends_on=["step_4"])

        results = {
            "step_1": make_pass("step_1"),
            "step_2": make_fail("step_2"),
            "step_3": make_fail("step_3"),
            "step_4": make_fail("step_4"),
            "step_5": make_fail("step_5"),
        }
        self.analyzer = CascadeAnalyzer()
        self.analysis = self.analyzer.analyze(graph, results)

    def test_step_2_is_root_cause(self) -> None:
        assert "step_2" in self.analysis.root_causes

    def test_steps_3_4_5_are_cascade(self) -> None:
        assert "step_3" in self.analysis.cascade_failures
        assert "step_4" in self.analysis.cascade_failures
        assert "step_5" in self.analysis.cascade_failures

    def test_only_one_root_cause(self) -> None:
        assert len(self.analysis.root_causes) == 1

    def test_three_cascade_failures(self) -> None:
        assert len(self.analysis.cascade_failures) == 3

    def test_actual_failure_count_is_one(self) -> None:
        assert self.analysis.actual_failure_count == 1

    def test_cascade_amplification_is_four(self) -> None:
        # 4 total failures / 1 root cause = 4.0
        assert abs(self.analysis.cascade_amplification - 4.0) < 1e-9

    def test_root_cause_caused_failures_populated(self) -> None:
        analysis = self.analysis.get_step_analysis("step_2")
        assert analysis is not None
        assert len(analysis.caused_failures) == 3


class TestCascadeAnalyzerIndependentFailures:
    """Two independent failures: steps 2 and 7 fail independently."""

    def setup_method(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_1"])
        # step_4 is outside the graph entirely

        results = {
            "step_1": make_pass("step_1"),
            "step_2": make_fail("step_2"),
            "step_3": make_pass("step_3"),
            "step_4": make_fail("step_4"),  # not in graph → INDEPENDENT
        }
        self.analyzer = CascadeAnalyzer()
        self.analysis = self.analyzer.analyze(graph, results)

    def test_step_2_is_root_cause(self) -> None:
        assert "step_2" in self.analysis.root_causes

    def test_step_4_is_independent(self) -> None:
        assert "step_4" in self.analysis.independent_failures

    def test_no_cascade_failures(self) -> None:
        assert self.analysis.cascade_failures == []

    def test_actual_failure_count_is_two(self) -> None:
        assert self.analysis.actual_failure_count == 2


class TestCascadeAnalyzerAllFail:
    """Root at step_1 cascades to everything."""

    def setup_method(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_1"])

        results = {
            "step_1": make_fail("step_1"),
            "step_2": make_fail("step_2"),
            "step_3": make_fail("step_3"),
        }
        self.analyzer = CascadeAnalyzer()
        self.analysis = self.analyzer.analyze(graph, results)

    def test_step_1_is_root_cause(self) -> None:
        assert "step_1" in self.analysis.root_causes

    def test_steps_2_and_3_are_cascade(self) -> None:
        assert "step_2" in self.analysis.cascade_failures
        assert "step_3" in self.analysis.cascade_failures

    def test_root_cause_triggered_cascade(self) -> None:
        analysis = self.analysis.get_step_analysis("step_1")
        assert analysis is not None
        assert set(analysis.caused_failures) == {"step_2", "step_3"}


class TestCascadeAnalyzerDiamondDependency:
    """Diamond: step_4 depends on both step_2 and step_3, both depend on step_1."""

    def setup_method(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_1"])
        graph.add_step("step_4", depends_on=["step_2", "step_3"])

        # step_1 fails → step_2, step_3 cascade → step_4 cascades
        results = {
            "step_1": make_fail("step_1"),
            "step_2": make_fail("step_2"),
            "step_3": make_fail("step_3"),
            "step_4": make_fail("step_4"),
        }
        self.analyzer = CascadeAnalyzer()
        self.analysis = self.analyzer.analyze(graph, results)

    def test_step_1_is_only_root_cause(self) -> None:
        assert self.analysis.root_causes == ["step_1"]

    def test_steps_2_3_4_are_cascade(self) -> None:
        assert "step_2" in self.analysis.cascade_failures
        assert "step_3" in self.analysis.cascade_failures
        assert "step_4" in self.analysis.cascade_failures

    def test_actual_failure_count_is_one(self) -> None:
        assert self.analysis.actual_failure_count == 1


class TestCascadeAnalyzerStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_passed(self) -> None:
        result = StepResult(step_id="s1", passed=True, score=0.9)
        assert result.passed is True
        assert result.score == 0.9
        assert result.error is None

    def test_step_result_failed_with_error(self) -> None:
        result = StepResult(
            step_id="s1",
            passed=False,
            score=0.0,
            error="TimeoutError: step timed out",
        )
        assert result.passed is False
        assert result.error == "TimeoutError: step timed out"


class TestCascadeAnalysis:
    """Tests for CascadeAnalysis computed properties."""

    def test_cascade_amplification_no_failures(self) -> None:
        analysis = CascadeAnalysis(total_steps=5)
        # No failures → amplification = 1.0
        assert analysis.cascade_amplification == 1.0

    def test_cascade_amplification_computed(self) -> None:
        analysis = CascadeAnalysis(
            total_steps=5,
            failed_steps=["s1", "s2", "s3", "s4"],
            root_causes=["s1"],
            cascade_failures=["s2", "s3", "s4"],
        )
        # 4 total / 1 root = 4.0
        assert abs(analysis.cascade_amplification - 4.0) < 1e-9

    def test_get_step_analysis_returns_correct(self) -> None:
        from agent_eval.cascade.analyzer import FailedStepAnalysis

        step_analysis = FailedStepAnalysis(
            step_id="s1",
            classification=FailureClassification.ROOT_CAUSE,
        )
        analysis = CascadeAnalysis(
            total_steps=1,
            failed_steps=["s1"],
            root_causes=["s1"],
            step_analyses=[step_analysis],
        )
        result = analysis.get_step_analysis("s1")
        assert result is not None
        assert result.step_id == "s1"

    def test_get_step_analysis_returns_none_for_missing(self) -> None:
        analysis = CascadeAnalysis(total_steps=0)
        assert analysis.get_step_analysis("nonexistent") is None
