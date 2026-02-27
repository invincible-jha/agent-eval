"""Tests for agent_eval.statistical.report.

Verifies StatisticalReport creation from StatisticalResult and output methods.
"""
from __future__ import annotations

import json

import pytest

from agent_eval.core.evaluator import (
    Dimension,
    DimensionScore,
    EvalResult,
)
from agent_eval.statistical.report import StatisticalReport
from agent_eval.statistical.runner import StatisticalRunner


def _make_result(passed: bool, score: float = 1.0) -> EvalResult:
    return EvalResult(
        case_id="test",
        run_index=0,
        agent_output="output",
        dimension_scores=[
            DimensionScore(
                dimension=Dimension.ACCURACY,
                score=score,
                passed=passed,
            )
        ],
    )


def always_pass() -> EvalResult:
    return _make_result(passed=True, score=1.0)


def always_fail() -> EvalResult:
    return _make_result(passed=False, score=0.0)


class TestStatisticalReportFromResult:
    """Tests for StatisticalReport.from_result factory."""

    def test_from_result_all_pass(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result, label="all_pass")

        assert report.n_runs == 5
        assert report.n_passed == 5
        assert report.n_failed == 0
        assert report.n_errors == 0
        assert report.pass_at_1 == 1.0
        assert report.pass_at_3 == 1.0
        assert report.pass_at_5 == 1.0
        assert report.label == "all_pass"

    def test_from_result_all_fail(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        stat_result = runner.run(always_fail)
        report = StatisticalReport.from_result(stat_result)

        assert report.n_passed == 0
        assert report.pass_at_1 == 0.0
        assert report.pass_at_3 == 0.0
        assert report.pass_at_5 == 0.0

    def test_ci_bounds_present_and_ordered(self) -> None:
        runner = StatisticalRunner(n_runs=10)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result)

        assert report.ci_lower <= report.ci_upper
        assert 0.0 <= report.ci_lower <= 1.0
        assert 0.0 <= report.ci_upper <= 1.0

    def test_score_statistics_all_pass(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result)

        assert abs(report.mean_score - 1.0) < 1e-9
        assert abs(report.score_stddev - 0.0) < 1e-9


class TestStatisticalReportToDict:
    """Tests for the to_dict() serialization method."""

    def setup_method(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        stat_result = runner.run(always_pass)
        self.report = StatisticalReport.from_result(stat_result, label="dict_test")
        self.data = self.report.to_dict()

    def test_top_level_keys_present(self) -> None:
        required_keys = {
            "label", "created_at", "runs", "pass_at_k",
            "confidence_interval_95", "score_statistics",
        }
        assert required_keys.issubset(set(self.data.keys()))

    def test_label_in_dict(self) -> None:
        assert self.data["label"] == "dict_test"

    def test_pass_at_k_keys_present(self) -> None:
        pak = self.data["pass_at_k"]
        assert isinstance(pak, dict)
        assert "pass_at_1" in pak
        assert "pass_at_3" in pak
        assert "pass_at_5" in pak

    def test_ci_keys_present(self) -> None:
        ci = self.data["confidence_interval_95"]
        assert isinstance(ci, dict)
        assert "lower" in ci
        assert "upper" in ci

    def test_runs_section_correct(self) -> None:
        runs = self.data["runs"]
        assert runs["total"] == 5
        assert runs["passed"] == 5
        assert runs["failed"] == 0


class TestStatisticalReportToJson:
    """Tests for the to_json() serialization method."""

    def test_to_json_is_valid_json(self) -> None:
        runner = StatisticalRunner(n_runs=3)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result)

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_round_trips(self) -> None:
        runner = StatisticalRunner(n_runs=3)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result, label="json_test")

        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["label"] == "json_test"


class TestStatisticalReportToText:
    """Tests for the to_text() human-readable output method."""

    def test_to_text_contains_pass_at_k(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result)

        text = report.to_text()
        assert "pass@1" in text
        assert "pass@3" in text
        assert "pass@5" in text

    def test_to_text_contains_confidence_interval(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result)

        text = report.to_text()
        assert "Confidence" in text

    def test_to_text_contains_label(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        stat_result = runner.run(always_pass)
        report = StatisticalReport.from_result(stat_result, label="my_eval")

        text = report.to_text()
        assert "my_eval" in text

    def test_to_text_is_string(self) -> None:
        runner = StatisticalRunner(n_runs=3)
        stat_result = runner.run(always_fail)
        report = StatisticalReport.from_result(stat_result)
        assert isinstance(report.to_text(), str)
