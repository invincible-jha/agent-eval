"""Unit tests for agent_eval.core.report.

Tests EvalReport.from_results(), all computed properties, to_dict(),
to_json(), to_markdown(), and _aggregate_dimensions().
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent_eval.core.evaluator import Dimension, DimensionScore, EvalResult
from agent_eval.core.report import DimensionSummary, EvalReport


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _make_score(dimension: Dimension, score: float, passed: bool) -> DimensionScore:
    return DimensionScore(dimension=dimension, score=score, passed=passed)


def _make_result(
    case_id: str,
    passed_scores: list[bool],
    latency_ms: float = 100.0,
    error: str | None = None,
) -> EvalResult:
    scores = [_make_score(Dimension.ACCURACY, 1.0 if p else 0.0, p) for p in passed_scores]
    return EvalResult(
        case_id=case_id,
        run_index=0,
        agent_output="output",
        dimension_scores=scores,
        latency_ms=latency_ms,
        error=error,
    )


# ---------------------------------------------------------------------------
# EvalReport.from_results()
# ---------------------------------------------------------------------------


class TestEvalReportFromResults:
    def test_basic_from_results(self) -> None:
        results = [_make_result("c1", [True]), _make_result("c2", [True])]
        report = EvalReport.from_results(results, suite_name="test", agent_name="agent")
        assert report.suite_name == "test"
        assert report.agent_name == "agent"
        assert report.total_cases == 2

    def test_empty_results(self) -> None:
        report = EvalReport.from_results([])
        assert report.total_cases == 0
        assert report.overall_pass_rate == 0.0

    def test_run_config_stored(self) -> None:
        cfg = {"runs_per_case": 3, "evaluators": "accuracy"}
        report = EvalReport.from_results([], run_config=cfg)
        assert report.run_config["runs_per_case"] == 3

    def test_created_at_is_utc_datetime(self) -> None:
        report = EvalReport.from_results([])
        assert report.created_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Computed properties
# ---------------------------------------------------------------------------


class TestEvalReportProperties:
    def test_passed_cases_counts_all_passing(self) -> None:
        results = [
            _make_result("c1", [True]),
            _make_result("c2", [True]),
            _make_result("c3", [False]),
        ]
        report = EvalReport.from_results(results)
        assert report.passed_cases == 2

    def test_failed_cases_counts_failures_without_error(self) -> None:
        results = [
            _make_result("c1", [False]),
            _make_result("c2", [False], error="boom"),
        ]
        report = EvalReport.from_results(results)
        assert report.failed_cases == 1

    def test_error_cases_counts_error_results(self) -> None:
        results = [
            _make_result("c1", [], error="timeout"),
            _make_result("c2", [True]),
        ]
        report = EvalReport.from_results(results)
        assert report.error_cases == 1

    def test_overall_pass_rate_excludes_errors(self) -> None:
        results = [
            _make_result("c1", [True]),
            _make_result("c2", [True]),
            _make_result("c3", [], error="err"),
        ]
        report = EvalReport.from_results(results)
        # 2 non-error results, 2 passed
        assert report.overall_pass_rate == pytest.approx(1.0)

    def test_overall_pass_rate_zero_when_all_errors(self) -> None:
        results = [_make_result("c1", [], error="err")]
        report = EvalReport.from_results(results)
        assert report.overall_pass_rate == 0.0

    def test_mean_latency_averages_positive_latencies(self) -> None:
        results = [
            _make_result("c1", [True], latency_ms=100.0),
            _make_result("c2", [True], latency_ms=200.0),
        ]
        report = EvalReport.from_results(results)
        assert report.mean_latency_ms == pytest.approx(150.0)

    def test_mean_latency_zero_when_no_results(self) -> None:
        report = EvalReport.from_results([])
        assert report.mean_latency_ms == 0.0

    def test_summary_for_returns_correct_dimension(self) -> None:
        results = [
            EvalResult(
                case_id="c1", run_index=0, agent_output="ok",
                dimension_scores=[
                    _make_score(Dimension.ACCURACY, 0.9, True),
                    _make_score(Dimension.SAFETY, 1.0, True),
                ],
            )
        ]
        report = EvalReport.from_results(results)
        acc_summary = report.summary_for(Dimension.ACCURACY)
        assert acc_summary is not None
        assert acc_summary.dimension == Dimension.ACCURACY

    def test_summary_for_returns_none_for_missing_dimension(self) -> None:
        report = EvalReport.from_results([])
        assert report.summary_for(Dimension.COST) is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestDimensionAggregation:
    def test_aggregates_multiple_dimensions(self) -> None:
        results = [
            EvalResult(
                case_id="c1", run_index=0, agent_output="ok",
                dimension_scores=[
                    _make_score(Dimension.ACCURACY, 1.0, True),
                    _make_score(Dimension.LATENCY, 0.8, True),
                ],
            ),
            EvalResult(
                case_id="c2", run_index=0, agent_output="ok",
                dimension_scores=[
                    _make_score(Dimension.ACCURACY, 0.6, False),
                    _make_score(Dimension.LATENCY, 0.9, True),
                ],
            ),
        ]
        report = EvalReport.from_results(results)
        assert len(report.dimension_summaries) == 2

        acc = report.summary_for(Dimension.ACCURACY)
        assert acc is not None
        assert acc.mean_score == pytest.approx(0.8)
        assert acc.min_score == pytest.approx(0.6)
        assert acc.max_score == pytest.approx(1.0)
        assert acc.pass_rate == pytest.approx(0.5)
        assert acc.total_cases == 2

    def test_stddev_zero_for_single_score(self) -> None:
        results = [
            EvalResult(
                case_id="c1", run_index=0, agent_output="ok",
                dimension_scores=[_make_score(Dimension.ACCURACY, 0.8, True)],
            )
        ]
        report = EvalReport.from_results(results)
        acc = report.summary_for(Dimension.ACCURACY)
        assert acc is not None
        assert acc.score_stddev == 0.0

    def test_stddev_nonzero_for_multiple_scores(self) -> None:
        results = [
            EvalResult(
                case_id="c1", run_index=0, agent_output="ok",
                dimension_scores=[_make_score(Dimension.ACCURACY, 1.0, True)],
            ),
            EvalResult(
                case_id="c2", run_index=0, agent_output="ok",
                dimension_scores=[_make_score(Dimension.ACCURACY, 0.0, False)],
            ),
        ]
        report = EvalReport.from_results(results)
        acc = report.summary_for(Dimension.ACCURACY)
        assert acc is not None
        assert acc.score_stddev > 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestEvalReportSerialization:
    def _make_report(self) -> EvalReport:
        results = [
            EvalResult(
                case_id="c1", run_index=0, agent_output="good",
                dimension_scores=[_make_score(Dimension.ACCURACY, 0.9, True)],
                latency_ms=150.0,
            )
        ]
        return EvalReport.from_results(results, suite_name="my-suite", agent_name="my-agent")

    def test_to_dict_has_required_keys(self) -> None:
        report = self._make_report()
        data = report.to_dict()
        required = ["suite_name", "agent_name", "created_at", "total_cases",
                    "passed_cases", "failed_cases", "error_cases",
                    "overall_pass_rate", "mean_latency_ms", "dimension_summaries", "results"]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_to_dict_suite_name(self) -> None:
        report = self._make_report()
        assert report.to_dict()["suite_name"] == "my-suite"

    def test_to_json_is_valid_json(self) -> None:
        report = self._make_report()
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_results_list(self) -> None:
        report = self._make_report()
        data = json.loads(report.to_json())
        assert len(data["results"]) == 1
        assert data["results"][0]["case_id"] == "c1"

    def test_to_markdown_returns_string_with_headers(self) -> None:
        from unittest.mock import MagicMock, patch

        report = self._make_report()
        fake_md = "# Evaluation Report: my-suite\n\n**Agent:** my-agent\n"
        mock_formatter_instance = MagicMock()
        mock_formatter_instance.render.return_value = fake_md
        mock_reporter_class = MagicMock(return_value=mock_formatter_instance)

        mock_module = MagicMock()
        mock_module.MarkdownReporter = mock_reporter_class
        with patch.dict("sys.modules", {"agent_eval.reporting.markdown_report": mock_module}):
            md = report.to_markdown()

        assert isinstance(md, str)
        assert "#" in md
        mock_reporter_class.assert_called_once()
        mock_formatter_instance.render.assert_called_once_with(report)

    def test_to_html_returns_string_with_html_tag(self) -> None:
        from unittest.mock import MagicMock, patch

        report = self._make_report()
        fake_html = "<!DOCTYPE html>\n<html lang=\"en\"><body>Eval Report</body></html>"
        mock_formatter_instance = MagicMock()
        mock_formatter_instance.render.return_value = fake_html
        mock_reporter_class = MagicMock(return_value=mock_formatter_instance)

        mock_module = MagicMock()
        mock_module.HtmlReporter = mock_reporter_class
        with patch.dict("sys.modules", {"agent_eval.reporting.html_report": mock_module}):
            html = report.to_html()

        assert "<!DOCTYPE html>" in html or "<html" in html
        mock_reporter_class.assert_called_once()
        mock_formatter_instance.render.assert_called_once_with(report)

    def test_summary_method_returns_string(self) -> None:
        report = self._make_report()
        summary = report.summary()
        assert "my-suite" in summary
        assert "my-agent" in summary
