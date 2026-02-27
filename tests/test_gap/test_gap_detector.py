"""Tests for agent_eval.gap.gap_detector."""
from __future__ import annotations

import pytest

from agent_eval.gap.gap_detector import (
    DistributionSample,
    FeatureGap,
    GapDetector,
    GapReport,
    GapSeverity,
    _ks_statistic,
    _severity_from_ks,
)
from agent_eval.gap.trace_loader import ProductionTrace


def _make_traces(
    input_lengths: list[int],
    output_lengths: list[int] | None = None,
    tool_call_counts: list[int] | None = None,
) -> list[ProductionTrace]:
    traces = []
    for i, inp_len in enumerate(input_lengths):
        out_len = (output_lengths or [])[i] if output_lengths else 50
        tc_count = (tool_call_counts or [])[i] if tool_call_counts else 0
        traces.append(
            ProductionTrace(
                trace_id=f"t{i}",
                input_text="x" * inp_len,
                output_text="y" * out_len,
                tool_calls=[{"n": "t"} for _ in range(tc_count)],
            )
        )
    return traces


class TestKsStatistic:
    def test_identical_distributions_ks_zero(self) -> None:
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        ks = _ks_statistic(sample, sample)
        assert ks < 0.01

    def test_completely_different_distributions_ks_high(self) -> None:
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [100.0, 200.0, 300.0, 400.0, 500.0]
        ks = _ks_statistic(sample_a, sample_b)
        assert ks > 0.5

    def test_empty_samples_return_zero(self) -> None:
        ks = _ks_statistic([], [1.0, 2.0])
        assert ks == 0.0

    def test_ks_bounded_0_to_1(self) -> None:
        a = list(range(20))
        b = [float(x * 10) for x in range(20)]
        ks = _ks_statistic([float(v) for v in a], b)
        assert 0.0 <= ks <= 1.0


class TestSeverityFromKs:
    def test_negligible(self) -> None:
        assert _severity_from_ks(0.05) == GapSeverity.NEGLIGIBLE

    def test_low(self) -> None:
        assert _severity_from_ks(0.15) == GapSeverity.LOW

    def test_medium(self) -> None:
        assert _severity_from_ks(0.25) == GapSeverity.MEDIUM

    def test_high(self) -> None:
        assert _severity_from_ks(0.40) == GapSeverity.HIGH

    def test_critical(self) -> None:
        assert _severity_from_ks(0.60) == GapSeverity.CRITICAL


class TestDistributionSample:
    def test_from_list_creates_sample(self) -> None:
        sample = DistributionSample.from_list("feat", [1.0, 2.0, 3.0], "synthetic")
        assert sample.count == 3
        assert sample.source_label == "synthetic"

    def test_mean_computed(self) -> None:
        sample = DistributionSample.from_list("feat", [1.0, 2.0, 3.0])
        assert abs(sample.mean - 2.0) < 0.01

    def test_std_computed(self) -> None:
        sample = DistributionSample.from_list("feat", [1.0, 2.0, 3.0])
        assert sample.std > 0

    def test_empty_sample_mean_zero(self) -> None:
        sample = DistributionSample.from_list("feat", [])
        assert sample.mean == 0.0

    def test_frozen(self) -> None:
        sample = DistributionSample.from_list("feat", [1.0])
        with pytest.raises((AttributeError, TypeError)):
            sample.name = "other"  # type: ignore[misc]


class TestGapDetector:
    def test_detect_returns_gap_report(self) -> None:
        detector = GapDetector()
        synthetic = {
            "input_length": DistributionSample.from_list(
                "input_length", [50.0] * 20, "synthetic"
            )
        }
        traces = _make_traces([50] * 20)
        report = detector.detect(synthetic, traces)
        assert isinstance(report, GapReport)

    def test_identical_distributions_negligible_gap(self) -> None:
        detector = GapDetector(features=["input_length"])
        values = [float(i * 10) for i in range(20)]
        synthetic = {
            "input_length": DistributionSample.from_list("input_length", values)
        }
        traces = _make_traces([int(v) for v in values])
        report = detector.detect(synthetic, traces)
        assert report.overall_severity in (GapSeverity.NEGLIGIBLE, GapSeverity.LOW)

    def test_divergent_distributions_detect_gap(self) -> None:
        detector = GapDetector(features=["input_length"])
        # Synthetic: short inputs (10-30); production: long inputs (200-500)
        synthetic = {
            "input_length": DistributionSample.from_list(
                "input_length", [float(i) for i in range(10, 30)]
            )
        }
        traces = _make_traces([i * 10 for i in range(20, 40)])
        report = detector.detect(synthetic, traces)
        assert report.overall_severity not in (GapSeverity.NEGLIGIBLE,)

    def test_insufficient_samples_returns_negligible(self) -> None:
        detector = GapDetector(features=["input_length"], min_samples_for_comparison=10)
        synthetic = {
            "input_length": DistributionSample.from_list("input_length", [50.0, 60.0])
        }
        traces = _make_traces([50, 60])
        report = detector.detect(synthetic, traces)
        assert report.feature_gaps[0].severity == GapSeverity.NEGLIGIBLE

    def test_critical_features_property(self) -> None:
        report = GapReport(
            feature_gaps=[
                FeatureGap(
                    feature_name="f",
                    ks_statistic=0.8,
                    severity=GapSeverity.CRITICAL,
                    synthetic_mean=10.0,
                    production_mean=500.0,
                    synthetic_count=20,
                    production_count=20,
                )
            ]
        )
        assert len(report.critical_features) == 1

    def test_has_significant_gaps_true(self) -> None:
        report = GapReport(
            feature_gaps=[
                FeatureGap(
                    feature_name="f",
                    ks_statistic=0.3,
                    severity=GapSeverity.MEDIUM,
                    synthetic_mean=50.0,
                    production_mean=200.0,
                    synthetic_count=20,
                    production_count=20,
                )
            ]
        )
        assert report.has_significant_gaps is True

    def test_from_synthetic_eval_data(self) -> None:
        eval_data = [
            {"input": "x" * 50, "output": "y" * 30, "tool_calls": []}
            for _ in range(15)
        ]
        detector, samples = GapDetector.from_synthetic_eval_data(eval_data)
        assert "input_length" in samples
        assert "output_length" in samples

    def test_feature_not_in_synthetic_skipped(self) -> None:
        detector = GapDetector(features=["input_length", "latency_ms"])
        synthetic = {
            "input_length": DistributionSample.from_list(
                "input_length", [float(i) for i in range(20)]
            )
        }
        traces = _make_traces([i for i in range(20)])
        report = detector.detect(synthetic, traces)
        # Only input_length was in synthetic_samples; latency_ms was skipped
        feature_names = [g.feature_name for g in report.feature_gaps]
        assert "input_length" in feature_names
        assert "latency_ms" not in feature_names
