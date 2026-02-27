"""Tests for agent_eval.cascade.report.

Verifies CascadeReport creation from CascadeAnalysis and output methods.
"""
from __future__ import annotations

import json

from agent_eval.cascade.analyzer import CascadeAnalyzer, StepResult
from agent_eval.cascade.dependency_graph import DependencyGraph
from agent_eval.cascade.report import CascadeReport


def build_linear_chain_analysis() -> object:
    """Build a CascadeAnalysis for a linear 5-step chain with step_2 as root cause."""
    graph = DependencyGraph()
    graph.add_step("step_1")
    graph.add_step("step_2", depends_on=["step_1"])
    graph.add_step("step_3", depends_on=["step_2"])
    graph.add_step("step_4", depends_on=["step_3"])

    results = {
        "step_1": StepResult("step_1", passed=True),
        "step_2": StepResult("step_2", passed=False),
        "step_3": StepResult("step_3", passed=False),
        "step_4": StepResult("step_4", passed=False),
    }
    analyzer = CascadeAnalyzer()
    return analyzer.analyze(graph, results)


class TestCascadeReportFromAnalysis:
    """Tests for CascadeReport.from_analysis factory."""

    def test_report_has_correct_root_cause_count(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis, label="linear_chain")  # type: ignore[arg-type]
        assert report.n_root_causes == 1

    def test_report_has_correct_cascade_count(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        assert report.n_cascade == 2

    def test_report_label_is_set(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis, label="test_label")  # type: ignore[arg-type]
        assert report.label == "test_label"

    def test_cascade_chains_populated(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        assert len(report.cascade_chains) >= 1

    def test_cascade_chains_mention_root_cause(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        chain_text = " ".join(report.cascade_chains)
        assert "step_2" in chain_text

    def test_root_cause_ids_correct(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        assert "step_2" in report.root_cause_ids

    def test_cascade_ids_correct(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        assert "step_3" in report.cascade_ids
        assert "step_4" in report.cascade_ids


class TestCascadeReportToDict:
    """Tests for to_dict() serialization."""

    def setup_method(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis, label="dict_test")  # type: ignore[arg-type]
        self.data = report.to_dict()

    def test_top_level_keys_present(self) -> None:
        required_keys = {"label", "created_at", "summary", "root_cause_ids", "cascade_ids"}
        assert required_keys.issubset(set(self.data.keys()))

    def test_summary_has_correct_structure(self) -> None:
        summary = self.data["summary"]
        assert "total_steps" in summary
        assert "root_causes" in summary
        assert "cascade_failures" in summary

    def test_label_in_dict(self) -> None:
        assert self.data["label"] == "dict_test"


class TestCascadeReportToJson:
    """Tests for to_json() serialization."""

    def test_to_json_is_valid_json(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_contains_root_cause_ids(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        data = json.loads(report.to_json())
        assert "step_2" in data["root_cause_ids"]


class TestCascadeReportToText:
    """Tests for to_text() human-readable output."""

    def test_to_text_is_string(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        assert isinstance(report.to_text(), str)

    def test_to_text_contains_root_causes(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        text = report.to_text()
        assert "Root causes" in text or "root cause" in text.lower()

    def test_to_text_contains_cascade_info(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis)  # type: ignore[arg-type]
        text = report.to_text()
        assert "cascade" in text.lower() or "Cascade" in text

    def test_to_text_mentions_label(self) -> None:
        analysis = build_linear_chain_analysis()
        report = CascadeReport.from_analysis(analysis, label="my_pipeline")  # type: ignore[arg-type]
        text = report.to_text()
        assert "my_pipeline" in text
