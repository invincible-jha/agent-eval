"""Unit tests for agent_eval.core.config.

Tests EvaluatorConfig, GateConfig, ReportingConfig, RunnerConfig,
and EvalConfig including from_yaml() and validation logic.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_eval.core.config import (
    EvalConfig,
    EvaluatorConfig,
    GateConfig,
    ReportingConfig,
    RunnerConfig,
)
from agent_eval.core.exceptions import ConfigError


# ---------------------------------------------------------------------------
# EvaluatorConfig
# ---------------------------------------------------------------------------


class TestEvaluatorConfig:
    def test_minimal_valid(self) -> None:
        cfg = EvaluatorConfig(name="acc", type="accuracy")
        assert cfg.name == "acc"
        assert cfg.type == "accuracy"
        assert cfg.enabled is True
        assert cfg.settings == {}

    def test_disabled_evaluator(self) -> None:
        cfg = EvaluatorConfig(name="acc", type="accuracy", enabled=False)
        assert cfg.enabled is False

    def test_settings_stored(self) -> None:
        cfg = EvaluatorConfig(name="lat", type="latency", settings={"max_ms": 5000})
        assert cfg.settings["max_ms"] == 5000


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------


class TestGateConfig:
    def test_minimal_valid(self) -> None:
        cfg = GateConfig(name="gate1", type="threshold")
        assert cfg.name == "gate1"
        assert cfg.mode == "all"

    def test_valid_thresholds(self) -> None:
        cfg = GateConfig(
            name="g", type="threshold",
            thresholds={"accuracy": 0.8, "safety": 1.0}
        )
        assert cfg.thresholds["accuracy"] == 0.8

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
    def test_threshold_out_of_range_raises(self, threshold: float) -> None:
        with pytest.raises(ValidationError):
            GateConfig(name="g", type="threshold", thresholds={"accuracy": threshold})

    def test_mode_any(self) -> None:
        cfg = GateConfig(name="g", type="threshold", mode="any")
        assert cfg.mode == "any"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            GateConfig(name="g", type="invalid_type")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RunnerConfig
# ---------------------------------------------------------------------------


class TestRunnerConfig:
    def test_defaults(self) -> None:
        cfg = RunnerConfig()
        assert cfg.runs_per_case == 1
        assert cfg.timeout_ms == 30_000
        assert cfg.max_retries == 0
        assert cfg.concurrency == 1
        assert cfg.fail_fast is False

    def test_custom_values(self) -> None:
        cfg = RunnerConfig(runs_per_case=3, timeout_ms=10_000, concurrency=4, fail_fast=True)
        assert cfg.runs_per_case == 3
        assert cfg.concurrency == 4
        assert cfg.fail_fast is True

    def test_runs_per_case_min_bound(self) -> None:
        with pytest.raises(ValidationError):
            RunnerConfig(runs_per_case=0)

    def test_runs_per_case_max_bound(self) -> None:
        with pytest.raises(ValidationError):
            RunnerConfig(runs_per_case=101)

    def test_concurrency_max_bound(self) -> None:
        with pytest.raises(ValidationError):
            RunnerConfig(concurrency=51)

    def test_max_retries_max_bound(self) -> None:
        with pytest.raises(ValidationError):
            RunnerConfig(max_retries=6)

    def test_timeout_min_bound(self) -> None:
        with pytest.raises(ValidationError):
            RunnerConfig(timeout_ms=99)


# ---------------------------------------------------------------------------
# ReportingConfig
# ---------------------------------------------------------------------------


class TestReportingConfig:
    def test_defaults(self) -> None:
        cfg = ReportingConfig()
        assert "console" in cfg.formats
        assert cfg.output_dir == "./reports"
        assert cfg.include_raw_outputs is True

    def test_custom_formats(self) -> None:
        cfg = ReportingConfig(formats=["json", "html"])
        assert "json" in cfg.formats
        assert "html" in cfg.formats


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------


class TestEvalConfig:
    def test_empty_config_uses_defaults(self) -> None:
        cfg = EvalConfig()
        assert isinstance(cfg.runner, RunnerConfig)
        assert cfg.evaluators == []
        assert cfg.gates == []

    def test_active_evaluators_filters_disabled(self) -> None:
        cfg = EvalConfig(
            evaluators=[
                EvaluatorConfig(name="a", type="accuracy", enabled=True),
                EvaluatorConfig(name="b", type="latency", enabled=False),
            ]
        )
        active = cfg.active_evaluators()
        assert len(active) == 1
        assert active[0].name == "a"

    def test_active_gates_filters_disabled(self) -> None:
        cfg = EvalConfig(
            gates=[
                GateConfig(name="g1", type="threshold", enabled=True),
                GateConfig(name="g2", type="threshold", enabled=False),
            ]
        )
        active = cfg.active_gates()
        assert len(active) == 1
        assert active[0].name == "g1"

    def test_from_yaml_loads_valid_file(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            runner:
              runs_per_case: 2
              timeout_ms: 10000
            evaluators:
              - name: acc
                type: accuracy
                enabled: true
        """)
        config_file = tmp_path / "eval.yaml"
        config_file.write_text(content, encoding="utf-8")
        cfg = EvalConfig.from_yaml(config_file)
        assert cfg.runner.runs_per_case == 2
        assert len(cfg.evaluators) == 1

    def test_from_yaml_missing_file_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            EvalConfig.from_yaml("/no/such/eval.yaml")

    def test_from_yaml_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{bad: yaml: content:", encoding="utf-8")
        with pytest.raises(ConfigError):
            EvalConfig.from_yaml(config_file)

    def test_from_yaml_empty_file_uses_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")
        cfg = EvalConfig.from_yaml(config_file)
        assert cfg.runner.runs_per_case == 1

    def test_from_yaml_non_mapping_raises_config_error(self, tmp_path: Path) -> None:
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            EvalConfig.from_yaml(config_file)

    def test_suite_field_accepts_string(self) -> None:
        cfg = EvalConfig(suite="qa_basic")
        assert cfg.suite == "qa_basic"
