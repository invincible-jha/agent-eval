"""Configuration models for agent-eval.

EvalConfig is the root Pydantic model loaded from eval.yaml.
All runtime settings flow through this model so they can be
validated at startup rather than failing mid-run.

NOTE: Pydantic v2 is required. The model uses model_config
for strict validation at system boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from agent_eval.core.exceptions import ConfigError


class EvaluatorConfig(BaseModel):
    """Configuration for a single evaluator.

    Parameters
    ----------
    name:
        Evaluator identifier. Used to look up in the registry.
    type:
        Evaluator type string. Matches registered evaluator names.
    enabled:
        Whether this evaluator is active. Defaults to True.
    settings:
        Evaluator-specific key-value settings passed as kwargs.
    """

    name: str
    type: str
    enabled: bool = True
    settings: dict[str, str | int | float | bool] = Field(default_factory=dict)


class GateConfig(BaseModel):
    """Configuration for a deployment gate.

    Parameters
    ----------
    name:
        Gate identifier.
    type:
        Gate type: "threshold" or "composite".
    thresholds:
        Per-dimension minimum score thresholds (0.0-1.0).
    mode:
        "all" requires all dimensions to pass; "any" requires at least one.
    enabled:
        Whether this gate is active.
    """

    name: str
    type: Literal["threshold", "composite"] = "threshold"
    thresholds: dict[str, float] = Field(default_factory=dict)
    mode: Literal["all", "any"] = "all"
    enabled: bool = True

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        for dimension, threshold in value.items():
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"Threshold for {dimension!r} must be in [0.0, 1.0], got {threshold}"
                )
        return value


class ReportingConfig(BaseModel):
    """Configuration for report generation.

    Parameters
    ----------
    formats:
        List of output formats to generate: "json", "html", "markdown", "console".
    output_dir:
        Directory where report files are written. Defaults to "./reports".
    include_raw_outputs:
        Whether to include full agent outputs in reports. Defaults to True.
    """

    formats: list[Literal["json", "html", "markdown", "console"]] = Field(
        default_factory=lambda: ["console"]
    )
    output_dir: str = "./reports"
    include_raw_outputs: bool = True


class RunnerConfig(BaseModel):
    """Configuration for the evaluation runner.

    Parameters
    ----------
    runs_per_case:
        How many times each test case is run. Useful for measuring
        variance in non-deterministic agents. NOT statistical rigor.
    timeout_ms:
        Default per-case timeout in milliseconds.
    max_retries:
        Number of retry attempts on transient agent failures.
    concurrency:
        Maximum number of concurrent agent invocations. 1 = sequential.
    fail_fast:
        If True, abort the run after the first case failure.
    """

    runs_per_case: int = Field(default=1, ge=1, le=100)
    timeout_ms: int = Field(default=30_000, ge=100)
    max_retries: int = Field(default=0, ge=0, le=5)
    concurrency: int = Field(default=1, ge=1, le=50)
    fail_fast: bool = False


class EvalConfig(BaseModel):
    """Root configuration model for an agent-eval run.

    Loaded from eval.yaml. All sections are optional and have
    sensible defaults.

    Parameters
    ----------
    runner:
        Runner execution settings.
    evaluators:
        List of evaluator configurations.
    gates:
        List of deployment gate configurations.
    reporting:
        Report generation settings.
    suite:
        Path to the test suite YAML/JSON, or a built-in suite name.
    agent:
        Optional agent configuration (used by CLI run command).
    """

    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    evaluators: list[EvaluatorConfig] = Field(default_factory=list)
    gates: list[GateConfig] = Field(default_factory=list)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    suite: str | None = None
    agent: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_at_least_one_evaluator_or_gate(self) -> "EvalConfig":
        """Warn but do not fail if no evaluators are configured.

        An empty evaluator list is valid (e.g., gate-only mode).
        """
        return self

    @classmethod
    def from_yaml(cls, path: str | Path = "eval.yaml") -> "EvalConfig":
        """Load EvalConfig from a YAML file.

        Parameters
        ----------
        path:
            Path to the YAML configuration file.

        Returns
        -------
        EvalConfig

        Raises
        ------
        ConfigError
            If the file cannot be read or the config is invalid.
        """
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot read {path}: {exc}") from exc

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top-level must be a YAML mapping")

        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise ConfigError(str(exc)) from exc

    def active_evaluators(self) -> list[EvaluatorConfig]:
        """Return only enabled evaluator configs."""
        return [e for e in self.evaluators if e.enabled]

    def active_gates(self) -> list[GateConfig]:
        """Return only enabled gate configs."""
        return [g for g in self.gates if g.enabled]
