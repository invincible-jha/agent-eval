"""Core domain objects for agent-eval.

This package contains the fundamental abstractions:
- Evaluator ABC and Dimension taxonomy
- BenchmarkSuite and TestCase definitions
- EvalRunner orchestration
- EvalReport aggregation
- DeploymentGate ABC
- AgentUnderTest wrapper
- EvalConfig Pydantic model
- Custom exceptions
"""
from __future__ import annotations

from agent_eval.core.agent_wrapper import AgentUnderTest
from agent_eval.core.config import EvalConfig, EvaluatorConfig, GateConfig, RunnerConfig
from agent_eval.core.evaluator import Dimension, DimensionScore, EvalResult, Evaluator
from agent_eval.core.exceptions import (
    AgentEvalError,
    AgentTimeoutError,
    ConfigError,
    EvaluatorError,
    GateError,
    RunnerError,
    SuiteError,
)
from agent_eval.core.gate import DeploymentGate, GateResult
from agent_eval.core.report import DimensionSummary, EvalReport
from agent_eval.core.runner import EvalRunner, RunnerOptions
from agent_eval.core.suite import BenchmarkSuite, TestCase

__all__ = [
    # Evaluator types
    "Evaluator",
    "Dimension",
    "DimensionScore",
    "EvalResult",
    # Suite types
    "BenchmarkSuite",
    "TestCase",
    # Runner
    "EvalRunner",
    "RunnerOptions",
    # Report
    "EvalReport",
    "DimensionSummary",
    # Gate
    "DeploymentGate",
    "GateResult",
    # Agent wrapper
    "AgentUnderTest",
    # Config
    "EvalConfig",
    "EvaluatorConfig",
    "GateConfig",
    "RunnerConfig",
    # Exceptions
    "AgentEvalError",
    "AgentTimeoutError",
    "ConfigError",
    "EvaluatorError",
    "GateError",
    "RunnerError",
    "SuiteError",
]
