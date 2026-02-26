"""agent-eval — Framework for evaluating AI agents across multiple quality dimensions.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Example
-------
>>> import agent_eval
>>> agent_eval.__version__
'0.1.0'
"""
from __future__ import annotations

__version__: str = "0.1.0"

# Core abstractions
from agent_eval.core.evaluator import (
    Dimension,
    DimensionScore,
    EvalResult,
    Evaluator,
)
from agent_eval.core.gate import DeploymentGate, GateResult
from agent_eval.core.runner import EvalRunner, RunnerOptions
from agent_eval.core.suite import BenchmarkSuite, TestCase
from agent_eval.core.agent_wrapper import AgentUnderTest
from agent_eval.core.report import EvalReport

# Adapters
from agent_eval.adapters.callable import CallableAdapter
from agent_eval.adapters.http import HTTPAdapter

# Gates
from agent_eval.gates.threshold import BasicThresholdGate
from agent_eval.gates.composite import CompositeGate

# Suites
from agent_eval.suites.loader import SuiteLoader
from agent_eval.suites.builder import SuiteBuilder

__all__ = [
    "__version__",
    # Core
    "Dimension",
    "DimensionScore",
    "EvalResult",
    "Evaluator",
    "EvalRunner",
    "RunnerOptions",
    "EvalReport",
    "BenchmarkSuite",
    "TestCase",
    "AgentUnderTest",
    # Gates
    "DeploymentGate",
    "GateResult",
    "BasicThresholdGate",
    "CompositeGate",
    # Adapters
    "CallableAdapter",
    "HTTPAdapter",
    # Suites
    "SuiteLoader",
    "SuiteBuilder",
]
