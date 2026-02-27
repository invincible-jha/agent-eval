"""Multi-step cascade failure analysis for agent evaluation pipelines.

This subpackage provides tools to analyze multi-step evaluation failures and
distinguish root causes from cascade effects. When step 2 fails and causes
steps 3, 4, and 5 to also fail, the framework reports "1 root cause with
3 cascade failures" rather than "4/5 steps failed".

Exports
-------
DependencyGraph
    DAG model of step dependencies for a multi-step evaluation.
CascadeAnalyzer
    Classifies failures as ROOT_CAUSE, CASCADE, or INDEPENDENT.
CascadeAnalysis
    Raw analysis result with complete classification data.
CascadeReport
    Human-readable report with cascade chains and actual reliability.
StepResult
    Result of a single evaluation step.
FailureClassification
    Enum: ROOT_CAUSE, CASCADE, INDEPENDENT.
FailedStepAnalysis
    Detailed analysis of a single failed step.

Example
-------
::

    from agent_eval.cascade import (
        DependencyGraph, CascadeAnalyzer, CascadeReport, StepResult
    )

    graph = DependencyGraph()
    graph.add_step("auth")
    graph.add_step("fetch_data", depends_on=["auth"])
    graph.add_step("process", depends_on=["fetch_data"])
    graph.add_step("output", depends_on=["process"])

    results = {
        "auth": StepResult("auth", passed=False),
        "fetch_data": StepResult("fetch_data", passed=False),
        "process": StepResult("process", passed=False),
        "output": StepResult("output", passed=True),
    }

    analyzer = CascadeAnalyzer()
    analysis = analyzer.analyze(graph, results)
    report = CascadeReport.from_analysis(analysis, label="auth_pipeline")
    print(report.to_text())
    # Root causes: ['auth']
    # Cascades: ['fetch_data', 'process']
"""
from __future__ import annotations

from agent_eval.cascade.analyzer import (
    CascadeAnalysis,
    CascadeAnalyzer,
    FailedStepAnalysis,
    FailureClassification,
    StepResult,
)
from agent_eval.cascade.dependency_graph import CycleError, DependencyGraph
from agent_eval.cascade.report import CascadeReport

__all__ = [
    "DependencyGraph",
    "CycleError",
    "CascadeAnalyzer",
    "CascadeAnalysis",
    "CascadeReport",
    "StepResult",
    "FailureClassification",
    "FailedStepAnalysis",
]
