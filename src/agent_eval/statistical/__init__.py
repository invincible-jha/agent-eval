"""Statistical reliability scoring for agent evaluations.

This subpackage provides pass@k metrics, Wilson score confidence intervals,
and multi-run statistical reporting. Use it to measure the reliability of
stochastic agents where a single evaluation run is insufficient.

Exports
-------
StatisticalRunner
    Runs an evaluation callable N times and collects raw results.
StatisticalResult
    Aggregated outcome of N evaluation runs.
StatisticalReport
    Human-readable report with pass@k, confidence intervals, and variance.
pass_at_k
    Standalone function: compute the pass@k scalar for known n/c values.
confidence_interval
    Standalone function: compute a Wilson score CI for a proportion.

Example
-------
::

    from agent_eval.statistical import StatisticalRunner, StatisticalReport

    runner = StatisticalRunner(n_runs=10)
    result = runner.run(my_eval_fn)
    report = StatisticalReport.from_result(result, label="my_test")
    print(report.to_text())
"""
from __future__ import annotations

from agent_eval.statistical.metrics import (
    ConfidenceInterval,
    PassAtKResult,
    confidence_interval,
    pass_at_k,
    pass_at_k_result,
)
from agent_eval.statistical.report import StatisticalReport
from agent_eval.statistical.runner import RunRecord, StatisticalResult, StatisticalRunner

__all__ = [
    "StatisticalRunner",
    "StatisticalResult",
    "RunRecord",
    "StatisticalReport",
    "pass_at_k",
    "pass_at_k_result",
    "PassAtKResult",
    "confidence_interval",
    "ConfidenceInterval",
]
