"""pytest-agent-eval — evaluation-driven development for AI agents.

This package provides a pytest plugin that integrates agent evaluation
into the standard test workflow. Import :class:`EvalContext` directly
from this package for type annotations in test files.

Public API
----------
:class:`EvalContext`
    Per-test evaluation accumulator. Injected via the ``eval_context``
    fixture.
:class:`~agent_eval.pytest_plugin.similarity.SimilarityScorer`
    Multi-strategy text similarity scorer used by ``assert_accuracy``.
:class:`~agent_eval.pytest_plugin.baseline.BaselineStore`
    Persist and compare evaluation scores across test sessions.
:class:`~agent_eval.pytest_plugin.report.EvalReport`
    Aggregated session report, rendered as JSON or Markdown.
:class:`~agent_eval.pytest_plugin.multi_run.MultiRunEvaluator`
    Record scores across multiple runs and compute consistency metrics.
:class:`~agent_eval.pytest_plugin.scaffold.EvalScaffoldGenerator`
    Generate pytest evaluation test files from BSL specs or dict specs.

Example
-------
::

    from agent_eval.pytest_plugin import EvalContext


    def test_agent_accuracy(eval_context: EvalContext) -> None:
        response = my_agent("What is the capital of France?")
        eval_context.assert_accuracy(response, expected_intent="Paris")
        eval_context.assert_safety(response)
        assert eval_context.all_passed
"""
from __future__ import annotations

from agent_eval.pytest_plugin.baseline import BaselineStore
from agent_eval.pytest_plugin.context import EvalContext
from agent_eval.pytest_plugin.multi_run import MultiRunEvaluator
from agent_eval.pytest_plugin.report import EvalReport
from agent_eval.pytest_plugin.scaffold import EvalScaffoldGenerator
from agent_eval.pytest_plugin.similarity import SimilarityScorer

__all__ = [
    "BaselineStore",
    "EvalContext",
    "EvalReport",
    "EvalScaffoldGenerator",
    "MultiRunEvaluator",
    "SimilarityScorer",
]
