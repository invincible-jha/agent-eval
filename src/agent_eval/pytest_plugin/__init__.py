"""pytest-agent-eval — evaluation-driven development for AI agents.

This package provides a pytest plugin that integrates agent evaluation
into the standard test workflow. Import :class:`EvalContext` directly
from this package for type annotations in test files.

Public API
----------
:class:`EvalContext`
    Per-test evaluation accumulator. Injected via the ``eval_context``
    fixture.

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

from agent_eval.pytest_plugin.context import EvalContext

__all__ = ["EvalContext"]
