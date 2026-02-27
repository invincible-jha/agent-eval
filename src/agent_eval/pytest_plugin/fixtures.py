"""pytest fixtures for pytest-agent-eval.

Provides the ``eval_context`` function-scoped fixture that injects a fresh
:class:`~agent_eval.pytest_plugin.context.EvalContext` into each test.

The fixture is registered automatically when the plugin is active (i.e.
when ``agent-eval`` is installed and the ``pytest11`` entry-point is
declared in ``pyproject.toml``).

Example
-------
::

    import pytest
    from agent_eval.pytest_plugin import EvalContext


    def test_my_agent(eval_context: EvalContext) -> None:
        result = my_agent.run("What is 2 + 2?")
        eval_context.assert_accuracy(result, expected_intent="4")
        eval_context.assert_safety(result)
        assert eval_context.all_passed
"""
from __future__ import annotations

import pytest

from agent_eval.pytest_plugin.context import EvalContext


@pytest.fixture()
def eval_context() -> EvalContext:
    """Provide a fresh :class:`EvalContext` for each test.

    The context is function-scoped so each test receives an independent
    instance with no carry-over state from other tests.

    Returns
    -------
    EvalContext
        An empty evaluation context ready to accumulate assertions.
    """
    return EvalContext()
