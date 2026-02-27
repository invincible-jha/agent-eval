"""Convenience API for agent-eval — 3-line quickstart.

Example
-------
::

    from agent_eval import evaluate
    report = evaluate(my_agent_fn, tasks=["What is 2+2?", "Explain AI."])
    print(report.summary())

"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Sequence


def evaluate(
    agent_fn: Callable[[str], str],
    tasks: Sequence[str],
    n: int = 5,
    agent_name: str = "agent",
) -> Any:
    """Evaluate an agent function against a list of task prompts.

    Zero-config entry point for the 80% use case. Wraps the full
    EvalRunner / BenchmarkSuite / EvalReport machinery with sensible
    defaults (accuracy + latency evaluators, sequential execution).

    Parameters
    ----------
    agent_fn:
        A callable that accepts a string prompt and returns a string response.
    tasks:
        Sequence of string prompts to evaluate the agent against.
    n:
        Number of runs per task for variance observation (default 5).
    agent_name:
        Human-readable name for the agent under test.

    Returns
    -------
    EvalReport
        Aggregated evaluation results with per-dimension statistics.

    Example
    -------
    ::

        from agent_eval import evaluate

        def my_agent(prompt: str) -> str:
            return f"Response to: {prompt}"

        report = evaluate(my_agent, tasks=["Hello?", "What is AI?"])
        print(report.passed)
    """
    from agent_eval.adapters.callable import CallableAdapter
    from agent_eval.core.agent_wrapper import AgentUnderTest
    from agent_eval.core.runner import EvalRunner, RunnerOptions
    from agent_eval.core.suite import BenchmarkSuite, TestCase
    from agent_eval.evaluators.accuracy import BasicAccuracyEvaluator
    from agent_eval.evaluators.latency import BasicLatencyEvaluator

    test_cases = [
        TestCase(id=f"task-{i}", input=task, expected_output="")
        for i, task in enumerate(tasks)
    ]
    suite = BenchmarkSuite(name="quickstart-suite", cases=test_cases)

    agent = AgentUnderTest(callable_fn=agent_fn, name=agent_name)

    runner = EvalRunner(
        evaluators=[
            BasicAccuracyEvaluator(),
            BasicLatencyEvaluator(max_ms=30000),
        ],
        options=RunnerOptions(runs_per_case=n, concurrency=1),
    )

    return asyncio.run(runner.run(agent, suite))
