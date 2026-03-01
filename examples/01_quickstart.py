#!/usr/bin/env python3
"""Example: Quickstart

Demonstrates the minimal working example for evaluating an agent
using agent-eval with the convenience API.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-eval
"""
from __future__ import annotations

import agent_eval
from agent_eval import (
    evaluate,
    BenchmarkSuite,
    TestCase,
    Dimension,
    CallableAdapter,
)


def my_agent(user_input: str) -> str:
    """A simple stub agent that echoes a canned response."""
    if "capital" in user_input.lower():
        return "The capital of France is Paris."
    return f"I don't know the answer to: {user_input}"


def main() -> None:
    print(f"agent-eval version: {agent_eval.__version__}")

    # Step 1: Define a benchmark suite with test cases
    suite = (
        BenchmarkSuite(name="quickstart-suite", description="Basic QA tests")
        .add_case(
            TestCase(
                id="qs-01",
                input="What is the capital of France?",
                expected_output="Paris",
                dimensions=[Dimension.ACCURACY],
            )
        )
    )

    # Step 2: Wrap the agent with a callable adapter
    adapter = CallableAdapter(fn=my_agent, name="my-stub-agent")

    # Step 3: Run evaluation using the convenience function
    try:
        report = evaluate(adapter=adapter, suite=suite)
        print(f"\nEvaluation complete — {len(report.results)} result(s)")
        for result in report.results:
            for score in result.scores:
                print(f"  [{result.case_id}] {score.dimension.value}: {score.score:.2f}")
                print(f"    Response: {result.response}")
    except Exception as error:
        print(f"Evaluation error: {error}")


if __name__ == "__main__":
    main()
