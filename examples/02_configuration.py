#!/usr/bin/env python3
"""Example: Advanced Configuration

Demonstrates how to configure EvalRunner with custom options,
thresholds, and suite-level settings.

Usage:
    python examples/02_configuration.py

Requirements:
    pip install agent-eval
"""
from __future__ import annotations

import agent_eval
from agent_eval import (
    RunnerOptions,
    EvalRunner,
    BenchmarkSuite,
    TestCase,
    Dimension,
    CallableAdapter,
    BasicThresholdGate,
)


def build_suite() -> BenchmarkSuite:
    """Build a benchmark suite with multiple test cases."""
    return (
        BenchmarkSuite(name="qa-suite", description="Question-answering benchmark")
        .add_case(
            TestCase(
                id="tc-01",
                input="What is the capital of France?",
                expected_output="Paris",
                dimensions=[Dimension.ACCURACY],
            )
        )
        .add_case(
            TestCase(
                id="tc-02",
                input="Explain recursion briefly.",
                expected_output="A function calling itself",
                dimensions=[Dimension.ACCURACY, Dimension.COHERENCE],
            )
        )
    )


def simple_agent(user_input: str) -> str:
    """Stub agent for demonstration."""
    responses: dict[str, str] = {
        "What is the capital of France?": "Paris",
        "Explain recursion briefly.": "A function that calls itself to solve subproblems.",
    }
    return responses.get(user_input, "I don't know.")


def main() -> None:
    # Step 1: Configure runner options
    options = RunnerOptions(
        max_workers=2,
        timeout_seconds=30,
        fail_fast=False,
        verbose=True,
    )
    print(f"Runner options: max_workers={options.max_workers}, timeout={options.timeout_seconds}s")

    # Step 2: Build a threshold gate
    gate = BasicThresholdGate(
        thresholds={
            Dimension.ACCURACY: 0.7,
            Dimension.COHERENCE: 0.6,
        }
    )
    print(f"Gate thresholds: {gate.thresholds}")

    # Step 3: Create adapter and runner
    adapter = CallableAdapter(fn=simple_agent, name="stub-agent")
    suite = build_suite()
    runner = EvalRunner(adapter=adapter, suite=suite, options=options)

    # Step 4: Run evaluation
    try:
        report = runner.run()
        gate_result = gate.evaluate(report)
        print(f"\nEval complete — {len(report.results)} results")
        print(f"Gate passed: {gate_result.passed}")
        for dimension, score in gate_result.scores.items():
            threshold = gate.thresholds.get(dimension, 0.0)
            status = "PASS" if score >= threshold else "FAIL"
            print(f"  {dimension.value}: {score:.2f} (threshold {threshold:.2f}) [{status}]")
    except Exception as error:
        print(f"Evaluation failed: {error}")


if __name__ == "__main__":
    main()
