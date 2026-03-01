#!/usr/bin/env python3
"""Example: CrewAI Integration

Demonstrates how to evaluate a CrewAI crew's output using agent-eval
by wrapping the crew's kickoff as a CallableAdapter.

Usage:
    python examples/07_crewai_eval.py

Requirements:
    pip install agent-eval crewai
"""
from __future__ import annotations

try:
    from crewai import Agent, Task, Crew, Process
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

from agent_eval import (
    BenchmarkSuite,
    CallableAdapter,
    CompositeGate,
    Dimension,
    EvalRunner,
    RunnerOptions,
    TestCase,
    BasicThresholdGate,
)


def build_crewai_fn() -> "object":
    """Build a CrewAI crew and return a callable."""
    if not _CREWAI_AVAILABLE:
        return None

    researcher = Agent(
        role="Research Analyst",
        goal="Provide accurate answers to questions",
        backstory="You are a knowledgeable research analyst.",
        verbose=False,
    )

    def run_crew(user_input: str) -> str:
        task = Task(
            description=user_input,
            agent=researcher,
            expected_output="A concise, accurate answer",
        )
        crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        return str(result)

    return run_crew


def stub_crewai(user_input: str) -> str:
    """Fallback when crewai is unavailable."""
    return f"[crewai-stub] Response to: {user_input[:40]}"


def main() -> None:
    if not _CREWAI_AVAILABLE:
        print("crewai not installed — using stub agent for demonstration.")
        print("Install with: pip install crewai")
        agent_fn = stub_crewai
    else:
        print("crewai available — building research crew.")
        agent_fn = build_crewai_fn()

    # Step 1: Build evaluation suite
    suite = (
        BenchmarkSuite(name="crewai-suite", description="CrewAI crew evaluation")
        .add_case(TestCase(
            id="crew-01",
            input="What is the boiling point of water in Celsius?",
            expected_output="100",
            dimensions=[Dimension.ACCURACY],
        ))
        .add_case(TestCase(
            id="crew-02",
            input="Who wrote the play Hamlet?",
            expected_output="Shakespeare",
            dimensions=[Dimension.ACCURACY],
        ))
        .add_case(TestCase(
            id="crew-03",
            input="Explain what an API is in simple terms.",
            expected_output="",
            dimensions=[Dimension.COHERENCE],
        ))
    )

    # Step 2: Build composite gate with multiple thresholds
    composite_gate = CompositeGate(
        gates=[
            BasicThresholdGate(thresholds={Dimension.ACCURACY: 0.6}),
            BasicThresholdGate(thresholds={Dimension.COHERENCE: 0.5}),
        ]
    )

    # Step 3: Run evaluation
    adapter = CallableAdapter(fn=agent_fn, name="crewai-research-crew")
    runner = EvalRunner(
        adapter=adapter,
        suite=suite,
        options=RunnerOptions(max_workers=1, verbose=True),
    )

    try:
        report = runner.run()
        gate_result = composite_gate.evaluate(report)
        print(f"\nCrewAI eval complete — {len(report.results)} cases")
        print(f"Composite gate passed: {gate_result.passed}")
        for dim, score in gate_result.scores.items():
            print(f"  {dim.value}: {score:.2f}")
    except Exception as error:
        print(f"Eval error: {error}")


if __name__ == "__main__":
    main()
