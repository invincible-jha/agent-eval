#!/usr/bin/env python3
"""Example: Integration with agentcore-sdk

Demonstrates wrapping an agentcore EventBus-driven agent with a
CallableAdapter so agent-eval can score it end-to-end.

Usage:
    python examples/05_integration_agentcore.py

Requirements:
    pip install agent-eval agentcore-sdk
"""
from __future__ import annotations

import asyncio

try:
    from agentcore import EventBus, AgentEvent, EventType, CostTracker, TokenUsage
    _AGENTCORE_AVAILABLE = True
except ImportError:
    _AGENTCORE_AVAILABLE = False

from agent_eval import (
    BenchmarkSuite,
    CallableAdapter,
    Dimension,
    EvalRunner,
    RunnerOptions,
    TestCase,
)


def build_agentcore_agent() -> "object":
    """Build a minimal agentcore-backed agent that emits lifecycle events."""
    if not _AGENTCORE_AVAILABLE:
        return None

    bus = EventBus()
    cost_tracker = CostTracker()
    recorded_events: list[AgentEvent] = []

    bus.subscribe(EventType.AGENT_STARTED, recorded_events.append)
    bus.subscribe(EventType.AGENT_COMPLETED, recorded_events.append)

    def respond(user_input: str) -> str:
        asyncio.run(bus.emit(AgentEvent(EventType.AGENT_STARTED, "eval-agent")))
        response = f"Processed: {user_input[:50]}"
        usage = TokenUsage(input_tokens=len(user_input.split()), output_tokens=10)
        cost_tracker.record("claude-haiku-4", usage)
        asyncio.run(bus.emit(AgentEvent(EventType.AGENT_COMPLETED, "eval-agent")))
        return response

    return respond, cost_tracker, recorded_events


def main() -> None:
    if not _AGENTCORE_AVAILABLE:
        print("agentcore-sdk not installed — showing structure only.")
        print("Install with: pip install agentcore-sdk")
        print("\nExpected workflow:")
        print("  1. Build agentcore EventBus + CostTracker")
        print("  2. Wrap agent function with CallableAdapter")
        print("  3. Run agent-eval suite against the wrapped agent")
        print("  4. Review scores alongside cost data from CostTracker")
        return

    # Step 1: Build agentcore-backed agent
    agent_fn, cost_tracker, events = build_agentcore_agent()
    print("Agentcore agent built with EventBus and CostTracker")

    # Step 2: Create eval suite
    suite = (
        BenchmarkSuite(name="agentcore-suite", description="agentcore integration test")
        .add_case(TestCase(id="ac-01", input="Hello agent", expected_output="Processed: Hello agent",
                           dimensions=[Dimension.ACCURACY]))
        .add_case(TestCase(id="ac-02", input="Tell me about Python",
                           expected_output="Processed: Tell me about Python",
                           dimensions=[Dimension.ACCURACY]))
    )

    # Step 3: Wrap with CallableAdapter and run eval
    adapter = CallableAdapter(fn=agent_fn, name="agentcore-agent")
    runner = EvalRunner(adapter=adapter, suite=suite, options=RunnerOptions(verbose=True))
    report = runner.run()

    # Step 4: Print results alongside agentcore cost data
    print(f"\nEval results: {len(report.results)} cases")
    for result in report.results:
        mean = sum(s.score for s in result.scores) / max(len(result.scores), 1)
        print(f"  [{result.case_id}] score={mean:.2f}")

    summary = cost_tracker.summary()
    print(f"\nAgentcore cost summary: total_calls={summary.total_calls}")
    print(f"  Events emitted: {len(events)}")


if __name__ == "__main__":
    main()
