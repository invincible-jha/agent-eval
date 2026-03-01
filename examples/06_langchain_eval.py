#!/usr/bin/env python3
"""Example: LangChain Integration

Demonstrates wrapping a LangChain chain with a CallableAdapter so
agent-eval can score it using standard dimensions.

Usage:
    python examples/06_langchain_eval.py

Requirements:
    pip install agent-eval langchain langchain-openai
"""
from __future__ import annotations

try:
    from langchain.schema import HumanMessage
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

from agent_eval import (
    BenchmarkSuite,
    CallableAdapter,
    Dimension,
    EvalRunner,
    RunnerOptions,
    TestCase,
    BasicThresholdGate,
)


def build_langchain_agent() -> "object":
    """Build a minimal LangChain chat agent."""
    if not _LANGCHAIN_AVAILABLE:
        return None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def respond(user_input: str) -> str:
        result = llm.invoke([HumanMessage(content=user_input)])
        return result.content

    return respond


def stub_agent(user_input: str) -> str:
    """Fallback stub when LangChain is unavailable."""
    return f"[stub] Answer to: {user_input}"


def main() -> None:
    if not _LANGCHAIN_AVAILABLE:
        print("LangChain not installed — using stub agent for demonstration.")
        print("Install with: pip install langchain langchain-openai")
        agent_fn = stub_agent
    else:
        print("LangChain available — building ChatOpenAI agent.")
        agent_fn = build_langchain_agent()

    # Step 1: Define evaluation suite
    suite = (
        BenchmarkSuite(name="langchain-suite", description="LangChain agent evaluation")
        .add_case(TestCase(
            id="lc-01",
            input="What is the largest planet in our solar system?",
            expected_output="Jupiter",
            dimensions=[Dimension.ACCURACY],
        ))
        .add_case(TestCase(
            id="lc-02",
            input="Summarise the concept of machine learning in one sentence.",
            expected_output="",
            dimensions=[Dimension.COHERENCE],
        ))
        .add_case(TestCase(
            id="lc-03",
            input="What programming language is commonly used for data science?",
            expected_output="Python",
            dimensions=[Dimension.ACCURACY],
        ))
    )

    # Step 2: Wrap with adapter and run
    adapter = CallableAdapter(fn=agent_fn, name="langchain-agent")
    runner = EvalRunner(
        adapter=adapter,
        suite=suite,
        options=RunnerOptions(max_workers=1, verbose=True),
    )

    try:
        report = runner.run()
        gate = BasicThresholdGate(thresholds={Dimension.ACCURACY: 0.5, Dimension.COHERENCE: 0.5})
        gate_result = gate.evaluate(report)

        print(f"\nLangChain eval complete — {len(report.results)} cases")
        print(f"Gate passed: {gate_result.passed}")
        for dim, score in gate_result.scores.items():
            print(f"  {dim.value}: {score:.2f}")
    except Exception as error:
        print(f"Eval error: {error}")


if __name__ == "__main__":
    main()
