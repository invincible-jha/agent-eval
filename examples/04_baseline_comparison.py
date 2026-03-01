#!/usr/bin/env python3
"""Example: Baseline Comparison

Demonstrates how to run two agents against the same suite and compare
their scores to identify regressions or improvements.

Usage:
    python examples/04_baseline_comparison.py

Requirements:
    pip install agent-eval
"""
from __future__ import annotations

from agent_eval import (
    BenchmarkSuite,
    CallableAdapter,
    Dimension,
    EvalRunner,
    EvalReport,
    RunnerOptions,
    TestCase,
)


SUITE = (
    BenchmarkSuite(name="comparison-suite", description="Baseline vs. candidate comparison")
    .add_case(TestCase(id="q1", input="What year did WWII end?", expected_output="1945",
                       dimensions=[Dimension.ACCURACY]))
    .add_case(TestCase(id="q2", input="What is photosynthesis?",
                       expected_output="Plants converting light to energy",
                       dimensions=[Dimension.ACCURACY, Dimension.COHERENCE]))
    .add_case(TestCase(id="q3", input="Name a sorting algorithm.",
                       expected_output="quicksort",
                       dimensions=[Dimension.ACCURACY]))
)


def baseline_agent(user_input: str) -> str:
    """Older agent with partial knowledge."""
    answers: dict[str, str] = {
        "What year did WWII end?": "1945",
        "What is photosynthesis?": "Plants use sunlight.",
        "Name a sorting algorithm.": "bubble sort",
    }
    return answers.get(user_input, "unknown")


def candidate_agent(user_input: str) -> str:
    """Improved candidate agent."""
    answers: dict[str, str] = {
        "What year did WWII end?": "World War II ended in 1945.",
        "What is photosynthesis?": "Photosynthesis is the process by which plants convert sunlight into energy.",
        "Name a sorting algorithm.": "Quicksort is an efficient sorting algorithm using divide and conquer.",
    }
    return answers.get(user_input, "I don't know.")


def average_score(report: EvalReport) -> float:
    """Compute the mean score across all results and dimensions."""
    all_scores: list[float] = []
    for result in report.results:
        for score in result.scores:
            all_scores.append(score.score)
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


def main() -> None:
    options = RunnerOptions(max_workers=1, verbose=False)

    # Step 1: Run baseline
    baseline_adapter = CallableAdapter(fn=baseline_agent, name="baseline-v1")
    baseline_runner = EvalRunner(adapter=baseline_adapter, suite=SUITE, options=options)
    baseline_report = baseline_runner.run()
    baseline_avg = average_score(baseline_report)

    # Step 2: Run candidate
    candidate_adapter = CallableAdapter(fn=candidate_agent, name="candidate-v2")
    candidate_runner = EvalRunner(adapter=candidate_adapter, suite=SUITE, options=options)
    candidate_report = candidate_runner.run()
    candidate_avg = average_score(candidate_report)

    # Step 3: Print comparison
    print("Baseline vs. Candidate comparison")
    print(f"  Baseline  average score: {baseline_avg:.2f}")
    print(f"  Candidate average score: {candidate_avg:.2f}")
    delta = candidate_avg - baseline_avg
    direction = "improvement" if delta >= 0 else "regression"
    print(f"  Delta: {delta:+.2f} ({direction})")

    print("\nPer-case breakdown:")
    for base_result, cand_result in zip(baseline_report.results, candidate_report.results):
        base_mean = sum(s.score for s in base_result.scores) / max(len(base_result.scores), 1)
        cand_mean = sum(s.score for s in cand_result.scores) / max(len(cand_result.scores), 1)
        print(f"  [{base_result.case_id}] baseline={base_mean:.2f}  candidate={cand_mean:.2f}")


if __name__ == "__main__":
    main()
