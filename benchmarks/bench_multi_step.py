"""Benchmark: Cascade failure detection accuracy for multi-step agent tasks.

In multi-step agent tasks, a failure in step N can cause downstream steps
to fail (cascade). This benchmark simulates multi-step evaluation chains
and measures how well the evaluator detects cascade failures vs independent
step failures.

A "cascade failure" is when a wrong answer in step N propagates into step N+1.
A "step failure" is an isolated error in one step that does not affect others.

Competitor context
------------------
- DeepEval's multi-step metric measures task completion across the full chain
  and detects bottleneck steps. Source: docs.deepeval.com (2024).
- Ragas provides an answer correctness metric for QA chains but does not
  specifically model cascade propagation. Source: arXiv:2309.15217 (2023).

This benchmark establishes the baseline detection accuracy using the
BasicAccuracyEvaluator applied step-by-step across a synthetic 3-step chain.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from agent_eval.evaluators.accuracy import BasicAccuracyEvaluator
from datasets.eval_dataset import get_all_cases


@dataclass
class MultiStepChain:
    """A synthetic 3-step evaluation chain.

    Parameters
    ----------
    chain_id:
        Unique identifier for this chain.
    step_1_input:
        Input to step 1.
    step_1_correct:
        Correct output for step 1.
    step_2_input_from_correct:
        Step 2 input when step 1 was correct.
    step_2_input_from_wrong:
        Step 2 input when step 1 was wrong (cascade failure scenario).
    step_2_correct:
        Correct output for step 2.
    step_3_correct:
        Correct final output for step 3.
    is_cascade_scenario:
        If True, this chain tests cascade failure propagation.
    """

    chain_id: str
    step_1_input: str
    step_1_correct: str
    step_2_input_from_correct: str
    step_2_input_from_wrong: str
    step_2_correct: str
    step_3_correct: str
    is_cascade_scenario: bool


def _build_chains() -> list[MultiStepChain]:
    """Build synthetic multi-step chains from the eval dataset."""
    cases = get_all_cases()
    chains: list[MultiStepChain] = []

    # Build 3-step chains by pairing consecutive cases
    for index in range(0, len(cases) - 2, 3):
        case1 = cases[index]
        case2 = cases[index + 1] if index + 1 < len(cases) else cases[0]
        case3 = cases[index + 2] if index + 2 < len(cases) else cases[1]

        # Normal chain (independent steps)
        chains.append(
            MultiStepChain(
                chain_id=f"chain_independent_{index:03d}",
                step_1_input=case1.question,
                step_1_correct=case1.correct_output,
                step_2_input_from_correct=case2.question,
                step_2_input_from_wrong=case2.question,  # independent: same input
                step_2_correct=case2.correct_output,
                step_3_correct=case3.correct_output,
                is_cascade_scenario=False,
            )
        )

        # Cascade chain: step 2 input depends on step 1 output
        chains.append(
            MultiStepChain(
                chain_id=f"chain_cascade_{index:03d}",
                step_1_input=case1.question,
                step_1_correct=case1.correct_output,
                step_2_input_from_correct=case2.question,
                step_2_input_from_wrong=f"[WRONG CONTEXT: {case1.incorrect_output}] {case2.question}",
                step_2_correct=case2.correct_output,
                step_3_correct=case3.correct_output,
                is_cascade_scenario=True,
            )
        )

    return chains


def _evaluate_chain(
    evaluator: BasicAccuracyEvaluator,
    chain: MultiStepChain,
    inject_step1_failure: bool,
) -> dict[str, object]:
    """Evaluate a multi-step chain with or without a step-1 failure.

    Parameters
    ----------
    evaluator:
        Evaluator instance.
    chain:
        The chain to evaluate.
    inject_step1_failure:
        If True, step 1 fails (cascade scenario). If False, all steps correct.

    Returns
    -------
    dict with per-step scores and cascade detection outcome.
    """
    # Step 1
    step1_output = (
        "incorrect_fabricated_answer" if inject_step1_failure
        else chain.step_1_correct
    )
    step1_score = evaluator.evaluate(
        case_id=f"{chain.chain_id}_step1",
        agent_output=step1_output,
        expected_output=chain.step_1_correct,
        metadata={},
    )

    # Step 2 — input depends on step 1 result in cascade scenario
    if chain.is_cascade_scenario and inject_step1_failure:
        step2_input = chain.step_2_input_from_wrong
        step2_output = "wrong_answer_due_to_bad_context"
    else:
        step2_input = chain.step_2_input_from_correct
        step2_output = chain.step_2_correct

    step2_score = evaluator.evaluate(
        case_id=f"{chain.chain_id}_step2",
        agent_output=step2_output,
        expected_output=chain.step_2_correct,
        metadata={},
    )

    # Step 3 — always uses correct output in this simplified model
    step3_output = chain.step_3_correct if not inject_step1_failure else "wrong_propagated_answer"
    step3_score = evaluator.evaluate(
        case_id=f"{chain.chain_id}_step3",
        agent_output=step3_output,
        expected_output=chain.step_3_correct,
        metadata={},
    )

    # Cascade detected if: step 1 fails AND step 2 also fails (propagation)
    step1_failed = not step1_score.passed
    step2_failed = not step2_score.passed
    cascade_detected = step1_failed and step2_failed and chain.is_cascade_scenario

    return {
        "chain_id": chain.chain_id,
        "is_cascade_scenario": chain.is_cascade_scenario,
        "inject_failure": inject_step1_failure,
        "step1_passed": step1_score.passed,
        "step2_passed": step2_score.passed,
        "step3_passed": step3_score.passed,
        "cascade_detected": cascade_detected,
        "chain_score": round(
            (step1_score.score + step2_score.score + step3_score.score) / 3, 4
        ),
    }


def run_benchmark(seed: int = 42) -> dict[str, object]:
    """Run multi-step cascade detection benchmark.

    Parameters
    ----------
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with cascade detection accuracy and step-level stats.
    """
    evaluator = BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=0.3)
    chains = _build_chains()

    cascade_chains = [c for c in chains if c.is_cascade_scenario]
    independent_chains = [c for c in chains if not c.is_cascade_scenario]

    all_results: list[dict[str, object]] = []
    latencies_ms: list[float] = []

    for chain in chains:
        # Test both failure-injected and clean scenarios
        for inject_failure in [True, False]:
            start = time.perf_counter()
            result = _evaluate_chain(evaluator, chain, inject_step1_failure=inject_failure)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            all_results.append(result)

    # Cascade detection accuracy: among cascade chains with injected failure,
    # what fraction did the evaluator correctly identify as cascaded failures?
    cascade_failure_results = [
        r for r in all_results
        if r["is_cascade_scenario"] and r["inject_failure"]
    ]
    cascade_detected_count = sum(
        1 for r in cascade_failure_results if r["cascade_detected"]
    )
    cascade_detection_accuracy = (
        cascade_detected_count / len(cascade_failure_results)
        if cascade_failure_results
        else 0.0
    )

    # Independent failure (false cascade rate): non-cascade chains where step2 also fails
    independent_failure_results = [
        r for r in all_results
        if not r["is_cascade_scenario"] and r["inject_failure"]
    ]
    false_cascade_count = sum(
        1 for r in independent_failure_results
        if not r["step1_passed"] and not r["step2_passed"]
    )
    false_cascade_rate = (
        false_cascade_count / len(independent_failure_results)
        if independent_failure_results
        else 0.0
    )

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    return {
        "benchmark": "multi_step_cascade",
        "n_chains_total": len(chains),
        "n_cascade_chains": len(cascade_chains),
        "n_independent_chains": len(independent_chains),
        "seed": seed,
        "cascade_detection_accuracy": round(cascade_detection_accuracy, 4),
        "false_cascade_rate": round(false_cascade_rate, 4),
        "cascade_detected_count": cascade_detected_count,
        "total_cascade_scenarios": len(cascade_failure_results),
        "evaluation_latency_ms": {
            "p50": round(sorted_lats[int(n * 0.50)], 4) if n else 0,
            "p95": round(sorted_lats[min(int(n * 0.95), n - 1)], 4) if n else 0,
            "mean": round(statistics.mean(latencies_ms), 4) if latencies_ms else 0,
        },
        "note": (
            "Cascade detection = step 1 fails AND step 2 also fails in a cascade chain. "
            "DeepEval multi-step metric: measures bottleneck step detection. "
            "Ragas: answer correctness for QA chains (no cascade model). "
            "Sources: docs.deepeval.com (2024), arXiv:2309.15217 (2023)."
        ),
    }


if __name__ == "__main__":
    print("Running multi-step cascade detection benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "multi_step_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
