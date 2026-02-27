"""Benchmark: Automated metrics vs synthetic human judgements (Spearman correlation).

Tests how well the BasicAccuracyEvaluator's score correlates with synthetic
human judgement scores. Uses Spearman rank correlation (stdlib only, no scipy).

Competitor context
------------------
- DeepEval reports Spearman correlations of 0.7-0.9 with human judgement
  for their LLM-judge-based metrics on QA benchmarks.
  Source: DeepEval documentation (https://docs.deepeval.com, 2024).
- Ragas reports 0.65-0.85 correlation for faithfulness and relevance metrics.
  Source: Ragas paper (arXiv:2309.15217, 2023).

This benchmark measures the simpler Jaccard/exact-match evaluator correlation
with synthetic human scores. Higher correlation = better automated proxy.
The goal is >0.7 Spearman correlation for the automated metrics.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from agent_eval.evaluators.accuracy import BasicAccuracyEvaluator
from datasets.eval_dataset import EvalTestCase, get_all_cases


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient using stdlib only.

    Parameters
    ----------
    x:
        First sequence of values.
    y:
        Second sequence of values (same length as x).

    Returns
    -------
    float
        Spearman rho in [-1, 1]. Returns 0.0 if constant sequence.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    n = len(x)
    if n < 2:
        return 0.0

    def _ranks(values: list[float]) -> list[float]:
        """Assign average ranks (handles ties)."""
        indexed = sorted(enumerate(values), key=lambda pair: pair[1])
        ranks = [0.0] * n
        index = 0
        while index < n:
            j = index
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (index + j) / 2 + 1
            for k in range(index, j + 1):
                ranks[indexed[k][0]] = avg_rank
            index = j + 1
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    denominator = n * (n * n - 1)
    if denominator == 0:
        return 0.0
    return round(1.0 - (6 * d_sq_sum / denominator), 4)


def run_benchmark(
    modes: list[str] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Measure Spearman correlation between automated scores and human scores.

    For each evaluation mode, the evaluator scores all (output, expected) pairs
    and the resulting scores are correlated with the synthetic human judgements.

    Parameters
    ----------
    modes:
        Evaluation modes to test. Defaults to ['exact', 'fuzzy', 'contains'].
    seed:
        Unused (reserved for consistency).

    Returns
    -------
    dict with Spearman correlation per mode.
    """
    if modes is None:
        modes = ["exact", "fuzzy", "contains"]

    cases = get_all_cases()

    # Build triplets: (agent_output, expected, human_score)
    triplets: list[tuple[str, str, float, str]] = []
    for case in cases:
        triplets.append((case.correct_output, case.expected_answer, case.human_score_correct, case.case_id))
        triplets.append((case.partial_output, case.expected_answer, case.human_score_partial, case.case_id))
        triplets.append((case.incorrect_output, case.expected_answer, case.human_score_incorrect, case.case_id))

    mode_results: dict[str, dict[str, object]] = {}
    latencies_ms: list[float] = []

    for mode in modes:
        evaluator = BasicAccuracyEvaluator(mode=mode, fuzzy_threshold=0.3)
        automated_scores: list[float] = []
        human_scores: list[float] = []

        for agent_output, expected, human_score, case_id in triplets:
            start = time.perf_counter()
            dim_score = evaluator.evaluate(
                case_id=case_id,
                agent_output=agent_output,
                expected_output=expected,
                metadata={},
            )
            latencies_ms.append((time.perf_counter() - start) * 1000)
            automated_scores.append(dim_score.score)
            human_scores.append(human_score)

        correlation = _spearman_correlation(automated_scores, human_scores)
        mode_results[mode] = {
            "spearman_rho": correlation,
            "n_pairs": len(triplets),
            "mean_automated_score": round(statistics.mean(automated_scores), 4),
            "mean_human_score": round(statistics.mean(human_scores), 4),
        }

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    return {
        "benchmark": "metric_correlation",
        "seed": seed,
        "n_cases": len(cases),
        "n_triplets": len(triplets),
        "correlation_by_mode": mode_results,
        "evaluation_latency_ms": {
            "p50": round(sorted_lats[int(n * 0.50)], 4) if n else 0,
            "p95": round(sorted_lats[min(int(n * 0.95), n - 1)], 4) if n else 0,
            "mean": round(statistics.mean(latencies_ms), 4) if latencies_ms else 0,
        },
        "note": (
            "DeepEval: 0.7-0.9 Spearman vs human (LLM-judge-based). "
            "Ragas: 0.65-0.85 for faithfulness/relevance. "
            "Sources: docs.deepeval.com (2024), arXiv:2309.15217 (2023). "
            "This measures BasicAccuracyEvaluator (Jaccard/exact) vs synthetic human scores. "
            "LLM-judge-based metrics expected to correlate significantly higher."
        ),
    }


if __name__ == "__main__":
    print("Running metric correlation benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
