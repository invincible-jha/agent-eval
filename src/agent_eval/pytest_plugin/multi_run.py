"""Multi-run evaluator for consistency measurement in pytest-agent-eval.

Running an agent evaluation multiple times exposes non-determinism in LLM
outputs. :class:`MultiRunEvaluator` records scores from repeated runs and
computes per-metric consistency statistics via :meth:`compute_consistency`.

Usage example
-------------
::

    from agent_eval.pytest_plugin.multi_run import MultiRunEvaluator

    evaluator = MultiRunEvaluator()
    for _ in range(5):
        scores = run_agent_and_score()
        evaluator.record_run(scores)

    for result in evaluator.compute_consistency():
        print(result.metric, result.consistency_score)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class ConsistencyResult:
    """Per-metric consistency statistics across multiple evaluation runs.

    Parameters
    ----------
    metric:
        The dimension name (e.g. ``"accuracy"``).
    mean:
        Mean score across all recorded runs.
    std_dev:
        Sample standard deviation of scores (0.0 when only one run recorded).
    min_val:
        Minimum score observed across all runs.
    max_val:
        Maximum score observed across all runs.
    consistency_score:
        ``1.0 - (std_dev / mean)`` when mean > 0, else 0.0.
        Clamped to [0.0, 1.0]. Higher values indicate more consistent output.
    """

    metric: str
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    consistency_score: float


class MultiRunEvaluator:
    """Record evaluation scores across multiple runs and compute consistency.

    Each call to :meth:`record_run` adds one run's scores. After at least
    two runs, :meth:`compute_consistency` returns per-metric statistics.
    """

    def __init__(self) -> None:
        self._runs: list[dict[str, float]] = []

    def record_run(self, scores: dict[str, float]) -> None:
        """Record the dimension scores from one evaluation run.

        Parameters
        ----------
        scores:
            Mapping of dimension name to score in [0.0, 1.0].
        """
        self._runs.append(dict(scores))

    @property
    def run_count(self) -> int:
        """Return the number of runs recorded so far."""
        return len(self._runs)

    def compute_consistency(self) -> list[ConsistencyResult]:
        """Compute consistency statistics across all recorded runs.

        Returns an empty list when fewer than two runs have been recorded,
        since standard deviation is undefined for a single data point.

        Returns
        -------
        list[ConsistencyResult]
            One entry per metric found in any run, sorted alphabetically.
        """
        if len(self._runs) < 2:
            return []

        metrics: set[str] = set()
        for run in self._runs:
            metrics.update(run.keys())

        results: list[ConsistencyResult] = []
        for metric in sorted(metrics):
            values = [run.get(metric, 0.0) for run in self._runs]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            consistency = max(0.0, 1.0 - (std / mean)) if mean > 0 else 0.0
            results.append(
                ConsistencyResult(
                    metric=metric,
                    mean=round(mean, 4),
                    std_dev=round(std, 4),
                    min_val=min(values),
                    max_val=max(values),
                    consistency_score=round(consistency, 4),
                )
            )

        return results
