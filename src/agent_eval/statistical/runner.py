"""Multi-run statistical evaluation engine.

StatisticalRunner executes an evaluation callable N times and collects
all raw results. It is intentionally decoupled from any specific
evaluation framework so it can wrap any Callable that returns an EvalResult.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from agent_eval.core.evaluator import EvalResult
from agent_eval.statistical.metrics import (
    ConfidenceInterval,
    PassAtKResult,
    confidence_interval,
    pass_at_k_result,
    score_stddev,
    score_variance,
)

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Record of a single evaluation run.

    Parameters
    ----------
    run_index:
        Zero-based index of this run within the N-run sequence.
    passed:
        Whether the evaluation passed on this run.
    score:
        Overall score from this run (mean of dimension scores), in [0.0, 1.0].
    latency_ms:
        Wall-clock time for this run in milliseconds.
    error:
        Exception message if the eval_fn raised, else None.
    result:
        The raw EvalResult from this run. None if eval_fn raised an exception.
    """

    run_index: int
    passed: bool
    score: float
    latency_ms: float
    error: str | None = None
    result: EvalResult | None = None


@dataclass
class StatisticalResult:
    """Aggregated outcome of N evaluation runs.

    Parameters
    ----------
    n_runs:
        Total number of runs attempted.
    records:
        One RunRecord per run.
    pass_at_k_values:
        Precomputed PassAtKResult for k in {1, 3, 5} (where k <= n_runs).
    ci_95:
        Wilson score 95% confidence interval on the pass rate.
    mean_score:
        Mean overall_score across all runs.
    score_std:
        Sample standard deviation of overall_score across runs.
    score_var:
        Sample variance of overall_score across runs.
    """

    n_runs: int
    records: list[RunRecord] = field(default_factory=list)
    pass_at_k_values: list[PassAtKResult] = field(default_factory=list)
    ci_95: ConfidenceInterval | None = None
    mean_score: float = 0.0
    score_std: float = 0.0
    score_var: float = 0.0

    @property
    def n_passed(self) -> int:
        """Number of runs that produced a passing result."""
        return sum(1 for r in self.records if r.passed)

    @property
    def n_failed(self) -> int:
        """Number of runs that produced a failing result."""
        return sum(1 for r in self.records if not r.passed and r.error is None)

    @property
    def n_errors(self) -> int:
        """Number of runs where eval_fn raised an exception."""
        return sum(1 for r in self.records if r.error is not None)

    @property
    def pass_rate(self) -> float:
        """Fraction of runs that passed. 0.0 when no runs completed."""
        if not self.records:
            return 0.0
        return self.n_passed / len(self.records)

    def get_pass_at_k(self, k: int) -> PassAtKResult | None:
        """Return the precomputed pass@k result for a specific k, or None."""
        for result in self.pass_at_k_values:
            if result.k == k:
                return result
        return None


class StatisticalRunner:
    """Executes an evaluation function N times and collects statistical results.

    This runner is decoupled from any specific agent or evaluator — it wraps
    any callable that returns an EvalResult. Use it to measure the reliability
    of stochastic agents where a single evaluation run is insufficient.

    Parameters
    ----------
    n_runs:
        Default number of runs when ``run()`` is called without an explicit
        ``n`` override. Must be >= 1.

    Example
    -------
    ::

        def my_eval() -> EvalResult:
            output = my_agent.call("What is 2+2?")
            return evaluator.evaluate(output)

        runner = StatisticalRunner(n_runs=10)
        result = runner.run(my_eval)
        print(result.get_pass_at_k(3).value)
    """

    _DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)

    def __init__(self, n_runs: int = 5) -> None:
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        self._n_runs = n_runs

    @property
    def n_runs(self) -> int:
        """The default number of runs configured for this runner."""
        return self._n_runs

    def run(
        self,
        eval_fn: Callable[[], EvalResult],
        n: int | None = None,
    ) -> StatisticalResult:
        """Execute eval_fn n times and return aggregated statistical results.

        Parameters
        ----------
        eval_fn:
            A zero-argument callable that returns an EvalResult. Called n times.
            Should be a closure over any inputs needed for evaluation.
        n:
            Number of runs to execute. Overrides the instance default when
            provided. Must be >= 1.

        Returns
        -------
        StatisticalResult
            Aggregated results with pass@k values and confidence intervals.

        Notes
        -----
        Exceptions raised by eval_fn are caught per-run and recorded as
        error records rather than propagating. This allows the runner to
        complete all n runs even if some fail.
        """
        effective_n = n if n is not None else self._n_runs
        if effective_n < 1:
            raise ValueError(f"n must be >= 1, got {effective_n}")

        records: list[RunRecord] = []

        for run_index in range(effective_n):
            start = time.perf_counter()
            try:
                result = eval_fn()
                latency_ms = (time.perf_counter() - start) * 1000.0
                record = RunRecord(
                    run_index=run_index,
                    passed=result.passed,
                    score=result.overall_score,
                    latency_ms=latency_ms,
                    error=result.error,
                    result=result,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "eval_fn raised on run %d: %s", run_index, error_msg
                )
                record = RunRecord(
                    run_index=run_index,
                    passed=False,
                    score=0.0,
                    latency_ms=latency_ms,
                    error=error_msg,
                    result=None,
                )

            records.append(record)

        return self._aggregate(records, effective_n)

    def _aggregate(
        self, records: list[RunRecord], n_runs: int
    ) -> StatisticalResult:
        """Build a StatisticalResult from raw RunRecord data.

        Parameters
        ----------
        records:
            All RunRecord objects from the run loop.
        n_runs:
            Total runs attempted (may differ from len(records) on partial abort).

        Returns
        -------
        StatisticalResult
        """
        n_passed = sum(1 for r in records if r.passed)
        n_total = len(records)

        # Compute pass@k for all default k values where k <= n_total
        pass_at_k_values: list[PassAtKResult] = []
        for k in self._DEFAULT_K_VALUES:
            if k <= n_total or n_total >= 1:
                # Always include k=1; for larger k, compute even if k > n (handles gracefully)
                result = pass_at_k_result(
                    n_correct=n_passed,
                    n_total=max(n_total, 1),
                    k=k,
                )
                pass_at_k_values.append(result)

        # Wilson CI on the pass rate
        ci_95: ConfidenceInterval | None = None
        if n_total >= 1:
            ci_95 = confidence_interval(
                n_successes=n_passed,
                n_trials=n_total,
                confidence=0.95,
            )

        # Score statistics (exclude error records from score aggregation)
        scores = [r.score for r in records if r.error is None]
        if scores:
            mean_score = statistics.mean(scores)
            std = score_stddev(scores)
            var = score_variance(scores)
        else:
            mean_score = 0.0
            std = 0.0
            var = 0.0

        return StatisticalResult(
            n_runs=n_runs,
            records=records,
            pass_at_k_values=pass_at_k_values,
            ci_95=ci_95,
            mean_score=mean_score,
            score_std=std,
            score_var=var,
        )
