"""Tests for agent_eval.statistical.runner.

Uses deterministic mock eval_fn callables to test pass@k, confidence
intervals, error handling, and aggregation logic.
"""
from __future__ import annotations

from agent_eval.core.evaluator import (
    Dimension,
    DimensionScore,
    EvalResult,
)
from agent_eval.statistical.runner import (
    RunRecord,
    StatisticalResult,
    StatisticalRunner,
)


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _make_eval_result(passed: bool, score: float = 1.0) -> EvalResult:
    """Create a minimal EvalResult for testing."""
    dimension_score = DimensionScore(
        dimension=Dimension.ACCURACY,
        score=score,
        passed=passed,
    )
    return EvalResult(
        case_id="test_case",
        run_index=0,
        agent_output="mock output",
        dimension_scores=[dimension_score],
    )


def always_pass() -> EvalResult:
    """Deterministic eval_fn: always passes with score 1.0."""
    return _make_eval_result(passed=True, score=1.0)


def always_fail() -> EvalResult:
    """Deterministic eval_fn: always fails with score 0.0."""
    return _make_eval_result(passed=False, score=0.0)


class FlakyEvalFn:
    """Alternates between pass and fail every call."""

    def __init__(self, pass_on_even: bool = True) -> None:
        self._count = 0
        self._pass_on_even = pass_on_even

    def __call__(self) -> EvalResult:
        should_pass = (self._count % 2 == 0) == self._pass_on_even
        self._count += 1
        return _make_eval_result(passed=should_pass, score=1.0 if should_pass else 0.0)


class CountedPassRateEvalFn:
    """Passes for the first `n_pass` of `n_total` calls."""

    def __init__(self, n_pass: int, n_total: int) -> None:
        self._n_pass = n_pass
        self._count = 0

    def __call__(self) -> EvalResult:
        should_pass = self._count < self._n_pass
        self._count += 1
        return _make_eval_result(passed=should_pass, score=1.0 if should_pass else 0.0)


def raising_eval_fn() -> EvalResult:
    """An eval_fn that always raises an exception."""
    raise RuntimeError("Simulated eval failure")


# ---------------------------------------------------------------------------
# Tests for StatisticalRunner initialization
# ---------------------------------------------------------------------------

class TestStatisticalRunnerInit:
    def test_default_n_runs(self) -> None:
        runner = StatisticalRunner()
        assert runner.n_runs == 5

    def test_custom_n_runs(self) -> None:
        runner = StatisticalRunner(n_runs=10)
        assert runner.n_runs == 10

    def test_raises_on_zero_n_runs(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="n_runs must be >= 1"):
            StatisticalRunner(n_runs=0)

    def test_raises_on_negative_n_runs(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="n_runs must be >= 1"):
            StatisticalRunner(n_runs=-1)


# ---------------------------------------------------------------------------
# Tests for run() with deterministic callables
# ---------------------------------------------------------------------------

class TestStatisticalRunnerAlwaysPass:
    """Deterministic: eval_fn always passes."""

    def setup_method(self) -> None:
        self.runner = StatisticalRunner(n_runs=5)
        self.result = self.runner.run(always_pass)

    def test_returns_statistical_result(self) -> None:
        assert isinstance(self.result, StatisticalResult)

    def test_all_records_pass(self) -> None:
        assert all(r.passed for r in self.result.records)

    def test_n_passed_equals_n_runs(self) -> None:
        assert self.result.n_passed == 5

    def test_n_failed_is_zero(self) -> None:
        assert self.result.n_failed == 0

    def test_n_errors_is_zero(self) -> None:
        assert self.result.n_errors == 0

    def test_pass_rate_is_one(self) -> None:
        assert self.result.pass_rate == 1.0

    def test_pass_at_1_is_one(self) -> None:
        pak1 = self.result.get_pass_at_k(1)
        assert pak1 is not None
        assert pak1.value == 1.0

    def test_pass_at_3_is_one(self) -> None:
        pak3 = self.result.get_pass_at_k(3)
        assert pak3 is not None
        assert pak3.value == 1.0

    def test_pass_at_5_is_one(self) -> None:
        pak5 = self.result.get_pass_at_k(5)
        assert pak5 is not None
        assert pak5.value == 1.0

    def test_mean_score_is_one(self) -> None:
        assert abs(self.result.mean_score - 1.0) < 1e-9

    def test_score_std_is_zero(self) -> None:
        assert self.result.score_std == 0.0

    def test_ci_95_is_present(self) -> None:
        assert self.result.ci_95 is not None

    def test_ci_lower_near_one(self) -> None:
        assert self.result.ci_95 is not None
        assert self.result.ci_95.lower >= 0.5


class TestStatisticalRunnerAlwaysFail:
    """Deterministic: eval_fn always fails."""

    def setup_method(self) -> None:
        self.runner = StatisticalRunner(n_runs=5)
        self.result = self.runner.run(always_fail)

    def test_all_records_fail(self) -> None:
        assert not any(r.passed for r in self.result.records)

    def test_n_passed_is_zero(self) -> None:
        assert self.result.n_passed == 0

    def test_n_failed_equals_n_runs(self) -> None:
        assert self.result.n_failed == 5

    def test_pass_at_1_is_zero(self) -> None:
        pak1 = self.result.get_pass_at_k(1)
        assert pak1 is not None
        assert pak1.value == 0.0

    def test_pass_at_k_all_zero(self) -> None:
        for k in [1, 3, 5]:
            pak = self.result.get_pass_at_k(k)
            assert pak is not None
            assert pak.value == 0.0

    def test_mean_score_is_zero(self) -> None:
        assert abs(self.result.mean_score - 0.0) < 1e-9


class TestStatisticalRunnerFlaky:
    """Flaky: alternating pass/fail, 50% pass rate."""

    def setup_method(self) -> None:
        self.runner = StatisticalRunner(n_runs=10)
        # 10 runs, alternating: 5 pass, 5 fail
        flaky = FlakyEvalFn(pass_on_even=True)
        self.result = self.runner.run(flaky)

    def test_n_passed_is_five(self) -> None:
        assert self.result.n_passed == 5

    def test_pass_rate_is_half(self) -> None:
        assert abs(self.result.pass_rate - 0.5) < 1e-9

    def test_pass_at_1_less_than_one(self) -> None:
        pak1 = self.result.get_pass_at_k(1)
        assert pak1 is not None
        assert pak1.value < 1.0

    def test_pass_at_k_increasing(self) -> None:
        p1 = self.result.get_pass_at_k(1)
        p3 = self.result.get_pass_at_k(3)
        p5 = self.result.get_pass_at_k(5)
        assert p1 is not None and p3 is not None and p5 is not None
        assert p1.value <= p3.value <= p5.value


class TestStatisticalRunnerErrors:
    """eval_fn raises exceptions."""

    def setup_method(self) -> None:
        self.runner = StatisticalRunner(n_runs=3)
        self.result = self.runner.run(raising_eval_fn)

    def test_n_errors_equals_n_runs(self) -> None:
        assert self.result.n_errors == 3

    def test_n_passed_is_zero(self) -> None:
        assert self.result.n_passed == 0

    def test_records_have_error_set(self) -> None:
        for record in self.result.records:
            assert record.error is not None
            assert "RuntimeError" in record.error

    def test_records_have_no_result(self) -> None:
        for record in self.result.records:
            assert record.result is None


class TestStatisticalRunnerOverrideN:
    """Test that n parameter overrides the instance default."""

    def test_override_n_runs(self) -> None:
        runner = StatisticalRunner(n_runs=5)
        result = runner.run(always_pass, n=3)
        assert result.n_runs == 3
        assert len(result.records) == 3

    def test_zero_n_raises(self) -> None:
        import pytest
        runner = StatisticalRunner(n_runs=5)
        with pytest.raises(ValueError, match="n must be >= 1"):
            runner.run(always_pass, n=0)


class TestRunRecord:
    """Tests for RunRecord dataclass."""

    def test_run_record_attributes(self) -> None:
        record = RunRecord(
            run_index=0,
            passed=True,
            score=0.9,
            latency_ms=12.5,
        )
        assert record.run_index == 0
        assert record.passed is True
        assert abs(record.score - 0.9) < 1e-9
        assert abs(record.latency_ms - 12.5) < 1e-9
        assert record.error is None
        assert record.result is None
