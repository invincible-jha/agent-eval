"""Tests for agent_eval.statistical.metrics.

Covers pass_at_k, confidence_interval, and helper statistics functions.
"""
from __future__ import annotations

import math

import pytest

from agent_eval.statistical.metrics import (
    ConfidenceInterval,
    PassAtKResult,
    _z_score_for_confidence,
    confidence_interval,
    pass_at_k,
    pass_at_k_result,
    score_stddev,
    score_variance,
)


class TestPassAtK:
    """Tests for the pass_at_k function."""

    def test_all_pass_returns_one_for_any_k(self) -> None:
        """When all runs pass, pass@k should be 1.0 for any k."""
        assert pass_at_k(n_correct=5, n_total=5, k=1) == 1.0
        assert pass_at_k(n_correct=5, n_total=5, k=3) == 1.0
        assert pass_at_k(n_correct=5, n_total=5, k=5) == 1.0

    def test_all_fail_returns_zero_for_any_k(self) -> None:
        """When no runs pass, pass@k should be 0.0 for any k."""
        assert pass_at_k(n_correct=0, n_total=5, k=1) == 0.0
        assert pass_at_k(n_correct=0, n_total=5, k=3) == 0.0
        assert pass_at_k(n_correct=0, n_total=5, k=5) == 0.0

    def test_pass_at_1_equals_pass_rate(self) -> None:
        """pass@1 should equal the simple pass rate c/n."""
        # With n=10, c=5: pass@1 = 1 - C(5,1)/C(10,1) = 1 - 5/10 = 0.5
        value = pass_at_k(n_correct=5, n_total=10, k=1)
        assert abs(value - 0.5) < 1e-9

    def test_pass_at_k_increases_with_k(self) -> None:
        """pass@k should be non-decreasing as k increases."""
        n, c = 10, 4
        p1 = pass_at_k(n_correct=c, n_total=n, k=1)
        p3 = pass_at_k(n_correct=c, n_total=n, k=3)
        p5 = pass_at_k(n_correct=c, n_total=n, k=5)
        assert p1 <= p3 <= p5

    def test_k_greater_than_n_returns_one_if_any_pass(self) -> None:
        """When k > n and at least one run passes, return 1.0."""
        assert pass_at_k(n_correct=1, n_total=3, k=5) == 1.0

    def test_k_greater_than_n_returns_zero_if_none_pass(self) -> None:
        """When k > n and no runs pass, return 0.0."""
        assert pass_at_k(n_correct=0, n_total=3, k=5) == 0.0

    def test_single_run_pass(self) -> None:
        """With n=1, c=1, k=1: pass@1 = 1.0."""
        assert pass_at_k(n_correct=1, n_total=1, k=1) == 1.0

    def test_single_run_fail(self) -> None:
        """With n=1, c=0, k=1: pass@1 = 0.0."""
        assert pass_at_k(n_correct=0, n_total=1, k=1) == 0.0

    def test_large_n_stability(self) -> None:
        """With n=100, c=50, k=5: result should be in (0, 1)."""
        value = pass_at_k(n_correct=50, n_total=100, k=5)
        assert 0.0 < value < 1.0

    def test_raises_on_invalid_n_total(self) -> None:
        with pytest.raises(ValueError, match="n_total must be >= 1"):
            pass_at_k(n_correct=0, n_total=0, k=1)

    def test_raises_on_negative_n_correct(self) -> None:
        with pytest.raises(ValueError, match="n_correct must be >= 0"):
            pass_at_k(n_correct=-1, n_total=5, k=1)

    def test_raises_when_n_correct_exceeds_n_total(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            pass_at_k(n_correct=6, n_total=5, k=1)

    def test_raises_on_k_less_than_one(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            pass_at_k(n_correct=3, n_total=5, k=0)

    def test_formula_known_value(self) -> None:
        """Verify against manually calculated value: n=5, c=3, k=2."""
        # pass@2 = 1 - C(n-c, k) / C(n, k) = 1 - C(2,2) / C(5,2)
        # = 1 - 1/10 = 0.9
        value = pass_at_k(n_correct=3, n_total=5, k=2)
        assert abs(value - 0.9) < 1e-9

    def test_almost_all_fail_large_n(self) -> None:
        """n=100, c=1, k=1: pass@1 should be close to 0.01."""
        value = pass_at_k(n_correct=1, n_total=100, k=1)
        assert abs(value - 0.01) < 1e-9


class TestPassAtKResult:
    """Tests for pass_at_k_result which returns a structured object."""

    def test_returns_pass_at_k_result_type(self) -> None:
        result = pass_at_k_result(n_correct=3, n_total=5, k=1)
        assert isinstance(result, PassAtKResult)

    def test_fields_are_correctly_set(self) -> None:
        result = pass_at_k_result(n_correct=3, n_total=5, k=1)
        assert result.k == 1
        assert result.n_total == 5
        assert result.n_correct == 3
        assert 0.0 <= result.value <= 1.0

    def test_value_matches_standalone_function(self) -> None:
        expected = pass_at_k(n_correct=7, n_total=10, k=3)
        result = pass_at_k_result(n_correct=7, n_total=10, k=3)
        assert abs(result.value - expected) < 1e-12


class TestConfidenceInterval:
    """Tests for the Wilson score confidence interval."""

    def test_returns_confidence_interval_type(self) -> None:
        ci = confidence_interval(n_successes=5, n_trials=10)
        assert isinstance(ci, ConfidenceInterval)

    def test_all_success_upper_bound_near_one(self) -> None:
        """When all trials succeed, upper CI bound should be near 1.0."""
        ci = confidence_interval(n_successes=10, n_trials=10)
        assert ci.upper >= 0.9
        assert ci.lower >= 0.7

    def test_all_failure_lower_bound_near_zero(self) -> None:
        """When no trials succeed, lower CI bound should be near 0.0."""
        ci = confidence_interval(n_successes=0, n_trials=10)
        assert ci.lower == 0.0
        assert ci.upper <= 0.3

    def test_bounds_are_ordered(self) -> None:
        """Lower bound must always be <= upper bound."""
        ci = confidence_interval(n_successes=5, n_trials=10)
        assert ci.lower <= ci.upper

    def test_bounds_within_unit_interval(self) -> None:
        """Both bounds must be in [0.0, 1.0]."""
        for n_suc in [0, 1, 5, 10]:
            ci = confidence_interval(n_successes=n_suc, n_trials=10)
            assert 0.0 <= ci.lower <= 1.0
            assert 0.0 <= ci.upper <= 1.0

    def test_confidence_level_stored_correctly(self) -> None:
        ci = confidence_interval(n_successes=5, n_trials=10, confidence=0.90)
        assert ci.confidence == 0.90

    def test_point_estimate_is_proportion(self) -> None:
        ci = confidence_interval(n_successes=3, n_trials=10)
        assert abs(ci.point_estimate - 0.3) < 1e-6

    def test_n_trials_stored_correctly(self) -> None:
        ci = confidence_interval(n_successes=3, n_trials=10)
        assert ci.n_trials == 10

    def test_raises_on_zero_trials(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            confidence_interval(n_successes=0, n_trials=0)

    def test_raises_on_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            confidence_interval(n_successes=5, n_trials=10, confidence=1.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            confidence_interval(n_successes=5, n_trials=10, confidence=0.0)

    def test_raises_on_successes_exceed_trials(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            confidence_interval(n_successes=11, n_trials=10)

    def test_narrower_ci_for_larger_n(self) -> None:
        """Larger sample should produce a narrower confidence interval."""
        ci_small = confidence_interval(n_successes=5, n_trials=10)
        ci_large = confidence_interval(n_successes=50, n_trials=100)
        width_small = ci_small.upper - ci_small.lower
        width_large = ci_large.upper - ci_large.lower
        assert width_large < width_small

    def test_symmetric_around_half_for_50_percent_rate(self) -> None:
        """At 50% pass rate, CI should be roughly symmetric around 0.5."""
        ci = confidence_interval(n_successes=50, n_trials=100)
        assert abs((ci.lower + ci.upper) / 2 - 0.5) < 0.05


class TestScoreStatistics:
    """Tests for score_variance and score_stddev helpers."""

    def test_variance_empty_list(self) -> None:
        assert score_variance([]) == 0.0

    def test_variance_single_element(self) -> None:
        assert score_variance([0.5]) == 0.0

    def test_variance_known_value(self) -> None:
        """Variance of [0.0, 1.0] = 0.5 (sample variance)."""
        var = score_variance([0.0, 1.0])
        assert abs(var - 0.5) < 1e-9

    def test_stddev_empty_list(self) -> None:
        assert score_stddev([]) == 0.0

    def test_stddev_single_element(self) -> None:
        assert score_stddev([0.5]) == 0.0

    def test_stddev_is_sqrt_variance(self) -> None:
        scores = [0.2, 0.4, 0.6, 0.8, 1.0]
        var = score_variance(scores)
        std = score_stddev(scores)
        assert abs(std - math.sqrt(var)) < 1e-9

    def test_z_score_for_95_percent(self) -> None:
        """Z-score for 95% confidence should be approximately 1.96."""
        z = _z_score_for_confidence(0.95)
        assert abs(z - 1.96) < 0.01

    def test_z_score_for_99_percent(self) -> None:
        """Z-score for 99% confidence should be approximately 2.576."""
        z = _z_score_for_confidence(0.99)
        assert abs(z - 2.576) < 0.01

    def test_z_score_upper_tail_branch(self) -> None:
        """confidence=0.98 maps to p=0.99 which hits the upper-tail branch (p > p_high)."""
        # p = (1 + 0.98) / 2 = 0.99, p_high = 1 - 0.02425 = 0.97575
        # 0.99 > 0.97575 → upper tail branch executes
        z = _z_score_for_confidence(0.98)
        # Z-score for 98% CI should be approximately 2.326
        assert 2.0 < z < 3.0

    def test_z_score_central_region_nonstandard(self) -> None:
        """confidence=0.92 maps to p=0.96 which is in the central region."""
        # p = (1 + 0.92) / 2 = 0.96, p_low=0.02425, p_high=0.97575
        # 0.02425 <= 0.96 <= 0.97575 → central region
        z = _z_score_for_confidence(0.92)
        assert 1.5 < z < 2.5


class TestPassAtKEdgeCases:
    """Additional pass_at_k edge cases for uncovered branches."""

    def test_n_wrong_less_than_k_returns_one(self) -> None:
        """When n_wrong < k but not all passed, the n_wrong < k branch returns 1.0.

        n_correct=4, n_total=5, k=3: n_wrong=1 < k=3 → line 131 returns 1.0.
        """
        result = pass_at_k(n_correct=4, n_total=5, k=3)
        assert result == 1.0

    def test_n_wrong_equals_k_minus_one_returns_one(self) -> None:
        """n_wrong=k-1 still triggers the n_wrong < k branch."""
        result = pass_at_k(n_correct=9, n_total=10, k=5)
        # n_wrong=1, k=5, n_wrong < k → return 1.0
        assert result == 1.0
