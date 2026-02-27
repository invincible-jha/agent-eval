"""Statistical metrics for agent evaluation reliability.

Implements pass@k from the HumanEval paper and Wilson score confidence
intervals. These metrics quantify reliability for stochastic agents where
a single evaluation run is insufficient to draw conclusions.

References
----------
- Chen et al. (2021). "Evaluating Large Language Models Trained on Code."
  https://arxiv.org/abs/2107.03374
- Wilson, E.B. (1927). "Probable inference, the law of succession, and
  statistical inference." Journal of the American Statistical Association.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class PassAtKResult:
    """Computed pass@k value for a specific k.

    Parameters
    ----------
    k:
        The number of attempts considered in this metric.
    value:
        Probability estimate in [0.0, 1.0]. Higher means more reliable.
    n_total:
        Total number of runs used in the computation.
    n_correct:
        Number of runs that passed (correct).
    """

    k: int
    value: float
    n_total: int
    n_correct: int


@dataclass(frozen=True)
class ConfidenceInterval:
    """A statistical confidence interval for a proportion.

    Parameters
    ----------
    lower:
        Lower bound of the interval, in [0.0, 1.0].
    upper:
        Upper bound of the interval, in [0.0, 1.0].
    confidence:
        The confidence level, e.g. 0.95 for a 95% CI.
    point_estimate:
        The raw proportion (successes / trials).
    n_trials:
        Number of trials used to compute this interval.
    """

    lower: float
    upper: float
    confidence: float
    point_estimate: float
    n_trials: int


def pass_at_k(n_correct: int, n_total: int, k: int) -> float:
    """Compute pass@k using the unbiased estimator from Chen et al. (2021).

    The formula avoids enumerating all possible program sets:

        pass@k = 1 - C(n - c, k) / C(n, k)

    where n = total runs, c = correct runs, k = number of attempts.

    Parameters
    ----------
    n_correct:
        Number of runs that produced a passing result.
    n_total:
        Total number of runs executed (n >= k required).
    k:
        Number of code samples to generate per problem.

    Returns
    -------
    float
        Estimated probability in [0.0, 1.0] that at least one of k
        attempts passes.

    Raises
    ------
    ValueError
        If n_total < 1, n_correct < 0, n_correct > n_total, or k < 1.

    Notes
    -----
    When k > n_total, returns 1.0 if n_correct > 0 else 0.0. This is a
    conservative estimate rather than raising an error, matching common
    use-cases where you evaluate k > n.
    """
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1, got {n_total}")
    if n_correct < 0:
        raise ValueError(f"n_correct must be >= 0, got {n_correct}")
    if n_correct > n_total:
        raise ValueError(
            f"n_correct ({n_correct}) cannot exceed n_total ({n_total})"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # All runs passed — pass@k is 1.0 for any k
    if n_correct == n_total:
        return 1.0

    # All runs failed — pass@k is 0.0 for any k
    if n_correct == 0:
        return 0.0

    # k > n: we cannot apply the formula — return 1.0 if any pass, else 0.0
    if k > n_total:
        return 1.0 if n_correct > 0 else 0.0

    # Standard formula: 1 - C(n-c, k) / C(n, k)
    # Computed in log-space to avoid overflow with large n
    n_wrong = n_total - n_correct
    if n_wrong < k:
        # Not enough failing runs to fill k slots → must pass at least once
        return 1.0

    # Use math.comb which is exact for integers and handles large values
    numerator = math.comb(n_wrong, k)
    denominator = math.comb(n_total, k)

    if denominator == 0:
        return 0.0

    return 1.0 - (numerator / denominator)


def pass_at_k_result(n_correct: int, n_total: int, k: int) -> PassAtKResult:
    """Compute pass@k and return a structured result object.

    Parameters
    ----------
    n_correct:
        Number of passing runs.
    n_total:
        Total number of runs.
    k:
        The k value for pass@k.

    Returns
    -------
    PassAtKResult
    """
    value = pass_at_k(n_correct=n_correct, n_total=n_total, k=k)
    return PassAtKResult(
        k=k,
        value=value,
        n_total=n_total,
        n_correct=n_correct,
    )


def confidence_interval(
    n_successes: int,
    n_trials: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Compute a Wilson score confidence interval for a proportion.

    The Wilson interval is preferred over the normal approximation
    (Wald interval) because it performs well for small samples and
    extreme proportions (near 0 or 1).

    Parameters
    ----------
    n_successes:
        Number of successful trials.
    n_trials:
        Total number of trials.
    confidence:
        Confidence level, e.g. 0.95 for a 95% CI. Must be in (0, 1).

    Returns
    -------
    ConfidenceInterval
        Lower and upper bounds of the Wilson score interval.

    Raises
    ------
    ValueError
        If n_trials < 1 or confidence is not in (0, 1).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_successes < 0 or n_successes > n_trials:
        raise ValueError(
            f"n_successes ({n_successes}) must be in [0, {n_trials}]"
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    proportion = n_successes / n_trials

    # z-score for the desired confidence level (two-tailed)
    alpha = 1.0 - confidence
    # Using the standard normal quantile via the inverse error function
    # scipy.stats.norm.ppf(1 - alpha/2) — approximated with math.erfinv
    z = _z_score_for_confidence(confidence)

    z_squared = z * z
    n = n_trials
    p_hat = proportion

    # Wilson score formula
    center = (p_hat + z_squared / (2 * n)) / (1 + z_squared / n)
    margin = (z / (1 + z_squared / n)) * math.sqrt(
        p_hat * (1 - p_hat) / n + z_squared / (4 * n * n)
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return ConfidenceInterval(
        lower=round(lower, 6),
        upper=round(upper, 6),
        confidence=confidence,
        point_estimate=round(proportion, 6),
        n_trials=n_trials,
    )


def _z_score_for_confidence(confidence: float) -> float:
    """Return the z-score for a given two-tailed confidence level.

    Uses a lookup table for common confidence levels (0.90, 0.95, 0.99)
    and a rational approximation for other values. This avoids requiring
    scipy or math.erfinv (which is only available in Python >= 3.12).

    Parameters
    ----------
    confidence:
        Confidence level in (0, 1), e.g. 0.95.

    Returns
    -------
    float
        The z-score such that P(-z <= Z <= z) = confidence.
    """
    # Fast path: common confidence levels
    _COMMON_Z: dict[float, float] = {
        0.80: 1.2816,
        0.85: 1.4395,
        0.90: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758,
        0.999: 3.2905,
    }
    if confidence in _COMMON_Z:
        return _COMMON_Z[confidence]

    # General case: rational approximation of the normal quantile function
    # Based on Peter Acklam's algorithm (max error < 1.15e-9)
    # https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    # p = P(Z <= z) = (1 + confidence) / 2  (two-tailed → one-tailed)
    p = (1.0 + confidence) / 2.0

    # Rational approximation coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p_low <= p <= p_high:
        # Rational approximation for central region
        q = math.sqrt(-2.0 * math.log(0.5 * (1.0 - p) if p > 0.5 else 0.5 * (1.0 + p - 1.0 + p)))
        # Actually use the standard formula for interior:
        q = p - 0.5
        r = q * q
        z = (q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) /
             (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    elif p < p_low:
        # Lower tail
        q = math.sqrt(-2.0 * math.log(p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        # Upper tail
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    return abs(z)


def score_variance(scores: list[float]) -> float:
    """Compute variance of a list of scores.

    Parameters
    ----------
    scores:
        List of float scores, each in [0.0, 1.0].

    Returns
    -------
    float
        Sample variance, or 0.0 for fewer than 2 scores.
    """
    if len(scores) < 2:
        return 0.0
    return statistics.variance(scores)


def score_stddev(scores: list[float]) -> float:
    """Compute standard deviation of a list of scores.

    Parameters
    ----------
    scores:
        List of float scores, each in [0.0, 1.0].

    Returns
    -------
    float
        Sample standard deviation, or 0.0 for fewer than 2 scores.
    """
    if len(scores) < 2:
        return 0.0
    return statistics.stdev(scores)
