"""Basic latency evaluator for agent-eval.

Measures whether agent response time falls within the allowed threshold.

NOTE: This is a commodity latency evaluator. It performs simple threshold
comparison. It is NOT a percentile-based SLO evaluator, does NOT compute
p50/p95/p99 statistics, and does NOT perform trend analysis or anomaly
detection. Those capabilities are available via the plugin system.
"""
from __future__ import annotations

from agent_eval.core.evaluator import Dimension, DimensionScore, Evaluator
from agent_eval.core.exceptions import EvaluatorError

_DEFAULT_MAX_MS = 5_000


class BasicLatencyEvaluator(Evaluator):
    """Evaluates whether agent response time is within an acceptable threshold.

    The latency value is read from the ``latency_ms`` key in the metadata
    dict, which the runner populates automatically. The threshold is taken
    from (in priority order):
    1. ``max_latency_ms`` key in metadata (set from TestCase or suite default)
    2. The ``max_ms`` parameter passed to this constructor

    NOTE: This is NOT a statistical latency evaluator. It does not compute
    percentiles, standard deviations, or trend lines. It is a simple
    threshold comparison. Advanced latency analysis (p99 SLOs, regression
    detection) is available via the plugin system.

    Parameters
    ----------
    max_ms:
        Default maximum acceptable latency in milliseconds. Used when
        the test case does not specify max_latency_ms. Default: 5000ms.
    """

    def __init__(self, max_ms: int = _DEFAULT_MAX_MS) -> None:
        if max_ms <= 0:
            raise EvaluatorError(
                self.name,
                f"max_ms must be positive, got {max_ms}",
            )
        self.max_ms = max_ms

    @property
    def dimension(self) -> Dimension:
        return Dimension.LATENCY

    @property
    def name(self) -> str:
        return "BasicLatencyEvaluator"

    def evaluate(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        """Score latency of the agent response.

        Parameters
        ----------
        case_id:
            Test case identifier.
        agent_output:
            The agent's output (not used for latency scoring).
        expected_output:
            Not used by this evaluator.
        metadata:
            Must contain ``latency_ms`` (float). May contain
            ``max_latency_ms`` to override the constructor default.

        Returns
        -------
        DimensionScore
            Score of 1.0 if within threshold, 0.0 if exceeded,
            proportional scores for near-threshold values.
        """
        latency_raw = metadata.get("latency_ms", 0.0)
        latency_ms = float(latency_raw) if isinstance(latency_raw, (int, float)) else 0.0

        # Test case threshold takes priority over constructor default
        threshold_raw = metadata.get("max_latency_ms", self.max_ms)
        threshold_ms = float(threshold_raw) if isinstance(threshold_raw, (int, float)) else float(self.max_ms)

        if threshold_ms <= 0:
            threshold_ms = float(self.max_ms)

        if latency_ms <= threshold_ms:
            # Score scales linearly: 1.0 at 0ms, passes down to threshold
            score = 1.0 - (latency_ms / threshold_ms) * 0.1
            score = max(0.9, min(1.0, score))
            passed = True
            reason = f"Latency {latency_ms:.1f}ms within threshold {threshold_ms:.0f}ms"
        else:
            # Exceeded threshold: score decays, but never below 0
            overage_ratio = latency_ms / threshold_ms
            score = max(0.0, 1.0 - (overage_ratio - 1.0))
            score = min(score, 0.0) if overage_ratio >= 2.0 else score
            # Actually: cap at 0.0 when 2x over threshold
            score = 0.0 if overage_ratio >= 2.0 else max(0.0, 1.0 - (overage_ratio - 1.0))
            passed = False
            reason = (
                f"Latency {latency_ms:.1f}ms exceeds threshold {threshold_ms:.0f}ms "
                f"(+{latency_ms - threshold_ms:.1f}ms)"
            )

        return DimensionScore(
            dimension=self.dimension,
            score=round(score, 4),
            passed=passed,
            reason=reason,
            raw_value=latency_ms,
        )
