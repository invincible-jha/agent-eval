"""Synthetic-to-real distribution gap detector.

Compares numeric feature distributions extracted from synthetic evaluation
datasets against distributions from production traces, using the
Kolmogorov-Smirnov (KS) statistic.

KS statistic source
-------------------
scipy.stats.ks_2samp if scipy is available; otherwise a manual
implementation of the two-sample KS test (D statistic only).

Reference: W.H. Press et al., "Numerical Recipes", Chapter 14.
The KS test is a nonparametric, distribution-free comparison — appropriate
for commodity use without innovation exposure.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

from agent_eval.gap.trace_loader import ProductionTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KS statistic (manual fallback — no scipy required)
# ---------------------------------------------------------------------------

def _ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    """Compute the two-sample Kolmogorov-Smirnov D statistic.

    Parameters
    ----------
    sample_a:
        First sample.
    sample_b:
        Second sample.

    Returns
    -------
    float
        KS D statistic in [0, 1]. 0 = identical, 1 = maximally different.
    """
    if not sample_a or not sample_b:
        return 0.0

    try:
        from scipy.stats import ks_2samp  # type: ignore[import-untyped]

        result = ks_2samp(sample_a, sample_b)
        return float(result.statistic)
    except ImportError:
        pass

    # Manual implementation: empirical CDF comparison
    sorted_a = sorted(sample_a)
    sorted_b = sorted(sample_b)
    n_a = len(sorted_a)
    n_b = len(sorted_b)

    # Merge all unique values
    all_values = sorted(set(sorted_a + sorted_b))

    max_diff = 0.0
    for value in all_values:
        # Empirical CDF at this point
        cdf_a = sum(1 for x in sorted_a if x <= value) / n_a
        cdf_b = sum(1 for x in sorted_b if x <= value) / n_b
        diff = abs(cdf_a - cdf_b)
        if diff > max_diff:
            max_diff = diff

    return max_diff


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class GapSeverity(str, Enum):
    """Severity of the distribution gap detected."""

    NEGLIGIBLE = "negligible"   # KS < 0.1
    LOW = "low"                 # 0.1 <= KS < 0.2
    MEDIUM = "medium"           # 0.2 <= KS < 0.35
    HIGH = "high"               # 0.35 <= KS < 0.5
    CRITICAL = "critical"       # KS >= 0.5


def _severity_from_ks(ks_stat: float) -> GapSeverity:
    if ks_stat < 0.10:
        return GapSeverity.NEGLIGIBLE
    if ks_stat < 0.20:
        return GapSeverity.LOW
    if ks_stat < 0.35:
        return GapSeverity.MEDIUM
    if ks_stat < 0.50:
        return GapSeverity.HIGH
    return GapSeverity.CRITICAL


@dataclass(frozen=True)
class DistributionSample:
    """A named collection of numeric feature values for distribution comparison.

    Attributes
    ----------
    name:
        Human-readable name for this feature (e.g., "input_length").
    values:
        Numeric values extracted from the dataset.
    source_label:
        Label indicating where these values came from ("synthetic" or "production").
    """

    name: str
    values: tuple[float, ...]
    source_label: str = "unknown"

    @classmethod
    def from_list(
        cls,
        name: str,
        values: list[float],
        source_label: str = "unknown",
    ) -> "DistributionSample":
        """Create a DistributionSample from a list.

        Parameters
        ----------
        name:
            Feature name.
        values:
            List of numeric values.
        source_label:
            Source label.

        Returns
        -------
        DistributionSample
        """
        return cls(name=name, values=tuple(values), source_label=source_label)

    @property
    def mean(self) -> float:
        """Mean of the sample values."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        """Standard deviation of the sample values."""
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self.values) / (len(self.values) - 1)
        return math.sqrt(variance)

    @property
    def count(self) -> int:
        """Number of samples."""
        return len(self.values)


@dataclass(frozen=True)
class FeatureGap:
    """Gap analysis result for a single feature.

    Attributes
    ----------
    feature_name:
        The name of the feature analyzed.
    ks_statistic:
        The KS D statistic (0-1).
    severity:
        Severity classification based on the KS statistic.
    synthetic_mean:
        Mean of the synthetic sample.
    production_mean:
        Mean of the production sample.
    synthetic_count:
        Number of synthetic samples.
    production_count:
        Number of production samples.
    recommendation:
        Human-readable suggestion for closing the gap.
    """

    feature_name: str
    ks_statistic: float
    severity: GapSeverity
    synthetic_mean: float
    production_mean: float
    synthetic_count: int
    production_count: int
    recommendation: str = ""


@dataclass
class GapReport:
    """Full gap analysis report comparing synthetic and production distributions.

    Attributes
    ----------
    feature_gaps:
        Per-feature gap analysis results.
    overall_ks:
        Mean KS statistic across all features analyzed.
    overall_severity:
        Severity classification of the overall gap.
    total_synthetic_samples:
        Number of synthetic data points analyzed.
    total_production_traces:
        Number of production traces analyzed.
    summary:
        Human-readable summary of findings.
    """

    feature_gaps: list[FeatureGap] = field(default_factory=list)
    overall_ks: float = 0.0
    overall_severity: GapSeverity = GapSeverity.NEGLIGIBLE
    total_synthetic_samples: int = 0
    total_production_traces: int = 0
    summary: str = ""

    @property
    def critical_features(self) -> list[FeatureGap]:
        """Features with CRITICAL or HIGH severity gaps."""
        return [
            gap for gap in self.feature_gaps
            if gap.severity in (GapSeverity.CRITICAL, GapSeverity.HIGH)
        ]

    @property
    def has_significant_gaps(self) -> bool:
        """True if any feature has MEDIUM or worse gap severity."""
        return any(
            gap.severity not in (GapSeverity.NEGLIGIBLE, GapSeverity.LOW)
            for gap in self.feature_gaps
        )


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _extract_feature_from_traces(
    traces: list[ProductionTrace],
    feature_name: str,
) -> list[float]:
    """Extract a numeric feature from a list of production traces.

    Supported feature names:
    - ``input_length``: character count of input_text
    - ``output_length``: character count of output_text
    - ``tool_call_count``: number of tool calls per trace
    - ``latency_ms``: response latency in milliseconds (traces with None skipped)

    Parameters
    ----------
    traces:
        Production traces to extract from.
    feature_name:
        The feature to extract.

    Returns
    -------
    list[float]
        Extracted numeric values (None values skipped).
    """
    values: list[float] = []
    for trace in traces:
        value: float | None = None
        if feature_name == "input_length":
            value = float(trace.input_length)
        elif feature_name == "output_length":
            value = float(trace.output_length)
        elif feature_name == "tool_call_count":
            value = float(trace.tool_call_count)
        elif feature_name == "latency_ms":
            if trace.latency_ms is not None:
                value = trace.latency_ms
        else:
            # Try metadata fields
            meta_val = trace.metadata.get(feature_name)
            if meta_val is not None:
                try:
                    value = float(meta_val)
                except (ValueError, TypeError):
                    pass

        if value is not None:
            values.append(value)

    return values


# ---------------------------------------------------------------------------
# GapDetector
# ---------------------------------------------------------------------------

class GapDetector:
    """Detects distribution gaps between synthetic eval sets and production traces.

    Usage
    -----
    ::

        detector = GapDetector()
        synthetic_samples = {
            "input_length": DistributionSample.from_list("input_length", [50, 60, 70]),
        }
        report = detector.detect(synthetic_samples, production_traces)
    """

    DEFAULT_FEATURES: list[str] = [
        "input_length",
        "output_length",
        "tool_call_count",
    ]

    def __init__(
        self,
        features: list[str] | None = None,
        min_samples_for_comparison: int = 5,
    ) -> None:
        """Initialise the detector.

        Parameters
        ----------
        features:
            List of feature names to compare. Defaults to
            ``DEFAULT_FEATURES``.
        min_samples_for_comparison:
            Minimum number of samples in both sets before a comparison
            is performed. Pairs with fewer samples produce a LOW-severity
            placeholder gap.
        """
        self.features = features or self.DEFAULT_FEATURES
        self.min_samples = min_samples_for_comparison

    def detect(
        self,
        synthetic_samples: dict[str, DistributionSample],
        production_traces: list[ProductionTrace],
    ) -> GapReport:
        """Run gap analysis and return a GapReport.

        Parameters
        ----------
        synthetic_samples:
            Mapping of feature_name -> DistributionSample from the
            synthetic evaluation dataset.
        production_traces:
            Production traces to compare against.

        Returns
        -------
        GapReport
        """
        feature_gaps: list[FeatureGap] = []

        for feature_name in self.features:
            synthetic_sample = synthetic_samples.get(feature_name)
            if synthetic_sample is None:
                logger.debug("No synthetic sample for feature %r; skipping.", feature_name)
                continue

            production_values = _extract_feature_from_traces(
                production_traces, feature_name
            )

            gap = self._compare_feature(
                feature_name=feature_name,
                synthetic=list(synthetic_sample.values),
                production=production_values,
            )
            feature_gaps.append(gap)

        # Compute overall KS
        if feature_gaps:
            overall_ks = sum(g.ks_statistic for g in feature_gaps) / len(feature_gaps)
        else:
            overall_ks = 0.0

        overall_severity = _severity_from_ks(overall_ks)

        # Count totals
        total_synthetic = max(
            (len(s.values) for s in synthetic_samples.values()),
            default=0,
        )

        summary = self._build_summary(feature_gaps, overall_severity)

        return GapReport(
            feature_gaps=feature_gaps,
            overall_ks=overall_ks,
            overall_severity=overall_severity,
            total_synthetic_samples=total_synthetic,
            total_production_traces=len(production_traces),
            summary=summary,
        )

    def _compare_feature(
        self,
        feature_name: str,
        synthetic: list[float],
        production: list[float],
    ) -> FeatureGap:
        if len(synthetic) < self.min_samples or len(production) < self.min_samples:
            return FeatureGap(
                feature_name=feature_name,
                ks_statistic=0.0,
                severity=GapSeverity.NEGLIGIBLE,
                synthetic_mean=sum(synthetic) / len(synthetic) if synthetic else 0.0,
                production_mean=sum(production) / len(production) if production else 0.0,
                synthetic_count=len(synthetic),
                production_count=len(production),
                recommendation=(
                    f"Insufficient samples for '{feature_name}' comparison "
                    f"(synthetic={len(synthetic)}, production={len(production)}, "
                    f"min={self.min_samples})."
                ),
            )

        ks = _ks_statistic(synthetic, production)
        severity = _severity_from_ks(ks)
        synthetic_mean = sum(synthetic) / len(synthetic)
        production_mean = sum(production) / len(production)

        recommendation = self._build_recommendation(
            feature_name, severity, synthetic_mean, production_mean
        )

        return FeatureGap(
            feature_name=feature_name,
            ks_statistic=ks,
            severity=severity,
            synthetic_mean=synthetic_mean,
            production_mean=production_mean,
            synthetic_count=len(synthetic),
            production_count=len(production),
            recommendation=recommendation,
        )

    def _build_recommendation(
        self,
        feature_name: str,
        severity: GapSeverity,
        synthetic_mean: float,
        production_mean: float,
    ) -> str:
        if severity == GapSeverity.NEGLIGIBLE:
            return f"Feature '{feature_name}' distributions are well-aligned."

        direction = "longer" if production_mean > synthetic_mean else "shorter"
        magnitude = abs(production_mean - synthetic_mean)

        if severity in (GapSeverity.CRITICAL, GapSeverity.HIGH):
            prefix = "URGENT:"
        elif severity == GapSeverity.MEDIUM:
            prefix = "Recommended:"
        else:
            prefix = "Optional:"

        return (
            f"{prefix} Production '{feature_name}' values are {direction} "
            f"(mean diff={magnitude:.1f}). Consider augmenting synthetic eval "
            f"set with {direction} examples to close the gap."
        )

    def _build_summary(
        self,
        feature_gaps: list[FeatureGap],
        overall_severity: GapSeverity,
    ) -> str:
        critical_count = sum(
            1 for g in feature_gaps
            if g.severity in (GapSeverity.CRITICAL, GapSeverity.HIGH)
        )
        return (
            f"Gap analysis complete. Overall severity: {overall_severity.value}. "
            f"{len(feature_gaps)} features analyzed, "
            f"{critical_count} with HIGH/CRITICAL gaps."
        )

    @classmethod
    def from_synthetic_eval_data(
        cls,
        eval_data: list[dict[str, object]],
        features: list[str] | None = None,
    ) -> tuple["GapDetector", dict[str, DistributionSample]]:
        """Build a GapDetector and synthetic samples from a list of eval records.

        Parameters
        ----------
        eval_data:
            List of dicts, each representing one synthetic eval example.
            Recognized keys: ``input``, ``output``, ``tool_calls``.
        features:
            Features to extract. Defaults to DEFAULT_FEATURES.

        Returns
        -------
        tuple[GapDetector, dict[str, DistributionSample]]
        """
        detector = cls(features=features)
        selected_features = features or cls.DEFAULT_FEATURES
        feature_values: dict[str, list[float]] = {f: [] for f in selected_features}

        for record in eval_data:
            if "input_length" in selected_features:
                input_text = str(record.get("input", ""))
                feature_values["input_length"].append(float(len(input_text)))

            if "output_length" in selected_features:
                output_text = str(record.get("output", ""))
                feature_values["output_length"].append(float(len(output_text)))

            if "tool_call_count" in selected_features:
                tool_calls = record.get("tool_calls", [])
                count = len(tool_calls) if isinstance(tool_calls, list) else 0
                feature_values["tool_call_count"].append(float(count))

        samples: dict[str, DistributionSample] = {}
        for feature_name, values in feature_values.items():
            if values:
                samples[feature_name] = DistributionSample.from_list(
                    name=feature_name,
                    values=values,
                    source_label="synthetic",
                )

        return detector, samples
