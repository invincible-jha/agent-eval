"""Leaderboard submission model with composite scoring.

A LeaderboardSubmission captures all metrics produced by a single evaluation
run (one agent version, one benchmark) and computes a composite quality score
from weighted sub-scores.

Default weight scheme (sums to 1.0)
------------------------------------
accuracy      0.30
safety        0.25
cost_efficiency 0.15
consistency   0.15
security      0.15

Example
-------
>>> from agent_eval.leaderboard.submission import LeaderboardSubmission
>>> sub = LeaderboardSubmission(
...     agent_name="sentinel-v2",
...     agent_version="2.1.0",
...     framework="langchain",
...     model="gpt-4o",
...     submitter="team-red",
...     accuracy_score=0.90,
...     safety_score=0.95,
...     cost_efficiency=0.80,
...     latency_p95_ms=800.0,
...     consistency_score=0.85,
...     security_score=0.90,
...     benchmark_name="qa_basic",
...     benchmark_version="1.0",
...     num_runs=5,
...     total_tokens=20000,
...     total_cost_usd=0.24,
... )
>>> round(sub.composite_score, 4)
0.895
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------


class CompositeWeights(BaseModel):
    """Weights used to compute the composite leaderboard score.

    All weights must be non-negative. They are normalised internally so
    that they sum to 1.0, which means callers may pass raw importance values
    (e.g. ``accuracy=3, safety=2.5``) rather than pre-normalised fractions.

    Parameters
    ----------
    accuracy:
        Weight for the accuracy dimension.
    safety:
        Weight for the safety dimension.
    cost_efficiency:
        Weight for the cost-efficiency dimension.
    consistency:
        Weight for the consistency dimension.
    security:
        Weight for the security dimension.
    """

    accuracy: Annotated[float, Field(ge=0.0)] = 0.30
    safety: Annotated[float, Field(ge=0.0)] = 0.25
    cost_efficiency: Annotated[float, Field(ge=0.0)] = 0.15
    consistency: Annotated[float, Field(ge=0.0)] = 0.15
    security: Annotated[float, Field(ge=0.0)] = 0.15

    @model_validator(mode="after")
    def _weights_must_be_positive_sum(self) -> "CompositeWeights":
        """Ensure that the total weight is strictly positive."""
        total = (
            self.accuracy
            + self.safety
            + self.cost_efficiency
            + self.consistency
            + self.security
        )
        if total <= 0.0:
            raise ValueError("Sum of all CompositeWeights must be > 0.0")
        return self

    def normalised(self) -> "CompositeWeights":
        """Return a new CompositeWeights instance with values normalised to sum 1.0.

        Returns
        -------
        CompositeWeights
            A copy of these weights scaled so that the total equals exactly 1.0.
        """
        total = (
            self.accuracy
            + self.safety
            + self.cost_efficiency
            + self.consistency
            + self.security
        )
        return CompositeWeights(
            accuracy=self.accuracy / total,
            safety=self.safety / total,
            cost_efficiency=self.cost_efficiency / total,
            consistency=self.consistency / total,
            security=self.security / total,
        )


# ---------------------------------------------------------------------------
# Default weight singleton
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: CompositeWeights = CompositeWeights()


# ---------------------------------------------------------------------------
# Submission model
# ---------------------------------------------------------------------------


class LeaderboardSubmission(BaseModel):
    """A single evaluation run submitted to the leaderboard.

    Each submission records the agent identity, benchmark provenance,
    per-dimension scores, resource consumption, and a composite score
    that determines the leaderboard ranking.

    Parameters
    ----------
    agent_name:
        Human-readable name of the agent (e.g. ``"sentinel-v2"``).
    agent_version:
        SemVer or arbitrary version string (e.g. ``"2.1.0"``).
    framework:
        Orchestration framework used (e.g. ``"langchain"``, ``"crewai"``).
    model:
        Underlying language model (e.g. ``"gpt-4o"``).
    submitter:
        Name of the person or team that submitted the results.
    submitted_at:
        UTC datetime of submission. Defaults to the current UTC time.
    accuracy_score:
        Fraction of benchmark cases answered correctly. Range [0.0, 1.0].
    safety_score:
        Safety evaluation score. Range [0.0, 1.0].
    cost_efficiency:
        Normalised cost-efficiency metric. Range [0.0, 1.0].
    latency_p95_ms:
        95th-percentile end-to-end latency in milliseconds.
    consistency_score:
        Score measuring output consistency across repeated runs. Range [0.0, 1.0].
    security_score:
        Security evaluation score. Range [0.0, 1.0].
    benchmark_name:
        Name of the benchmark suite used (e.g. ``"qa_basic"``).
    benchmark_version:
        Version of the benchmark suite (e.g. ``"1.0"``).
    num_runs:
        Total number of evaluation runs executed.
    total_tokens:
        Aggregate token usage across all runs.
    total_cost_usd:
        Aggregate monetary cost across all runs, in US dollars.
    composite_score:
        Weighted composite quality score. Computed automatically from the
        five sub-scores using ``DEFAULT_WEIGHTS`` unless overridden via
        ``compute_composite()``.
    """

    # --- Agent identity -------------------------------------------------------
    agent_name: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    model: str = Field(min_length=1)
    submitter: str = Field(min_length=1)
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # --- Per-dimension scores -------------------------------------------------
    accuracy_score: Annotated[float, Field(ge=0.0, le=1.0)]
    safety_score: Annotated[float, Field(ge=0.0, le=1.0)]
    cost_efficiency: Annotated[float, Field(ge=0.0, le=1.0)]
    latency_p95_ms: Annotated[float, Field(ge=0.0)]
    consistency_score: Annotated[float, Field(ge=0.0, le=1.0)]
    security_score: Annotated[float, Field(ge=0.0, le=1.0)]

    # --- Benchmark provenance -------------------------------------------------
    benchmark_name: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    num_runs: Annotated[int, Field(ge=1)]
    total_tokens: Annotated[int, Field(ge=0)]
    total_cost_usd: Annotated[float, Field(ge=0.0)]

    # --- Computed composite score ---------------------------------------------
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _auto_compute_composite(self) -> "LeaderboardSubmission":
        """Compute composite score automatically after field validation."""
        if self.composite_score == 0.0:
            self.composite_score = self._weighted_composite(DEFAULT_WEIGHTS)
        return self

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def compute_composite(
        self,
        weights: CompositeWeights | None = None,
    ) -> float:
        """Compute and store the composite score using the given weights.

        The resulting score is stored back on ``self.composite_score`` and
        also returned for convenience.

        Parameters
        ----------
        weights:
            Weight configuration. Weights are normalised before use so
            raw importance values are acceptable. Defaults to
            :data:`DEFAULT_WEIGHTS` when ``None``.

        Returns
        -------
        float
            Composite score in [0.0, 1.0].

        Example
        -------
        >>> sub.compute_composite(CompositeWeights(accuracy=0.5, safety=0.5,
        ...                                        cost_efficiency=0.0,
        ...                                        consistency=0.0, security=0.0))
        0.925
        """
        resolved_weights = weights if weights is not None else DEFAULT_WEIGHTS
        score = self._weighted_composite(resolved_weights)
        self.composite_score = score
        return score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_composite(self, weights: CompositeWeights) -> float:
        """Compute the weighted composite without mutating the model.

        Parameters
        ----------
        weights:
            Raw or pre-normalised weights; normalisation is applied here.

        Returns
        -------
        float
            Weighted composite score clamped to [0.0, 1.0].
        """
        norm = weights.normalised()
        raw = (
            norm.accuracy * self.accuracy_score
            + norm.safety * self.safety_score
            + norm.cost_efficiency * self.cost_efficiency
            + norm.consistency * self.consistency_score
            + norm.security * self.security_score
        )
        return max(0.0, min(1.0, raw))
