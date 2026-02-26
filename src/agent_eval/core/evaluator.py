"""Core evaluator abstractions for agent-eval.

Defines the Dimension taxonomy, result dataclasses, and the Evaluator ABC
that all concrete evaluators must implement.

NOTE: This module contains the plugin boundary. The ABCs live here.
Commodity implementations live in agent_eval.evaluators.*
Additional implementations (e.g., advanced consistency metrics and
adversarial evaluation) can register via the plugin system.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Dimension(str, Enum):
    """Quality dimensions along which an agent is evaluated.

    Each dimension is orthogonal. A dimension score of 1.0 means
    perfect performance on that axis; 0.0 means complete failure.

    ACCURACY  -- Does the output match the expected answer?
    LATENCY   -- Did the agent respond within the allowed time?
    COST      -- Did the agent stay within the token/cost budget?
    SAFETY    -- Did the output avoid harmful or disallowed content?
    FORMAT    -- Does the output conform to the required structure?
    CUSTOM    -- Catch-all for user-defined evaluation criteria.
    """

    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    SAFETY = "safety"
    FORMAT = "format"
    CUSTOM = "custom"


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single evaluation dimension.

    Parameters
    ----------
    dimension:
        Which quality dimension this score applies to.
    score:
        Numeric score in [0.0, 1.0]. Higher is better.
    passed:
        Whether the score meets the threshold for this dimension.
    reason:
        Human-readable explanation for the score.
    raw_value:
        The underlying measurement before normalization (e.g., latency in ms,
        token count). None when not applicable.
    """

    dimension: Dimension
    score: float
    passed: bool
    reason: str = ""
    raw_value: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"DimensionScore.score must be in [0.0, 1.0], got {self.score}"
            )


@dataclass
class EvalResult:
    """Complete evaluation result for a single agent invocation.

    One EvalResult is produced per (test_case, run_index) pair.
    The runner aggregates multiple EvalResult instances into an EvalReport.

    Parameters
    ----------
    case_id:
        Identifier of the TestCase that was evaluated.
    run_index:
        Zero-based index when runs_per_case > 1.
    agent_output:
        The raw string output produced by the agent.
    dimension_scores:
        One DimensionScore per evaluator that ran.
    latency_ms:
        Wall-clock time from request to response, in milliseconds.
    error:
        If the agent raised an exception, the string representation.
        None on success.
    metadata:
        Arbitrary key-value pairs from the evaluators or adapter.
    """

    case_id: str
    run_index: int
    agent_output: str
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if ALL dimension scores passed."""
        if not self.dimension_scores:
            return False
        return all(ds.passed for ds in self.dimension_scores)

    @property
    def overall_score(self) -> float:
        """Mean of all dimension scores. Returns 0.0 when no scores present."""
        if not self.dimension_scores:
            return 0.0
        return sum(ds.score for ds in self.dimension_scores) / len(self.dimension_scores)

    def score_for(self, dimension: Dimension) -> DimensionScore | None:
        """Return the DimensionScore for a specific dimension, or None."""
        for ds in self.dimension_scores:
            if ds.dimension == dimension:
                return ds
        return None


class Evaluator(ABC):
    """Abstract base class for all evaluators.

    An Evaluator receives an agent output and test case metadata and
    produces a DimensionScore. Evaluators are stateless: all context
    needed for evaluation is passed through the method parameters.

    Implementing subclasses must:
    1. Override ``dimension`` to return the Dimension they measure.
    2. Implement ``evaluate()`` to return a DimensionScore.

    The ``evaluate_batch()`` default implementation loops over cases.
    Override it for evaluators that benefit from batching (e.g., LLM judges
    that can process multiple outputs in a single API call).

    NOTE: This is the plugin boundary. Additional evaluators can be added
    by subclassing this ABC and registering via the plugin system. This
    package ships commodity implementations in agent_eval.evaluators.*.
    """

    @property
    @abstractmethod
    def dimension(self) -> Dimension:
        """The quality dimension this evaluator measures."""

    @property
    def name(self) -> str:
        """Human-readable name for this evaluator. Defaults to class name."""
        return type(self).__name__

    @abstractmethod
    def evaluate(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        """Evaluate a single agent output.

        Parameters
        ----------
        case_id:
            The test case identifier.
        agent_output:
            The raw output produced by the agent under test.
        expected_output:
            The reference answer from the test case, if any.
        metadata:
            Arbitrary context from the test case (tools used, latency, etc.).

        Returns
        -------
        DimensionScore
            Score in [0.0, 1.0] with pass/fail determination.
        """

    def evaluate_batch(
        self,
        cases: list[
            tuple[str, str, str | None, dict[str, str | int | float | bool]]
        ],
    ) -> list[DimensionScore]:
        """Evaluate a batch of agent outputs.

        The default implementation calls ``evaluate()`` sequentially.
        Override this for evaluators that can process batches more
        efficiently (e.g., a single LLM prompt for multiple outputs).

        Parameters
        ----------
        cases:
            List of (case_id, agent_output, expected_output, metadata) tuples.

        Returns
        -------
        list[DimensionScore]
            One DimensionScore per input case, in the same order.
        """
        return [
            self.evaluate(case_id, agent_output, expected_output, metadata)
            for case_id, agent_output, expected_output, metadata in cases
        ]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dimension={self.dimension.value!r})"


def measure_latency(start_time: float) -> float:
    """Compute elapsed milliseconds since start_time (from time.perf_counter).

    Parameters
    ----------
    start_time:
        Value returned by ``time.perf_counter()`` before the operation.

    Returns
    -------
    float
        Elapsed time in milliseconds.
    """
    return (time.perf_counter() - start_time) * 1000.0
