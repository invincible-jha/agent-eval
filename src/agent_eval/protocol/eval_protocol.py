"""Pydantic models for the framework-agnostic eval protocol.

Protocol Version: 1.0

The eval protocol defines a standard JSON schema for exchanging evaluation
requests and responses between evaluation frameworks, evaluation runners,
and reporting systems.

Schema
------
EvalRequest:
  - protocol_version: str
  - request_id: str (UUID)
  - agent_id: str
  - input: str
  - expected_output: str | None
  - context: dict (arbitrary metadata)
  - metrics: list[str] (metric names to compute)
  - timeout_seconds: int

EvalResponse:
  - protocol_version: str
  - request_id: str (matches request)
  - agent_output: str
  - metrics: list[EvalMetric]
  - duration_ms: float
  - metadata: dict
"""
from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field, model_validator


CURRENT_PROTOCOL_VERSION: str = "1.0"
SUPPORTED_PROTOCOL_VERSIONS: frozenset[str] = frozenset({"1.0"})


class EvalProtocolVersion(str, Enum):
    """Supported eval protocol versions."""

    V1_0 = "1.0"


class MetricStatus(str, Enum):
    """Outcome status of a computed metric."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIPPED = "skipped"


class EvalMetric(BaseModel):
    """A single computed evaluation metric.

    Attributes
    ----------
    name:
        The metric name (e.g., "faithfulness", "relevance", "safety_score").
    score:
        Numeric score, typically in [0.0, 1.0]. None if status is ERROR.
    status:
        Whether the metric passed, failed, or errored.
    threshold:
        Optional threshold used to determine pass/fail.
    reason:
        Optional human-readable explanation of the score.
    metadata:
        Arbitrary metric-level metadata.
    """

    name: str = Field(..., min_length=1)
    score: float | None = Field(None, ge=0.0, le=1.0)
    status: MetricStatus = MetricStatus.SKIPPED
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    reason: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_score_consistency(self) -> "EvalMetric":
        """Ensure score is present when status is PASS or FAIL."""
        if self.status in (MetricStatus.PASS, MetricStatus.FAIL):
            if self.score is None:
                raise ValueError(
                    f"Metric '{self.name}' has status={self.status.value} "
                    "but no score was provided."
                )
        return self

    model_config = {"frozen": True}


class EvalRequest(BaseModel):
    """A framework-agnostic evaluation request.

    Attributes
    ----------
    protocol_version:
        The protocol version for this request.
    request_id:
        Unique identifier for this request (auto-generated UUID).
    agent_id:
        Identifier of the agent under evaluation.
    input:
        The input prompt or query to evaluate the agent on.
    expected_output:
        Optional expected output for reference-based metrics.
    context:
        Arbitrary context data (e.g., retrieved documents, session state).
    metrics:
        List of metric names to compute. Empty list = use defaults.
    timeout_seconds:
        Maximum seconds to wait for agent response.
    tags:
        Optional tags for filtering and grouping eval results.
    """

    protocol_version: str = Field(default=CURRENT_PROTOCOL_VERSION)
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    agent_id: str = Field(..., min_length=1)
    input: str = Field(..., min_length=1)
    expected_output: str | None = None
    context: dict[str, object] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=30, gt=0, le=600)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_protocol_version(self) -> "EvalRequest":
        """Ensure the protocol version is supported."""
        if self.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise ValueError(
                f"Unsupported protocol version: {self.protocol_version!r}. "
                f"Supported versions: {sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
            )
        return self

    model_config = {"frozen": True}


class EvalResponse(BaseModel):
    """A framework-agnostic evaluation response.

    Attributes
    ----------
    protocol_version:
        The protocol version for this response.
    request_id:
        The request ID this response corresponds to.
    agent_output:
        The raw output produced by the agent.
    metrics:
        List of computed metric results.
    duration_ms:
        Time taken to compute the evaluation in milliseconds.
    error:
        Error message if the evaluation failed at the top level.
    metadata:
        Arbitrary response-level metadata.
    """

    protocol_version: str = Field(default=CURRENT_PROTOCOL_VERSION)
    request_id: str
    agent_output: str
    metrics: list[EvalMetric] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def passed_metrics(self) -> list[EvalMetric]:
        """Metrics with status=PASS."""
        return [m for m in self.metrics if m.status == MetricStatus.PASS]

    @property
    def failed_metrics(self) -> list[EvalMetric]:
        """Metrics with status=FAIL."""
        return [m for m in self.metrics if m.status == MetricStatus.FAIL]

    @property
    def overall_passed(self) -> bool:
        """True if all computed metrics passed (or none were computed)."""
        computed = [m for m in self.metrics if m.status != MetricStatus.SKIPPED]
        if not computed:
            return True
        return all(m.status == MetricStatus.PASS for m in computed)

    @property
    def pass_rate(self) -> float:
        """Fraction of computed metrics that passed."""
        computed = [m for m in self.metrics if m.status != MetricStatus.SKIPPED]
        if not computed:
            return 1.0
        return sum(1 for m in computed if m.status == MetricStatus.PASS) / len(computed)

    def metric_by_name(self, name: str) -> EvalMetric | None:
        """Look up a metric by name.

        Parameters
        ----------
        name:
            The metric name to find.

        Returns
        -------
        EvalMetric | None
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------


def validate_eval_request(data: dict[str, object]) -> EvalRequest:
    """Parse and validate a dict as an EvalRequest.

    Parameters
    ----------
    data:
        Raw dict (e.g., loaded from JSON) to validate.

    Returns
    -------
    EvalRequest
        Validated request model.

    Raises
    ------
    pydantic.ValidationError
        If the data does not conform to the EvalRequest schema.
    """
    return EvalRequest.model_validate(data)


def validate_eval_response(data: dict[str, object]) -> EvalResponse:
    """Parse and validate a dict as an EvalResponse.

    Parameters
    ----------
    data:
        Raw dict (e.g., loaded from JSON) to validate.

    Returns
    -------
    EvalResponse
        Validated response model.

    Raises
    ------
    pydantic.ValidationError
        If the data does not conform to the EvalResponse schema.
    """
    return EvalResponse.model_validate(data)
