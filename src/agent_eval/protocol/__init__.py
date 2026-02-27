"""Framework-agnostic evaluation protocol for agent-eval.

Defines a JSON-based protocol for exchanging evaluation requests and
results between evaluation frameworks, with importers for DeepEval
and Ragas format compatibility.
"""
from __future__ import annotations

from agent_eval.protocol.eval_protocol import (
    EvalMetric,
    EvalProtocolVersion,
    EvalRequest,
    EvalResponse,
    MetricStatus,
    validate_eval_request,
    validate_eval_response,
)
from agent_eval.protocol.importers import (
    DeepEvalImporter,
    RagasImporter,
    ProtocolImporter,
)

__all__ = [
    "EvalMetric",
    "EvalProtocolVersion",
    "EvalRequest",
    "EvalResponse",
    "MetricStatus",
    "validate_eval_request",
    "validate_eval_response",
    "DeepEvalImporter",
    "RagasImporter",
    "ProtocolImporter",
]
