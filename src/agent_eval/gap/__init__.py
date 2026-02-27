"""Synthetic-to-real gap detection for agent-eval.

Compares distributions of synthetic evaluation data against production
trace distributions to surface coverage blind spots.
"""
from __future__ import annotations

from agent_eval.gap.gap_detector import (
    DistributionSample,
    GapDetector,
    GapReport,
    GapSeverity,
    FeatureGap,
)
from agent_eval.gap.trace_loader import (
    ProductionTrace,
    TraceLoader,
    load_traces_from_jsonl,
)

__all__ = [
    "DistributionSample",
    "GapDetector",
    "GapReport",
    "GapSeverity",
    "FeatureGap",
    "ProductionTrace",
    "TraceLoader",
    "load_traces_from_jsonl",
]
