"""Adversarial robustness tier evaluation for agent-eval.

Provides a 5-tier classification of adversarial robustness, from naive
inputs (T1) through adaptive adversarial scenarios (T5), with a tier
evaluator that scores agent robustness per tier.
"""
from __future__ import annotations

from agent_eval.adversarial.robustness_tiers import (
    AdversarialInput,
    RobustnessTier,
    TierDefinition,
    TIER_DEFINITIONS,
    generate_tier_inputs,
)
from agent_eval.adversarial.tier_evaluator import (
    TierEvalResult,
    TierEvaluator,
    RobustnessReport,
)

__all__ = [
    "AdversarialInput",
    "RobustnessTier",
    "TierDefinition",
    "TIER_DEFINITIONS",
    "generate_tier_inputs",
    "TierEvalResult",
    "TierEvaluator",
    "RobustnessReport",
]
