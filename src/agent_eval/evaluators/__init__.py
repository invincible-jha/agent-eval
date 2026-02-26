"""Built-in evaluator implementations for agent-eval.

All commodity evaluators are exported from this package.
Custom evaluators are loaded via the plugin system.

Evaluator registry maps type strings (used in eval.yaml) to classes.
"""
from __future__ import annotations

from agent_eval.evaluators.accuracy import BasicAccuracyEvaluator
from agent_eval.evaluators.cost import BasicCostEvaluator
from agent_eval.evaluators.format import BasicFormatEvaluator
from agent_eval.evaluators.latency import BasicLatencyEvaluator
from agent_eval.evaluators.llm_judge import BasicLLMJudge
from agent_eval.evaluators.safety import BasicSafetyEvaluator

# Registry mapping type strings from eval.yaml to evaluator classes.
# Keys match the "type" field in EvaluatorConfig.
EVALUATOR_REGISTRY: dict[str, type] = {
    "accuracy": BasicAccuracyEvaluator,
    "basic_accuracy": BasicAccuracyEvaluator,
    "latency": BasicLatencyEvaluator,
    "basic_latency": BasicLatencyEvaluator,
    "cost": BasicCostEvaluator,
    "basic_cost": BasicCostEvaluator,
    "safety": BasicSafetyEvaluator,
    "basic_safety": BasicSafetyEvaluator,
    "format": BasicFormatEvaluator,
    "basic_format": BasicFormatEvaluator,
    "llm_judge": BasicLLMJudge,
    "basic_llm_judge": BasicLLMJudge,
}

__all__ = [
    "BasicAccuracyEvaluator",
    "BasicLatencyEvaluator",
    "BasicCostEvaluator",
    "BasicSafetyEvaluator",
    "BasicFormatEvaluator",
    "BasicLLMJudge",
    "EVALUATOR_REGISTRY",
]
