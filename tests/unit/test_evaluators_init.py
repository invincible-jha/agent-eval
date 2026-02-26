"""Unit tests for agent_eval.evaluators.__init__.

Verifies the EVALUATOR_REGISTRY contains all expected keys and
that all registered classes can be imported.
"""
from __future__ import annotations

import pytest

from agent_eval.evaluators import (
    EVALUATOR_REGISTRY,
    BasicAccuracyEvaluator,
    BasicCostEvaluator,
    BasicFormatEvaluator,
    BasicLatencyEvaluator,
    BasicLLMJudge,
    BasicSafetyEvaluator,
)


class TestEvaluatorRegistry:
    def test_registry_is_dict(self) -> None:
        assert isinstance(EVALUATOR_REGISTRY, dict)

    def test_registry_not_empty(self) -> None:
        assert len(EVALUATOR_REGISTRY) > 0

    def test_accuracy_registered_under_multiple_keys(self) -> None:
        assert "accuracy" in EVALUATOR_REGISTRY
        assert "basic_accuracy" in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY["accuracy"] is BasicAccuracyEvaluator

    def test_latency_registered(self) -> None:
        assert "latency" in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY["latency"] is BasicLatencyEvaluator

    def test_cost_registered(self) -> None:
        assert "cost" in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY["cost"] is BasicCostEvaluator

    def test_safety_registered(self) -> None:
        assert "safety" in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY["safety"] is BasicSafetyEvaluator

    def test_format_registered(self) -> None:
        assert "format" in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY["format"] is BasicFormatEvaluator

    def test_llm_judge_registered(self) -> None:
        assert "llm_judge" in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY["llm_judge"] is BasicLLMJudge

    def test_all_registry_values_are_classes(self) -> None:
        for key, cls in EVALUATOR_REGISTRY.items():
            assert isinstance(cls, type), f"Registry entry {key!r} is not a class"

    def test_all_classes_importable(self) -> None:
        all_classes = [
            BasicAccuracyEvaluator,
            BasicCostEvaluator,
            BasicFormatEvaluator,
            BasicLatencyEvaluator,
            BasicLLMJudge,
            BasicSafetyEvaluator,
        ]
        for cls in all_classes:
            assert cls is not None
