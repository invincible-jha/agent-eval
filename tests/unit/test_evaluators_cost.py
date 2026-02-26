"""Unit tests for agent_eval.evaluators.cost.

Tests BasicCostEvaluator token estimation, budget enforcement,
model-based USD cost estimation, and metadata overrides.
"""
from __future__ import annotations

import pytest

from agent_eval.core.evaluator import Dimension
from agent_eval.core.exceptions import EvaluatorError
from agent_eval.evaluators.cost import (
    BasicCostEvaluator,
    _estimate_tokens,
)


# ---------------------------------------------------------------------------
# _estimate_tokens helper
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_zero(self) -> None:
        assert _estimate_tokens("") == 0

    def test_whitespace_only_returns_zero(self) -> None:
        assert _estimate_tokens("   ") == 0

    def test_single_word_returns_at_least_one(self) -> None:
        assert _estimate_tokens("hello") >= 1

    def test_token_count_proportional_to_words(self) -> None:
        # 10 words * 1.3 = 13 tokens
        text = " ".join(["word"] * 10)
        assert _estimate_tokens(text) == 13

    def test_minimum_one_for_non_empty(self) -> None:
        # A single very short word
        assert _estimate_tokens("a") >= 1


# ---------------------------------------------------------------------------
# BasicCostEvaluator construction
# ---------------------------------------------------------------------------


class TestBasicCostEvaluatorConstruction:
    def test_default_construction(self) -> None:
        ev = BasicCostEvaluator()
        assert ev.max_tokens == 4096
        assert ev.model is None

    def test_zero_max_tokens_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="max_tokens must be positive"):
            BasicCostEvaluator(max_tokens=0)

    def test_negative_max_tokens_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="max_tokens must be positive"):
            BasicCostEvaluator(max_tokens=-1)

    def test_unknown_model_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="Unknown model"):
            BasicCostEvaluator(model="nonexistent-model-xyz")

    def test_known_model_accepted(self) -> None:
        ev = BasicCostEvaluator(model="gpt-4o")
        assert ev.model == "gpt-4o"

    def test_dimension_property(self) -> None:
        ev = BasicCostEvaluator()
        assert ev.dimension == Dimension.COST

    def test_name_property(self) -> None:
        ev = BasicCostEvaluator()
        assert ev.name == "BasicCostEvaluator"


# ---------------------------------------------------------------------------
# evaluate — token budget enforcement
# ---------------------------------------------------------------------------


class TestCostEvaluatorBudget:
    @pytest.fixture
    def ev(self) -> BasicCostEvaluator:
        return BasicCostEvaluator(max_tokens=100)

    def test_within_budget_passes(self, ev: BasicCostEvaluator) -> None:
        # Small output well within 100-token budget
        result = ev.evaluate("c1", "short answer", None, {})
        assert result.passed is True
        assert result.score > 0.5

    def test_over_budget_fails(self, ev: BasicCostEvaluator) -> None:
        # Many words to exceed 100-token budget
        long_output = " ".join(["word"] * 100)
        result = ev.evaluate("c1", long_output, None, {})
        assert result.passed is False

    def test_score_one_for_zero_tokens(self) -> None:
        ev = BasicCostEvaluator(max_tokens=1000)
        result = ev.evaluate("c1", "", None, {})
        # Empty output => 0 tokens => score close to 1.0
        assert result.score >= 0.5

    def test_reason_contains_token_count(self, ev: BasicCostEvaluator) -> None:
        result = ev.evaluate("c1", "hello world", None, {})
        assert "token" in result.reason.lower()

    def test_raw_value_is_total_token_count(self, ev: BasicCostEvaluator) -> None:
        result = ev.evaluate("c1", "hello world", None, {"input_tokens": 10})
        # raw_value should be total tokens
        assert result.raw_value is not None
        assert result.raw_value > 0


# ---------------------------------------------------------------------------
# evaluate — metadata overrides
# ---------------------------------------------------------------------------


class TestCostEvaluatorMetadataOverrides:
    def test_max_cost_tokens_override_from_metadata(self) -> None:
        ev = BasicCostEvaluator(max_tokens=10)
        # metadata provides a generous budget
        result = ev.evaluate("c1", "hello world", None, {"max_cost_tokens": 10000})
        assert result.passed is True

    def test_input_tokens_override_from_metadata(self) -> None:
        ev = BasicCostEvaluator(max_tokens=100)
        result = ev.evaluate("c1", "short", None, {"input_tokens": 5})
        assert result.raw_value is not None

    def test_output_tokens_override_from_metadata(self) -> None:
        ev = BasicCostEvaluator(max_tokens=100)
        result = ev.evaluate("c1", "any output", None, {"output_tokens": 5})
        assert result.passed is True

    def test_input_text_used_for_input_token_estimation(self) -> None:
        ev = BasicCostEvaluator(max_tokens=100)
        metadata: dict[str, str | int | float | bool] = {
            "input_text": "a short input"
        }
        result = ev.evaluate("c1", "output text", None, metadata)
        assert result.raw_value is not None


# ---------------------------------------------------------------------------
# evaluate — USD cost estimation
# ---------------------------------------------------------------------------


class TestCostEvaluatorUSDEstimation:
    def test_cost_estimate_included_in_reason(self) -> None:
        ev = BasicCostEvaluator(max_tokens=1000, model="gpt-4o")
        result = ev.evaluate("c1", "hello world", None, {"input_tokens": 10})
        assert "Est. cost" in result.reason
        assert "gpt-4o" in result.reason

    def test_no_model_means_no_cost_in_reason(self) -> None:
        ev = BasicCostEvaluator(max_tokens=1000)
        result = ev.evaluate("c1", "hello world", None, {})
        assert "Est. cost" not in result.reason

    def test_all_known_models_accepted(self) -> None:
        known_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-5-sonnet", "claude-3-haiku", "claude-3-opus",
            "claude-sonnet-4", "gemini-1.5-pro", "gemini-1.5-flash",
        ]
        for model in known_models:
            ev = BasicCostEvaluator(model=model)
            assert ev.model == model

    def test_score_clipped_to_zero_at_two_x_budget(self) -> None:
        ev = BasicCostEvaluator(max_tokens=10)
        # Provide exact token counts: 20 input + 10 output = 30 tokens, 3x budget
        result = ev.evaluate("c1", "", None, {"input_tokens": 20, "output_tokens": 10})
        assert result.score == 0.0
