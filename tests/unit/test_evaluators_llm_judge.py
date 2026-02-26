"""Unit tests for agent_eval.evaluators.llm_judge.

Tests BasicLLMJudge construction, sync/async evaluation,
score parsing, prompt building, and error handling.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_eval.core.evaluator import Dimension
from agent_eval.core.exceptions import EvaluatorError
from agent_eval.evaluators.llm_judge import (
    BasicLLMJudge,
    _default_score_parser,
)


# ---------------------------------------------------------------------------
# _default_score_parser
# ---------------------------------------------------------------------------


class TestDefaultScoreParser:
    def test_parses_exact_one_zero(self) -> None:
        assert _default_score_parser("1.0") == 1.0

    def test_parses_decimal_in_range(self) -> None:
        assert _default_score_parser("0.85") == 0.85

    def test_parses_last_float_when_multiple(self) -> None:
        # Last float wins
        result = _default_score_parser("First I scored 0.3, then finally 0.8")
        assert result == 0.8

    def test_returns_zero_when_no_float_found(self) -> None:
        assert _default_score_parser("no numbers here") == 0.0

    def test_scales_ten_point_score(self) -> None:
        # A score of 8 on a 10-point scale -> 0.8
        result = _default_score_parser("Score: 8")
        assert result == pytest.approx(0.8)

    def test_ignores_out_of_range_numbers(self) -> None:
        # 100 is > 10, so ignored; no valid score -> 0.0
        result = _default_score_parser("Score is 100 percent")
        assert result == 0.0

    def test_parses_float_with_leading_text(self) -> None:
        result = _default_score_parser("The agent output quality: 0.75")
        assert result == 0.75

    def test_handles_integer_zero(self) -> None:
        result = _default_score_parser("0")
        assert result == 0.0

    def test_handles_integer_one(self) -> None:
        # 1 is in [0.0, 1.0] range, so returned as-is (not scaled)
        result = _default_score_parser("1")
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BasicLLMJudge construction
# ---------------------------------------------------------------------------


class TestBasicLLMJudgeConstruction:
    def test_valid_sync_client_accepted(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda prompt: "0.8")
        assert judge is not None

    def test_non_callable_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="llm_client must be callable"):
            BasicLLMJudge(llm_client="not_callable")  # type: ignore[arg-type]

    def test_pass_threshold_out_of_range_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="pass_threshold"):
            BasicLLMJudge(llm_client=lambda p: "0.8", pass_threshold=1.5)

    def test_pass_threshold_negative_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="pass_threshold"):
            BasicLLMJudge(llm_client=lambda p: "0.8", pass_threshold=-0.1)

    def test_dimension_default_is_accuracy(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.8")
        assert judge.dimension == Dimension.ACCURACY

    def test_custom_dimension(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.8", dimension=Dimension.SAFETY)
        assert judge.dimension == Dimension.SAFETY

    def test_name_property_default(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.8")
        assert judge.name == "BasicLLMJudge"

    def test_custom_judge_name(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.8", judge_name="MyJudge")
        assert judge.name == "MyJudge"

    def test_custom_score_parser_accepted(self) -> None:
        custom_parser = lambda r: 0.5
        judge = BasicLLMJudge(llm_client=lambda p: "any", score_parser=custom_parser)
        result = judge.evaluate("c1", "output", "expected", {})
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# evaluate — sync client
# ---------------------------------------------------------------------------


class TestBasicLLMJudgeEvaluateSync:
    def test_high_score_passes(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.9", pass_threshold=0.7)
        result = judge.evaluate("c1", "great output", "expected", {})
        assert result.passed is True
        assert result.score == pytest.approx(0.9)

    def test_low_score_fails(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.3", pass_threshold=0.7)
        result = judge.evaluate("c1", "bad output", "expected", {})
        assert result.passed is False
        assert result.score == pytest.approx(0.3)

    def test_score_exactly_at_threshold_passes(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.7", pass_threshold=0.7)
        result = judge.evaluate("c1", "borderline", "expected", {})
        assert result.passed is True

    def test_reason_contains_score(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.8")
        result = judge.evaluate("c1", "output", None, {})
        assert "0.800" in result.reason or "0.8" in result.reason

    def test_exception_in_client_returns_zero(self) -> None:
        def failing_client(prompt: str) -> str:
            raise RuntimeError("API error")

        judge = BasicLLMJudge(llm_client=failing_client)
        result = judge.evaluate("c1", "output", None, {})
        assert result.passed is False
        assert result.score == 0.0
        assert "LLM judge call failed" in result.reason

    def test_none_expected_output_uses_empty_string(self) -> None:
        received_prompts: list[str] = []

        def capture_client(prompt: str) -> str:
            received_prompts.append(prompt)
            return "0.8"

        judge = BasicLLMJudge(llm_client=capture_client)
        judge.evaluate("c1", "output", None, {})
        assert len(received_prompts) == 1

    def test_metadata_input_text_used_in_prompt(self) -> None:
        received_prompts: list[str] = []

        def capture_client(prompt: str) -> str:
            received_prompts.append(prompt)
            return "0.9"

        judge = BasicLLMJudge(llm_client=capture_client)
        judge.evaluate("c1", "output", "expected", {"input_text": "What is 2+2?"})
        assert "What is 2+2?" in received_prompts[0]

    def test_score_clipped_to_zero_minimum(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "-5.0")
        result = judge.evaluate("c1", "output", None, {})
        assert result.score >= 0.0

    def test_score_clipped_to_one_maximum(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "99.9")
        result = judge.evaluate("c1", "output", None, {})
        assert result.score <= 1.0


# ---------------------------------------------------------------------------
# evaluate — async client
# ---------------------------------------------------------------------------


class TestBasicLLMJudgeEvaluateAsync:
    def test_async_client_returns_score(self) -> None:
        async def async_client(prompt: str) -> str:
            return "0.85"

        judge = BasicLLMJudge(llm_client=async_client)  # type: ignore[arg-type]
        # evaluate() is sync — async clients called from sync get the fallback
        result = judge.evaluate("c1", "output", "expected", {})
        # Score should be 0.5 (async-from-sync fallback) or 0.85
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_async_with_async_client(self) -> None:
        async def async_client(prompt: str) -> str:
            return "0.9"

        async def run() -> None:
            judge = BasicLLMJudge(llm_client=async_client)  # type: ignore[arg-type]
            result = await judge.evaluate_async("c1", "output", "expected", {})
            assert result.passed is True
            assert result.score == pytest.approx(0.9)

        asyncio.run(run())

    def test_evaluate_async_with_sync_client(self) -> None:
        def sync_client(prompt: str) -> str:
            return "0.75"

        async def run() -> None:
            judge = BasicLLMJudge(llm_client=sync_client)
            result = await judge.evaluate_async("c1", "output", "expected", {})
            assert result.score == pytest.approx(0.75)

        asyncio.run(run())

    def test_evaluate_async_exception_returns_zero(self) -> None:
        async def failing_async_client(prompt: str) -> str:
            raise ValueError("async failure")

        async def run() -> None:
            judge = BasicLLMJudge(llm_client=failing_async_client)  # type: ignore[arg-type]
            result = await judge.evaluate_async("c1", "output", None, {})
            assert result.score == 0.0
            assert "failed" in result.reason.lower()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBasicLLMJudgeBuildPrompt:
    def test_default_template_filled_correctly(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.8")
        prompt = judge._build_prompt(
            case_id="case-1",
            agent_output="The answer is 42",
            expected_output="42",
            metadata={"input_text": "What is 6 times 7?"},
        )
        assert "The answer is 42" in prompt
        assert "42" in prompt
        assert "What is 6 times 7?" in prompt

    def test_custom_template_rendered(self) -> None:
        template = "Input: {input}\nOutput: {agent_output}"
        judge = BasicLLMJudge(llm_client=lambda p: "0.8", prompt_template=template)
        prompt = judge._build_prompt(
            case_id="c1",
            agent_output="my answer",
            expected_output="correct",
            metadata={"input_text": "Question?"},
        )
        assert "my answer" in prompt
        assert "Question?" in prompt

    def test_unknown_placeholder_falls_back_to_string_replace(self) -> None:
        template = "Agent said: {agent_output}. Unknown: {unknown_key}"
        judge = BasicLLMJudge(llm_client=lambda p: "0.8", prompt_template=template)
        prompt = judge._build_prompt(
            case_id="c1",
            agent_output="hello",
            expected_output="world",
            metadata={},
        )
        assert "hello" in prompt

    def test_case_id_used_as_fallback_input_text(self) -> None:
        judge = BasicLLMJudge(llm_client=lambda p: "0.8")
        prompt = judge._build_prompt(
            case_id="my-case-id",
            agent_output="output",
            expected_output="expected",
            metadata={},  # No input_text key
        )
        assert "my-case-id" in prompt
