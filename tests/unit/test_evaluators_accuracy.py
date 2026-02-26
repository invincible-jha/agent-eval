"""Unit tests for agent_eval.evaluators.accuracy.

Tests BasicAccuracyEvaluator across all four comparison modes,
validation errors, and the _jaccard_similarity helper.
"""
from __future__ import annotations

import pytest

from agent_eval.core.evaluator import Dimension
from agent_eval.core.exceptions import EvaluatorError
from agent_eval.evaluators.accuracy import (
    BasicAccuracyEvaluator,
    _jaccard_similarity,
)


# ---------------------------------------------------------------------------
# _jaccard_similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_strings_return_one(self) -> None:
        assert _jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different_strings_return_zero(self) -> None:
        assert _jaccard_similarity("apple banana", "cherry date") == 0.0

    def test_partial_overlap(self) -> None:
        # "hello world" vs "hello there" => intersection={hello}, union={hello,world,there}
        result = _jaccard_similarity("hello world", "hello there")
        assert pytest.approx(result) == 1 / 3

    def test_both_empty_strings_return_one(self) -> None:
        assert _jaccard_similarity("", "") == 1.0

    def test_one_empty_string_returns_zero(self) -> None:
        assert _jaccard_similarity("hello", "") == 0.0
        assert _jaccard_similarity("", "world") == 0.0

    def test_case_insensitive(self) -> None:
        result = _jaccard_similarity("Hello World", "hello world")
        assert result == 1.0

    def test_single_shared_word(self) -> None:
        # intersection={foo}, union={foo, bar, baz}
        result = _jaccard_similarity("foo bar", "foo baz")
        assert pytest.approx(result) == 1 / 3


# ---------------------------------------------------------------------------
# BasicAccuracyEvaluator construction
# ---------------------------------------------------------------------------


class TestBasicAccuracyEvaluatorConstruction:
    def test_default_mode_is_fuzzy(self) -> None:
        ev = BasicAccuracyEvaluator()
        assert ev.mode == "fuzzy"

    def test_invalid_mode_raises_evaluator_error(self) -> None:
        with pytest.raises(EvaluatorError, match="Invalid mode"):
            BasicAccuracyEvaluator(mode="invalid")

    def test_fuzzy_threshold_out_of_range_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="fuzzy_threshold"):
            BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=1.5)

    def test_fuzzy_threshold_negative_raises_error(self) -> None:
        with pytest.raises(EvaluatorError, match="fuzzy_threshold"):
            BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=-0.1)

    def test_valid_modes_accepted(self) -> None:
        for mode in ("exact", "fuzzy", "contains", "regex"):
            ev = BasicAccuracyEvaluator(mode=mode)
            assert ev.mode == mode

    def test_dimension_property(self) -> None:
        ev = BasicAccuracyEvaluator()
        assert ev.dimension == Dimension.ACCURACY

    def test_name_property(self) -> None:
        ev = BasicAccuracyEvaluator()
        assert ev.name == "BasicAccuracyEvaluator"


# ---------------------------------------------------------------------------
# evaluate — no expected_output
# ---------------------------------------------------------------------------


class TestAccuracyEvaluateNoExpected:
    def test_none_expected_output_returns_zero_score(self) -> None:
        ev = BasicAccuracyEvaluator()
        result = ev.evaluate("c1", "some output", None, {})
        assert result.score == 0.0
        assert result.passed is False
        assert "No expected_output" in result.reason


# ---------------------------------------------------------------------------
# Exact mode
# ---------------------------------------------------------------------------


class TestExactMode:
    @pytest.fixture
    def ev(self) -> BasicAccuracyEvaluator:
        return BasicAccuracyEvaluator(mode="exact")

    def test_exact_match_passes(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "Paris", "Paris", {})
        assert result.passed is True
        assert result.score == 1.0

    def test_case_insensitive_match_passes(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "paris", "PARIS", {})
        assert result.passed is True

    def test_whitespace_stripped_before_comparison(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "  Paris  ", "Paris", {})
        assert result.passed is True

    def test_different_strings_fail(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "London", "Paris", {})
        assert result.passed is False
        assert result.score == 0.0

    def test_reason_contains_expected_on_failure(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "London", "Paris", {})
        assert "Paris" in result.reason


# ---------------------------------------------------------------------------
# Fuzzy mode
# ---------------------------------------------------------------------------


class TestFuzzyMode:
    def test_identical_text_passes(self) -> None:
        ev = BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=0.7)
        result = ev.evaluate("c1", "The quick brown fox", "The quick brown fox", {})
        assert result.passed is True
        assert result.score == 1.0

    def test_low_similarity_fails(self) -> None:
        ev = BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=0.9)
        result = ev.evaluate("c1", "apple", "completely different text here", {})
        assert result.passed is False

    def test_score_reflects_jaccard_similarity(self) -> None:
        ev = BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=0.5)
        # "hello world" vs "hello there": 1/3 similarity
        result = ev.evaluate("c1", "hello world", "hello there", {})
        assert pytest.approx(result.score, abs=0.01) == 1 / 3

    def test_threshold_boundary_passes(self) -> None:
        # Both identical => similarity = 1.0, always passes any threshold <= 1.0
        ev = BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=1.0)
        result = ev.evaluate("c1", "exact match", "exact match", {})
        assert result.passed is True

    def test_reason_contains_similarity_and_threshold(self) -> None:
        ev = BasicAccuracyEvaluator(mode="fuzzy", fuzzy_threshold=0.7)
        result = ev.evaluate("c1", "hello", "hello", {})
        assert "similarity" in result.reason.lower() or "jaccard" in result.reason.lower()


# ---------------------------------------------------------------------------
# Contains mode
# ---------------------------------------------------------------------------


class TestContainsMode:
    @pytest.fixture
    def ev(self) -> BasicAccuracyEvaluator:
        return BasicAccuracyEvaluator(mode="contains")

    def test_expected_substring_passes(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "The capital is Paris.", "Paris", {})
        assert result.passed is True
        assert result.score == 1.0

    def test_missing_substring_fails(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "The capital is London.", "Paris", {})
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive_contains(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "The capital is PARIS.", "paris", {})
        assert result.passed is True

    def test_reason_mentions_substring(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "no match here", "needle", {})
        assert "needle" in result.reason


# ---------------------------------------------------------------------------
# Regex mode
# ---------------------------------------------------------------------------


class TestRegexMode:
    @pytest.fixture
    def ev(self) -> BasicAccuracyEvaluator:
        return BasicAccuracyEvaluator(mode="regex")

    def test_pattern_match_passes(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "Answer: 42", r"\d+", {})
        assert result.passed is True
        assert result.score == 1.0

    def test_pattern_no_match_fails(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "no digits here", r"^\d+$", {})
        assert result.passed is False
        assert result.score == 0.0

    def test_invalid_regex_returns_zero_score(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "output", "[invalid(regex", {})
        assert result.passed is False
        assert result.score == 0.0
        assert "Invalid regex" in result.reason

    def test_case_insensitive_regex(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "Hello World", "hello world", {})
        assert result.passed is True

    def test_dotall_flag_allows_multiline(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "line1\nline2", r"line1.*line2", {})
        assert result.passed is True

    def test_reason_contains_pattern_name_on_pass(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "foo 123", r"\d+", {})
        assert r"\d+" in result.reason

    def test_reason_contains_pattern_name_on_fail(self, ev: BasicAccuracyEvaluator) -> None:
        result = ev.evaluate("c1", "no numbers", r"\d+", {})
        assert r"\d+" in result.reason
