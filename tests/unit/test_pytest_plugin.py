"""Comprehensive tests for agent_eval.pytest_plugin.

Coverage targets
----------------
- EvalContext.assert_accuracy: match, no-match, threshold variants, empty inputs
- EvalContext.assert_safety: clean output, SSN, email, credit card, harmful content,
  combined flags, disabled checks
- EvalContext.assert_cost: under limit (estimated), over limit, actual_tokens override,
  boundary at max_tokens, zero tokens
- EvalContext.assert_latency: under max, at max, over max, zero duration
- EvalContext.scores property: empty, single entry, multiple entries, immutability
- EvalContext.all_passed property: empty, all pass, one fail, all fail
- EvalContext.assertions property: snapshot semantics
- EvalContext.__repr__
- Plugin marker registration (pytest_configure)
- Plugin collection modification (pytest_collection_modifyitems)
- Fixture availability and isolation
- MarkerSpec dataclass
- AgentEvalMarkerArgs defaults and from_marker parsing
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_eval.pytest_plugin import EvalContext
from agent_eval.pytest_plugin.context import (
    EvalContext as ContextEvalContext,
    _CREDIT_CARD_PATTERN,
    _EMAIL_PATTERN,
    _SSN_PATTERN,
)
from agent_eval.pytest_plugin.markers import (
    AGENT_EVAL_MARKER,
    ALL_MARKERS,
    AgentEvalMarkerArgs,
    MarkerSpec,
)
from agent_eval.pytest_plugin.plugin import pytest_configure, pytest_collection_modifyitems


# ===========================================================================
# Helpers
# ===========================================================================


def _make_eval_context() -> EvalContext:
    """Return a fresh EvalContext instance for test use."""
    return EvalContext()


# ===========================================================================
# EvalContext — assert_accuracy
# ===========================================================================


class TestAssertAccuracy:
    def test_exact_intent_match_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("The capital of France is Paris.", expected_intent="Paris")
        assert ctx.all_passed

    def test_case_insensitive_match_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("The answer is PARIS.", expected_intent="paris")
        assert ctx.all_passed

    def test_intent_not_in_result_fails(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("The capital of France is Lyon.", expected_intent="Paris")
        assert not ctx.all_passed

    def test_accuracy_score_is_high_on_match(self) -> None:
        # Multi-strategy combined score for a present keyword is well above threshold.
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris is the capital.", expected_intent="Paris")
        assert ctx.scores["accuracy"] >= 0.5

    def test_accuracy_score_is_low_on_no_match(self) -> None:
        # Multi-strategy scorer returns a low but non-zero score for unrelated text.
        # The binary 0.0 guarantee no longer holds; assert it stays well below threshold.
        ctx = _make_eval_context()
        ctx.assert_accuracy("Berlin is a great city.", expected_intent="Paris")
        assert ctx.scores["accuracy"] < 0.5

    def test_default_threshold_is_0_8(self) -> None:
        # Score 1.0 >= 0.8 → passes; score 0.0 < 0.8 → fails
        ctx = _make_eval_context()
        ctx.assert_accuracy("No match here.", expected_intent="Paris")
        assert not ctx.all_passed

    def test_threshold_zero_always_passes_on_match(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris", expected_intent="Paris", threshold=0.0)
        assert ctx.all_passed

    def test_threshold_one_fails_without_perfect_similarity(self) -> None:
        # Multi-strategy scorer rarely produces exactly 1.0 for a short keyword
        # embedded in a longer sentence. threshold=1.0 is a strict ceiling test.
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris is lovely.", expected_intent="Paris", threshold=1.0)
        # score will be < 1.0 due to extra surrounding text lowering all metrics
        assert not ctx.all_passed

    def test_threshold_one_fails_when_keyword_absent(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Rome is lovely.", expected_intent="Paris", threshold=1.0)
        assert not ctx.all_passed

    def test_empty_result_fails(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("", expected_intent="Paris")
        assert not ctx.all_passed

    def test_empty_expected_intent_fails(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Some result.", expected_intent="")
        assert not ctx.all_passed

    def test_both_empty_fails(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("", expected_intent="")
        assert not ctx.all_passed

    def test_accuracy_assertion_recorded(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris is the answer.", expected_intent="Paris")
        assert len(ctx.assertions) == 1
        dimension, passed, reason = ctx.assertions[0]
        assert dimension == "accuracy"
        assert passed is True
        assert "Paris" in reason

    def test_failed_accuracy_reason_mentions_intent(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("No relevant content.", expected_intent="unicorn")
        _, _, reason = ctx.assertions[0]
        assert "unicorn" in reason

    def test_multiword_intent_matched_as_substring(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy(
            "The recommended approach is test-driven development.",
            expected_intent="test-driven development",
        )
        assert ctx.all_passed


# ===========================================================================
# EvalContext — assert_safety
# ===========================================================================


class TestAssertSafety:
    def test_clean_text_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("The weather today is sunny and warm.")
        assert ctx.all_passed

    def test_clean_text_score_is_one(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("All good here.")
        assert ctx.scores["safety"] == 1.0

    def test_ssn_pattern_detected(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("Your SSN is 123-45-6789.")
        assert not ctx.all_passed
        assert ctx.scores["safety"] == 0.0

    def test_ssn_with_spaces_detected(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("SSN: 123 45 6789")
        assert not ctx.all_passed

    def test_email_address_detected(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("Contact admin@example.com for help.")
        assert not ctx.all_passed

    def test_email_with_subdomains_detected(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("Send to user@mail.company.org")
        assert not ctx.all_passed

    def test_visa_credit_card_detected(self) -> None:
        ctx = _make_eval_context()
        # Valid Luhn-format Visa test number
        ctx.assert_safety("Card: 4111111111111111")
        assert not ctx.all_passed

    def test_safety_assertion_reason_mentions_pii_type(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("SSN: 987-65-4321")
        _, _, reason = ctx.assertions[0]
        assert "SSN" in reason

    def test_no_pii_false_skips_pii_check(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("SSN: 123-45-6789", no_pii=False)
        # PII check disabled — should pass (no harmful content either)
        assert ctx.all_passed

    def test_no_harmful_content_false_skips_harmful_check(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("how to make a bomb step by step", no_harmful_content=False)
        # Harmful-content check disabled; no PII in text → passes
        assert ctx.all_passed

    def test_harmful_content_keyword_detected(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("how to make a bomb", no_pii=False, no_harmful_content=True)
        assert not ctx.all_passed

    def test_safety_score_zero_on_violation(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("user@evil.com")
        assert ctx.scores["safety"] == 0.0

    def test_multiple_pii_violations_all_captured_in_reason(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("SSN: 000-12-3456 and email: foo@bar.com")
        _, _, reason = ctx.assertions[0]
        assert "SSN" in reason
        assert "email" in reason

    def test_clean_output_reason_says_no_violations(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_safety("This is perfectly safe output.")
        _, _, reason = ctx.assertions[0]
        assert "no violations" in reason


# ===========================================================================
# EvalContext — assert_cost
# ===========================================================================


class TestAssertCost:
    def test_short_output_within_budget_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_cost("Hello world.", max_tokens=1000)
        assert ctx.all_passed

    def test_score_above_half_when_under_budget(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_cost("Hello world.", max_tokens=1000)
        assert ctx.scores["cost"] >= 0.5

    def test_long_output_over_budget_fails(self) -> None:
        ctx = _make_eval_context()
        # Generate text substantially exceeding the budget
        long_text = " ".join(["word"] * 1000)  # ~1300 estimated tokens
        ctx.assert_cost(long_text, max_tokens=100)
        assert not ctx.all_passed

    def test_over_budget_score_between_zero_and_one(self) -> None:
        ctx = _make_eval_context()
        long_text = " ".join(["word"] * 1000)
        ctx.assert_cost(long_text, max_tokens=100)
        assert 0.0 <= ctx.scores["cost"] <= 1.0

    def test_actual_tokens_overrides_estimation(self) -> None:
        ctx = _make_eval_context()
        # Tiny text but actual_tokens is huge
        ctx.assert_cost("Hi.", max_tokens=100, actual_tokens=500)
        assert not ctx.all_passed

    def test_actual_tokens_under_budget_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_cost("Some long text " * 100, max_tokens=1000, actual_tokens=50)
        assert ctx.all_passed

    def test_exact_boundary_at_max_tokens_passes(self) -> None:
        # actual_tokens == max_tokens → within budget
        ctx = _make_eval_context()
        ctx.assert_cost("ignored", max_tokens=100, actual_tokens=100)
        assert ctx.all_passed

    def test_zero_actual_tokens_passes_with_perfect_score(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_cost("", max_tokens=1000, actual_tokens=0)
        assert ctx.all_passed
        assert ctx.scores["cost"] == 1.0

    def test_empty_result_no_actual_tokens_estimates_zero(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_cost("", max_tokens=1000)
        assert ctx.all_passed

    def test_cost_assertion_recorded(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_cost("short", max_tokens=1000)
        assert len(ctx.assertions) == 1
        dimension, _, _ = ctx.assertions[0]
        assert dimension == "cost"

    def test_reason_mentions_token_count_and_budget(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_cost("word " * 10, max_tokens=500)
        _, _, reason = ctx.assertions[0]
        assert "500" in reason


# ===========================================================================
# EvalContext — assert_latency
# ===========================================================================


class TestAssertLatency:
    def test_fast_response_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=1.0, max_seconds=5.0)
        assert ctx.all_passed

    def test_exactly_at_limit_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=5.0, max_seconds=5.0)
        assert ctx.all_passed

    def test_over_limit_fails(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=6.0, max_seconds=5.0)
        assert not ctx.all_passed

    def test_zero_duration_passes(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=0.0, max_seconds=5.0)
        assert ctx.all_passed

    def test_zero_duration_score_is_one(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=0.0, max_seconds=5.0)
        assert ctx.scores["latency"] == 1.0

    def test_score_between_zero_and_one_when_over(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=10.0, max_seconds=5.0)
        assert 0.0 <= ctx.scores["latency"] <= 1.0

    def test_score_is_half_at_max_seconds(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=5.0, max_seconds=5.0)
        # At duration == max: score = 1.0 - (1.0 * 0.5) = 0.5
        assert ctx.scores["latency"] == pytest.approx(0.5, abs=0.0001)

    def test_default_max_seconds_is_five(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(duration_seconds=4.9)
        assert ctx.all_passed
        ctx2 = _make_eval_context()
        ctx2.assert_latency(duration_seconds=5.1)
        assert not ctx2.all_passed

    def test_latency_assertion_recorded(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(1.2)
        assert len(ctx.assertions) == 1
        dimension, _, _ = ctx.assertions[0]
        assert dimension == "latency"

    def test_reason_contains_duration_and_limit(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(2.5, max_seconds=10.0)
        _, _, reason = ctx.assertions[0]
        assert "2.500" in reason
        assert "10.000" in reason


# ===========================================================================
# EvalContext — scores property
# ===========================================================================


class TestScoresProperty:
    def test_empty_context_has_empty_scores(self) -> None:
        ctx = _make_eval_context()
        assert ctx.scores == {}

    def test_single_assertion_adds_one_score(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(1.0)
        assert "latency" in ctx.scores
        assert len(ctx.scores) == 1

    def test_multiple_assertions_add_multiple_scores(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris", "Paris")
        ctx.assert_latency(1.0)
        ctx.assert_cost("short", 1000)
        assert set(ctx.scores.keys()) == {"accuracy", "latency", "cost"}

    def test_scores_returns_copy_not_reference(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(1.0)
        snapshot = ctx.scores
        snapshot["injected"] = 99.9
        # Internal state should not be affected
        assert "injected" not in ctx.scores

    def test_score_values_are_floats_in_unit_interval(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris is great", "Paris")
        ctx.assert_safety("Clean output")
        ctx.assert_cost("short", 1000)
        ctx.assert_latency(1.0)
        for score in ctx.scores.values():
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_repeated_assertion_overwrites_score(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(0.5)
        ctx.assert_latency(10.0)
        # Second call overwrites the first; should now be failing score
        assert ctx.scores["latency"] < 1.0


# ===========================================================================
# EvalContext — all_passed property
# ===========================================================================


class TestAllPassedProperty:
    def test_empty_context_all_passed_is_true(self) -> None:
        ctx = _make_eval_context()
        assert ctx.all_passed is True

    def test_single_passing_assertion_all_passed_true(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(1.0)
        assert ctx.all_passed is True

    def test_single_failing_assertion_all_passed_false(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(100.0)
        assert ctx.all_passed is False

    def test_mixed_assertions_with_one_fail_returns_false(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris is great", "Paris")   # passes
        ctx.assert_latency(100.0)                          # fails
        ctx.assert_safety("Clean output")                  # passes
        assert ctx.all_passed is False

    def test_all_assertions_passing_returns_true(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris is great", "Paris")
        ctx.assert_latency(1.0)
        ctx.assert_safety("Clean output")
        assert ctx.all_passed is True

    def test_all_assertions_failing_returns_false(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("No match", "Paris")
        ctx.assert_latency(100.0)
        assert ctx.all_passed is False


# ===========================================================================
# EvalContext — assertions property
# ===========================================================================


class TestAssertionsProperty:
    def test_empty_context_has_no_assertions(self) -> None:
        ctx = _make_eval_context()
        assert ctx.assertions == []

    def test_assertions_returns_copy_not_reference(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(1.0)
        snapshot = ctx.assertions
        snapshot.append(("injected", True, "injected"))
        # Internal list must not be affected
        assert len(ctx.assertions) == 1

    def test_assertion_tuple_structure(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(1.0)
        entry = ctx.assertions[0]
        assert len(entry) == 3
        dimension, passed, reason = entry
        assert isinstance(dimension, str)
        assert isinstance(passed, bool)
        assert isinstance(reason, str)

    def test_assertion_order_preserved(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_accuracy("Paris", "Paris")
        ctx.assert_safety("clean")
        ctx.assert_cost("short", 100)
        dimensions = [a[0] for a in ctx.assertions]
        assert dimensions == ["accuracy", "safety", "cost"]


# ===========================================================================
# EvalContext — __repr__
# ===========================================================================


class TestEvalContextRepr:
    def test_repr_contains_class_name(self) -> None:
        ctx = _make_eval_context()
        assert "EvalContext" in repr(ctx)

    def test_repr_shows_zero_assertions_on_fresh_context(self) -> None:
        ctx = _make_eval_context()
        r = repr(ctx)
        assert "0/0" in r

    def test_repr_updates_after_assertion(self) -> None:
        ctx = _make_eval_context()
        ctx.assert_latency(1.0)  # passes
        r = repr(ctx)
        assert "1/1" in r


# ===========================================================================
# EvalContext — __init__ package re-export
# ===========================================================================


class TestPackageExport:
    def test_eval_context_importable_from_package(self) -> None:
        from agent_eval.pytest_plugin import EvalContext as PackageEvalContext
        assert PackageEvalContext is ContextEvalContext

    def test_eval_context_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(EvalContext)


# ===========================================================================
# Marker definitions
# ===========================================================================


class TestMarkerSpec:
    def test_marker_spec_is_frozen(self) -> None:
        spec = MarkerSpec(name="test", description="a test marker")
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "changed"  # type: ignore[misc]

    def test_agent_eval_marker_name(self) -> None:
        assert AGENT_EVAL_MARKER.name == "agent_eval"

    def test_agent_eval_marker_description_contains_dimensions(self) -> None:
        assert "dimensions" in AGENT_EVAL_MARKER.description

    def test_agent_eval_marker_description_contains_threshold(self) -> None:
        assert "threshold" in AGENT_EVAL_MARKER.description

    def test_all_markers_tuple_contains_agent_eval(self) -> None:
        assert AGENT_EVAL_MARKER in ALL_MARKERS

    def test_all_markers_is_non_empty(self) -> None:
        assert len(ALL_MARKERS) >= 1


class TestAgentEvalMarkerArgs:
    def test_default_dimensions(self) -> None:
        args = AgentEvalMarkerArgs()
        assert "accuracy" in args.dimensions
        assert "safety" in args.dimensions
        assert "cost" in args.dimensions
        assert "latency" in args.dimensions

    def test_default_runs_is_one(self) -> None:
        assert AgentEvalMarkerArgs().runs == 1

    def test_default_threshold_is_0_8(self) -> None:
        assert AgentEvalMarkerArgs().threshold == pytest.approx(0.8)

    def test_from_marker_with_empty_kwargs(self) -> None:
        mock_marker = MagicMock()
        mock_marker.kwargs = {}
        args = AgentEvalMarkerArgs.from_marker(mock_marker)
        assert args.runs == 1
        assert args.threshold == pytest.approx(0.8)

    def test_from_marker_parses_custom_dimensions(self) -> None:
        mock_marker = MagicMock()
        mock_marker.kwargs = {"dimensions": ["accuracy", "safety"], "runs": 3}
        args = AgentEvalMarkerArgs.from_marker(mock_marker)
        assert args.dimensions == ("accuracy", "safety")
        assert args.runs == 3

    def test_from_marker_parses_threshold(self) -> None:
        mock_marker = MagicMock()
        mock_marker.kwargs = {"threshold": 0.9}
        args = AgentEvalMarkerArgs.from_marker(mock_marker)
        assert args.threshold == pytest.approx(0.9)

    def test_from_marker_handles_missing_kwargs_attr(self) -> None:
        mock_marker = MagicMock(spec=[])  # no .kwargs attribute
        args = AgentEvalMarkerArgs.from_marker(mock_marker)
        assert args.runs == 1

    def test_marker_args_is_frozen(self) -> None:
        args = AgentEvalMarkerArgs()
        with pytest.raises((AttributeError, TypeError)):
            args.runs = 99  # type: ignore[misc]


# ===========================================================================
# Plugin hooks
# ===========================================================================


class TestPytestConfigure:
    def test_registers_agent_eval_marker(self) -> None:
        mock_config = MagicMock()
        pytest_configure(mock_config)
        mock_config.addinivalue_line.assert_called()
        # Collect all calls and check that "agent_eval" appears somewhere
        all_calls = mock_config.addinivalue_line.call_args_list
        descriptions = [str(call) for call in all_calls]
        assert any("agent_eval" in d for d in descriptions)

    def test_calls_addinivalue_line_for_each_marker(self) -> None:
        mock_config = MagicMock()
        pytest_configure(mock_config)
        assert mock_config.addinivalue_line.call_count == len(ALL_MARKERS)

    def test_first_arg_is_markers_key(self) -> None:
        mock_config = MagicMock()
        pytest_configure(mock_config)
        first_call = mock_config.addinivalue_line.call_args_list[0]
        # call_args_list entries are call objects: call_args[0] is positional args
        assert first_call[0][0] == "markers"


class TestPytestCollectionModifyItems:
    def _make_item_with_marker(self, marker_name: str) -> MagicMock:
        item = MagicMock(spec=pytest.Item)
        mock_marker = MagicMock()
        mock_marker.name = marker_name

        def get_closest_marker(name: str) -> MagicMock | None:
            return mock_marker if name == marker_name else None

        item.get_closest_marker.side_effect = get_closest_marker
        return item

    def _make_item_without_marker(self) -> MagicMock:
        item = MagicMock(spec=pytest.Item)
        item.get_closest_marker.return_value = None
        return item

    def test_agent_eval_item_gets_timeout_marker(self) -> None:
        item = self._make_item_with_marker("agent_eval")
        mock_config = MagicMock()
        pytest_collection_modifyitems(mock_config, [item])
        item.add_marker.assert_called_once()
        # The argument should be a timeout marker
        added = item.add_marker.call_args[0][0]
        assert "timeout" in str(added).lower() or hasattr(added, "name")

    def test_non_agent_eval_item_is_not_modified(self) -> None:
        item = self._make_item_without_marker()
        mock_config = MagicMock()
        pytest_collection_modifyitems(mock_config, [item])
        item.add_marker.assert_not_called()

    def test_mixed_items_only_agent_eval_gets_timeout(self) -> None:
        eval_item = self._make_item_with_marker("agent_eval")
        plain_item = self._make_item_without_marker()
        mock_config = MagicMock()
        pytest_collection_modifyitems(mock_config, [eval_item, plain_item])
        eval_item.add_marker.assert_called_once()
        plain_item.add_marker.assert_not_called()

    def test_empty_items_list_does_not_raise(self) -> None:
        mock_config = MagicMock()
        pytest_collection_modifyitems(mock_config, [])  # should not raise


# ===========================================================================
# Fixture isolation
# ===========================================================================


class TestFixtureIsolation:
    def test_eval_context_fixture_is_fresh_per_test(
        self, eval_context: EvalContext
    ) -> None:
        # Each test should receive an empty context
        assert eval_context.assertions == []
        assert eval_context.scores == {}

    def test_eval_context_fixture_accumulates_within_test(
        self, eval_context: EvalContext
    ) -> None:
        eval_context.assert_latency(1.0)
        eval_context.assert_accuracy("Paris", "Paris")
        assert len(eval_context.assertions) == 2

    def test_eval_context_fixture_all_passed_empty(
        self, eval_context: EvalContext
    ) -> None:
        assert eval_context.all_passed is True


# ===========================================================================
# PII pattern unit tests (regex correctness)
# ===========================================================================


class TestPiiPatterns:
    def test_ssn_pattern_matches_hyphen_format(self) -> None:
        assert _SSN_PATTERN.search("123-45-6789") is not None

    def test_ssn_pattern_matches_space_format(self) -> None:
        assert _SSN_PATTERN.search("123 45 6789") is not None

    def test_ssn_pattern_does_not_match_incomplete(self) -> None:
        assert _SSN_PATTERN.search("123-45-678") is None

    def test_email_pattern_matches_standard_email(self) -> None:
        assert _EMAIL_PATTERN.search("user@example.com") is not None

    def test_email_pattern_matches_subdomain_email(self) -> None:
        assert _EMAIL_PATTERN.search("a@b.co.uk") is not None

    def test_email_pattern_does_not_match_invalid(self) -> None:
        assert _EMAIL_PATTERN.search("not-an-email") is None

    def test_credit_card_pattern_matches_visa(self) -> None:
        assert _CREDIT_CARD_PATTERN.search("4111111111111111") is not None

    def test_credit_card_pattern_does_not_match_short_number(self) -> None:
        assert _CREDIT_CARD_PATTERN.search("4111") is None
