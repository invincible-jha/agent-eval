"""Tests for pytest-agent-eval similarity scoring and new plugin components.

Coverage targets
----------------
SimilarityScorer
    - Typo near-match scores above zero (regression: was 0.0 with binary scorer)
    - Synonym / partial-match scores above zero
    - Exact match returns 1.0
    - Complete mismatch returns low score (< 0.2)
    - Empty inputs return 0.0
    - Strategy selection: token_overlap, fuzzy, ngram, combined
    - Invalid strategy raises ValueError

assert_accuracy (context integration)
    - Backward compat: threshold parameter still works
    - New strategy parameter routes correctly
    - Scores are stored in context after call

BaselineStore
    - save_baseline persists an entry
    - get_baseline retrieves by name
    - compare returns per-metric delta dicts
    - Missing baseline returns empty dict
    - Delta arithmetic is correct (positive and negative)
    - Overwrite updates recorded_at and scores

EvalReport
    - add_result increments counters correctly
    - passed/failed counts are mutually exclusive
    - to_json returns valid JSON with expected keys
    - to_markdown returns a string containing the test name
    - to_markdown PASS/FAIL labels are correct

MultiRunEvaluator
    - record_run accumulates runs
    - compute_consistency returns empty list with < 2 runs
    - compute_consistency returns ConsistencyResult per metric
    - consistency_score is 1.0 for perfectly stable metric
    - consistency_score is lower for variable metric
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent_eval.pytest_plugin import EvalContext
from agent_eval.pytest_plugin.baseline import BaselineEntry, BaselineStore
from agent_eval.pytest_plugin.multi_run import ConsistencyResult, MultiRunEvaluator
from agent_eval.pytest_plugin.report import EvalReport
from agent_eval.pytest_plugin.similarity import SimilarityScorer


# ===========================================================================
# SimilarityScorer — core scoring behaviour
# ===========================================================================


class TestSimilarityScorerTypoRecovery:
    """Binary scorer returned 0.0 for these; multi-strategy must score > 0."""

    def test_typo_paris_pairs_scores_above_zero(self) -> None:
        # "Pairs" is a transposition-typo of "Paris" — should get partial credit.
        scorer = SimilarityScorer()
        result_score = scorer.score(
            "The capital of France is Pairs", "Paris", strategy="combined"
        )
        assert result_score > 0.0, f"Expected > 0.0 but got {result_score}"

    def test_synonym_unauthorized_not_authorized_scores_above_zero(self) -> None:
        # "not authorized" semantically matches "unauthorized" — fuzzy should help.
        scorer = SimilarityScorer()
        result_score = scorer.score("not authorized", "unauthorized", strategy="combined")
        assert result_score > 0.0, f"Expected > 0.0 but got {result_score}"

    def test_synonym_unauthorized_not_authorized_scores_above_threshold(self) -> None:
        # Fuzzy match on shared "authorized" substring should push score > 0.3.
        scorer = SimilarityScorer()
        result_score = scorer.score("not authorized", "unauthorized", strategy="fuzzy")
        assert result_score > 0.3, f"Expected > 0.3 but got {result_score}"


class TestSimilarityScorerExtremes:
    def test_exact_match_returns_one(self) -> None:
        scorer = SimilarityScorer()
        assert scorer.score("Paris", "Paris") == pytest.approx(1.0, abs=0.001)

    def test_exact_match_case_insensitive_combined_returns_high_score(self) -> None:
        scorer = SimilarityScorer()
        # "PARIS" contains "paris" (case-insensitive) — floor applies.
        result_score = scorer.score("PARIS", "paris")
        assert result_score >= 0.85

    def test_complete_mismatch_returns_low_score(self) -> None:
        scorer = SimilarityScorer()
        result_score = scorer.score("xyzzy qrs tuv", "abc", strategy="combined")
        assert result_score < 0.2, f"Expected < 0.2 but got {result_score}"

    def test_empty_result_returns_zero(self) -> None:
        scorer = SimilarityScorer()
        assert scorer.score("", "Paris") == 0.0

    def test_empty_expected_returns_zero(self) -> None:
        scorer = SimilarityScorer()
        assert scorer.score("Paris", "") == 0.0

    def test_both_empty_returns_zero(self) -> None:
        scorer = SimilarityScorer()
        assert scorer.score("", "") == 0.0


class TestSimilarityScorerStrategies:
    def test_token_overlap_strategy_returns_float_in_unit_interval(self) -> None:
        scorer = SimilarityScorer()
        result_score = scorer.score("Paris is great", "Paris", strategy="token_overlap")
        assert 0.0 <= result_score <= 1.0

    def test_fuzzy_strategy_returns_float_in_unit_interval(self) -> None:
        scorer = SimilarityScorer()
        result_score = scorer.score("Paris is great", "Paris", strategy="fuzzy")
        assert 0.0 <= result_score <= 1.0

    def test_ngram_strategy_returns_float_in_unit_interval(self) -> None:
        scorer = SimilarityScorer()
        result_score = scorer.score("Paris is great", "Paris", strategy="ngram")
        assert 0.0 <= result_score <= 1.0

    def test_combined_strategy_returns_float_in_unit_interval(self) -> None:
        scorer = SimilarityScorer()
        result_score = scorer.score("Paris is great", "Paris", strategy="combined")
        assert 0.0 <= result_score <= 1.0

    def test_token_overlap_identical_strings_returns_one(self) -> None:
        scorer = SimilarityScorer()
        assert scorer.score("hello world", "hello world", strategy="token_overlap") == pytest.approx(1.0)

    def test_fuzzy_identical_strings_returns_one(self) -> None:
        scorer = SimilarityScorer()
        assert scorer.score("hello world", "hello world", strategy="fuzzy") == pytest.approx(1.0)

    def test_ngram_identical_strings_returns_one(self) -> None:
        scorer = SimilarityScorer()
        assert scorer.score("hello world", "hello world", strategy="ngram") == pytest.approx(1.0)

    def test_invalid_strategy_raises_value_error(self) -> None:
        scorer = SimilarityScorer()
        with pytest.raises(ValueError, match="Unknown strategy"):
            scorer.score("result", "expected", strategy="magic_nlp")

    def test_token_overlap_no_shared_tokens_returns_zero(self) -> None:
        scorer = SimilarityScorer()
        result_score = scorer.score("alpha beta", "gamma delta", strategy="token_overlap")
        assert result_score == pytest.approx(0.0)

    def test_combined_exact_substring_applies_floor(self) -> None:
        # When expected is a substring of result, combined must be >= 0.85.
        scorer = SimilarityScorer()
        result_score = scorer.score(
            "The capital of France is Paris.", "Paris", strategy="combined"
        )
        assert result_score >= 0.85


# ===========================================================================
# assert_accuracy — integration with EvalContext
# ===========================================================================


class TestAssertAccuracyIntegration:
    def test_backward_compat_threshold_still_works(self) -> None:
        ctx = EvalContext()
        # Intent present → should pass with default threshold.
        ctx.assert_accuracy("The answer is Paris.", expected_intent="Paris", threshold=0.8)
        assert ctx.all_passed

    def test_backward_compat_no_match_fails_default_threshold(self) -> None:
        ctx = EvalContext()
        ctx.assert_accuracy("Berlin is a city.", expected_intent="Paris", threshold=0.8)
        assert not ctx.all_passed

    def test_strategy_parameter_token_overlap_works(self) -> None:
        ctx = EvalContext()
        # Shared tokens: "Paris" appears in both.
        ctx.assert_accuracy(
            "Paris and Lyon are cities.",
            expected_intent="Paris",
            threshold=0.1,
            strategy="token_overlap",
        )
        assert ctx.all_passed

    def test_strategy_parameter_fuzzy_works(self) -> None:
        ctx = EvalContext()
        ctx.assert_accuracy(
            "unauthorized access denied",
            expected_intent="unauthorized",
            threshold=0.5,
            strategy="fuzzy",
        )
        assert ctx.all_passed

    def test_score_stored_in_context_after_call(self) -> None:
        ctx = EvalContext()
        ctx.assert_accuracy("Paris is lovely.", expected_intent="Paris")
        assert "accuracy" in ctx.scores
        assert isinstance(ctx.scores["accuracy"], float)
        assert 0.0 <= ctx.scores["accuracy"] <= 1.0

    def test_empty_result_stores_zero_score(self) -> None:
        ctx = EvalContext()
        ctx.assert_accuracy("", expected_intent="Paris")
        assert ctx.scores["accuracy"] == 0.0

    def test_reason_contains_intent_on_pass(self) -> None:
        ctx = EvalContext()
        ctx.assert_accuracy("Paris is lovely.", expected_intent="Paris")
        _, _, reason = ctx.assertions[0]
        assert "Paris" in reason

    def test_reason_contains_intent_on_fail(self) -> None:
        ctx = EvalContext()
        ctx.assert_accuracy("Rome is lovely.", expected_intent="unicorn")
        _, _, reason = ctx.assertions[0]
        assert "unicorn" in reason


# ===========================================================================
# BaselineStore
# ===========================================================================


class TestBaselineStore:
    def _tmp_store(self) -> tuple[BaselineStore, Path]:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()  # remove so store starts fresh
        return BaselineStore(path), path

    def test_save_baseline_persists_entry(self) -> None:
        store, path = self._tmp_store()
        store.save_baseline("test_my_agent", {"accuracy": 0.9})
        # File should now exist.
        assert path.exists()
        data = json.loads(path.read_text())
        assert "test_my_agent" in data
        path.unlink(missing_ok=True)

    def test_get_baseline_returns_entry_after_save(self) -> None:
        store, path = self._tmp_store()
        store.save_baseline("test_abc", {"accuracy": 0.85, "safety": 1.0})
        entry = store.get_baseline("test_abc")
        assert isinstance(entry, BaselineEntry)
        assert entry.test_name == "test_abc"
        assert entry.scores["accuracy"] == pytest.approx(0.85)
        path.unlink(missing_ok=True)

    def test_compare_returns_per_metric_delta_dict(self) -> None:
        store, path = self._tmp_store()
        store.save_baseline("test_xyz", {"accuracy": 0.8, "safety": 1.0})
        deltas = store.compare("test_xyz", {"accuracy": 0.9, "safety": 0.8})
        assert "accuracy" in deltas
        assert "safety" in deltas
        assert deltas["accuracy"]["baseline"] == pytest.approx(0.8)
        assert deltas["accuracy"]["current"] == pytest.approx(0.9)
        assert deltas["accuracy"]["delta"] == pytest.approx(0.1)
        path.unlink(missing_ok=True)

    def test_compare_missing_baseline_returns_empty_dict(self) -> None:
        store, path = self._tmp_store()
        result = store.compare("nonexistent_test", {"accuracy": 0.7})
        assert result == {}
        path.unlink(missing_ok=True)

    def test_delta_is_negative_on_regression(self) -> None:
        store, path = self._tmp_store()
        store.save_baseline("test_reg", {"accuracy": 0.95})
        deltas = store.compare("test_reg", {"accuracy": 0.70})
        assert deltas["accuracy"]["delta"] == pytest.approx(-0.25)
        path.unlink(missing_ok=True)

    def test_overwrite_updates_scores(self) -> None:
        store, path = self._tmp_store()
        store.save_baseline("test_overwrite", {"accuracy": 0.5})
        store.save_baseline("test_overwrite", {"accuracy": 0.95})
        entry = store.get_baseline("test_overwrite")
        assert entry is not None
        assert entry.scores["accuracy"] == pytest.approx(0.95)
        path.unlink(missing_ok=True)

    def test_load_from_existing_file(self) -> None:
        store, path = self._tmp_store()
        store.save_baseline("test_persist", {"latency": 0.75})
        # Create a new store instance pointing to same file — should load existing data.
        store2 = BaselineStore(path)
        entry = store2.get_baseline("test_persist")
        assert entry is not None
        assert entry.scores["latency"] == pytest.approx(0.75)
        path.unlink(missing_ok=True)


# ===========================================================================
# EvalReport
# ===========================================================================


class TestEvalReport:
    def _sample_assertions(self) -> list[tuple[str, bool, str]]:
        return [("accuracy", True, "score 0.9 >= 0.8")]

    def test_add_result_increments_total_tests(self) -> None:
        report = EvalReport()
        report.add_result("test_a", True, {"accuracy": 0.9}, self._sample_assertions())
        assert report.total_tests == 1

    def test_add_result_increments_passed_on_pass(self) -> None:
        report = EvalReport()
        report.add_result("test_a", True, {"accuracy": 0.9}, self._sample_assertions())
        assert report.passed_tests == 1
        assert report.failed_tests == 0

    def test_add_result_increments_failed_on_fail(self) -> None:
        report = EvalReport()
        report.add_result("test_a", False, {"accuracy": 0.3}, self._sample_assertions())
        assert report.failed_tests == 1
        assert report.passed_tests == 0

    def test_to_json_returns_valid_json(self) -> None:
        report = EvalReport()
        report.add_result("test_a", True, {"accuracy": 0.9}, self._sample_assertions())
        raw = report.to_json()
        parsed: object = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_to_json_contains_test_name(self) -> None:
        report = EvalReport()
        report.add_result("my_special_test", True, {}, [])
        assert "my_special_test" in report.to_json()

    def test_to_markdown_contains_test_name(self) -> None:
        report = EvalReport()
        report.add_result("test_capital_agent", True, {"accuracy": 0.9}, [])
        assert "test_capital_agent" in report.to_markdown()

    def test_to_markdown_pass_label_for_passing_test(self) -> None:
        report = EvalReport()
        report.add_result("test_pass", True, {}, [])
        assert "PASS" in report.to_markdown()

    def test_to_markdown_fail_label_for_failing_test(self) -> None:
        report = EvalReport()
        report.add_result("test_fail", False, {}, [])
        assert "FAIL" in report.to_markdown()

    def test_to_markdown_contains_report_header(self) -> None:
        report = EvalReport()
        assert "Agent Evaluation Report" in report.to_markdown()

    def test_multiple_results_all_counted(self) -> None:
        report = EvalReport()
        report.add_result("test_a", True, {}, [])
        report.add_result("test_b", False, {}, [])
        report.add_result("test_c", True, {}, [])
        assert report.total_tests == 3
        assert report.passed_tests == 2
        assert report.failed_tests == 1


# ===========================================================================
# MultiRunEvaluator
# ===========================================================================


class TestMultiRunEvaluator:
    def test_record_run_increments_run_count(self) -> None:
        evaluator = MultiRunEvaluator()
        evaluator.record_run({"accuracy": 0.9})
        assert evaluator.run_count == 1

    def test_compute_consistency_empty_on_single_run(self) -> None:
        evaluator = MultiRunEvaluator()
        evaluator.record_run({"accuracy": 0.9})
        assert evaluator.compute_consistency() == []

    def test_compute_consistency_empty_on_zero_runs(self) -> None:
        evaluator = MultiRunEvaluator()
        assert evaluator.compute_consistency() == []

    def test_compute_consistency_returns_result_per_metric(self) -> None:
        evaluator = MultiRunEvaluator()
        evaluator.record_run({"accuracy": 0.9, "safety": 1.0})
        evaluator.record_run({"accuracy": 0.85, "safety": 1.0})
        results = evaluator.compute_consistency()
        metrics = {r.metric for r in results}
        assert "accuracy" in metrics
        assert "safety" in metrics

    def test_consistency_score_is_one_for_stable_metric(self) -> None:
        evaluator = MultiRunEvaluator()
        for _ in range(5):
            evaluator.record_run({"accuracy": 0.9})
        results = evaluator.compute_consistency()
        accuracy_result = next(r for r in results if r.metric == "accuracy")
        assert accuracy_result.consistency_score == pytest.approx(1.0, abs=0.001)

    def test_consistency_score_lower_for_variable_metric(self) -> None:
        evaluator = MultiRunEvaluator()
        evaluator.record_run({"accuracy": 0.5})
        evaluator.record_run({"accuracy": 0.9})
        results = evaluator.compute_consistency()
        accuracy_result = next(r for r in results if r.metric == "accuracy")
        # High variance → consistency_score well below 1.0.
        assert accuracy_result.consistency_score < 0.9

    def test_consistency_result_dataclass_fields(self) -> None:
        evaluator = MultiRunEvaluator()
        evaluator.record_run({"latency": 0.8})
        evaluator.record_run({"latency": 0.9})
        results = evaluator.compute_consistency()
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ConsistencyResult)
        assert result.metric == "latency"
        assert result.mean == pytest.approx(0.85, abs=0.001)
        assert result.min_val == pytest.approx(0.8)
        assert result.max_val == pytest.approx(0.9)

    def test_results_sorted_alphabetically_by_metric(self) -> None:
        evaluator = MultiRunEvaluator()
        evaluator.record_run({"safety": 1.0, "accuracy": 0.9, "cost": 0.7})
        evaluator.record_run({"safety": 1.0, "accuracy": 0.85, "cost": 0.65})
        results = evaluator.compute_consistency()
        metric_names = [r.metric for r in results]
        assert metric_names == sorted(metric_names)
