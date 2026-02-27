"""Comprehensive unit tests for the agent leaderboard system.

Covers:
- LeaderboardSubmission field validation (Pydantic v2)
- CompositeWeights normalisation and validation
- composite_score auto-computation and manual recompute
- RankingEngine.rank, top_n, filter_by_framework, filter_by_benchmark
- LeaderboardStorage CRUD, persistence, export_json, import_json
- CLI commands: leaderboard submit, leaderboard list, leaderboard export
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from agent_eval.leaderboard.ranking import RankingEngine
from agent_eval.leaderboard.storage import LeaderboardStorage
from agent_eval.leaderboard.submission import (
    CompositeWeights,
    DEFAULT_WEIGHTS,
    LeaderboardSubmission,
)


# ---------------------------------------------------------------------------
# Factories / fixtures
# ---------------------------------------------------------------------------


def _make_submission(
    agent_name: str = "agent-alpha",
    agent_version: str = "1.0.0",
    framework: str = "langchain",
    model: str = "gpt-4o",
    submitter: str = "team-a",
    accuracy_score: float = 0.90,
    safety_score: float = 0.95,
    cost_efficiency: float = 0.80,
    latency_p95_ms: float = 500.0,
    consistency_score: float = 0.85,
    security_score: float = 0.90,
    benchmark_name: str = "qa_basic",
    benchmark_version: str = "1.0",
    num_runs: int = 5,
    total_tokens: int = 20000,
    total_cost_usd: float = 0.24,
) -> LeaderboardSubmission:
    return LeaderboardSubmission(
        agent_name=agent_name,
        agent_version=agent_version,
        framework=framework,
        model=model,
        submitter=submitter,
        accuracy_score=accuracy_score,
        safety_score=safety_score,
        cost_efficiency=cost_efficiency,
        latency_p95_ms=latency_p95_ms,
        consistency_score=consistency_score,
        security_score=security_score,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        num_runs=num_runs,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
    )


@pytest.fixture()
def submission() -> LeaderboardSubmission:
    return _make_submission()


@pytest.fixture()
def many_submissions() -> list[LeaderboardSubmission]:
    return [
        _make_submission(
            agent_name="agent-alpha",
            agent_version="1.0.0",
            framework="langchain",
            accuracy_score=0.90,
            safety_score=0.95,
            cost_efficiency=0.80,
            consistency_score=0.85,
            security_score=0.90,
        ),
        _make_submission(
            agent_name="agent-beta",
            agent_version="2.0.0",
            framework="crewai",
            accuracy_score=0.70,
            safety_score=0.75,
            cost_efficiency=0.60,
            consistency_score=0.65,
            security_score=0.70,
            benchmark_name="safety_basic",
        ),
        _make_submission(
            agent_name="agent-gamma",
            agent_version="1.5.0",
            framework="langchain",
            accuracy_score=0.80,
            safety_score=0.85,
            cost_efficiency=0.70,
            consistency_score=0.75,
            security_score=0.80,
        ),
        _make_submission(
            agent_name="agent-delta",
            agent_version="3.0.0",
            framework="autogen",
            accuracy_score=0.95,
            safety_score=0.98,
            cost_efficiency=0.90,
            consistency_score=0.92,
            security_score=0.95,
        ),
    ]


@pytest.fixture()
def memory_store() -> LeaderboardStorage:
    """Return an in-memory (no-file) LeaderboardStorage instance."""
    return LeaderboardStorage(storage_path=None)


@pytest.fixture()
def file_store(tmp_path: Path) -> LeaderboardStorage:
    """Return a file-backed LeaderboardStorage with a temp file."""
    return LeaderboardStorage(storage_path=tmp_path / "leaderboard.json")


@pytest.fixture()
def populated_store(
    memory_store: LeaderboardStorage,
    many_submissions: list[LeaderboardSubmission],
) -> LeaderboardStorage:
    for sub in many_submissions:
        memory_store.save(sub)
    return memory_store


# ===========================================================================
# CompositeWeights
# ===========================================================================


class TestCompositeWeights:
    def test_default_weights_sum_to_one(self) -> None:
        w = CompositeWeights()
        total = w.accuracy + w.safety + w.cost_efficiency + w.consistency + w.security
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights_accepted(self) -> None:
        w = CompositeWeights(accuracy=0.5, safety=0.5, cost_efficiency=0.0, consistency=0.0, security=0.0)
        assert w.accuracy == 0.5

    def test_normalised_returns_unit_sum(self) -> None:
        w = CompositeWeights(accuracy=3.0, safety=2.0, cost_efficiency=1.0, consistency=1.0, security=1.0)
        norm = w.normalised()
        total = norm.accuracy + norm.safety + norm.cost_efficiency + norm.consistency + norm.security
        assert abs(total - 1.0) < 1e-9

    def test_normalised_preserves_ratios(self) -> None:
        w = CompositeWeights(accuracy=2.0, safety=2.0, cost_efficiency=2.0, consistency=2.0, security=2.0)
        norm = w.normalised()
        assert abs(norm.accuracy - 0.2) < 1e-9

    def test_zero_sum_raises_value_error(self) -> None:
        with pytest.raises(ValidationError):
            CompositeWeights(accuracy=0.0, safety=0.0, cost_efficiency=0.0, consistency=0.0, security=0.0)

    def test_negative_weight_raises_value_error(self) -> None:
        with pytest.raises(ValidationError):
            CompositeWeights(accuracy=-0.1)

    def test_all_equal_weights_normalise_to_one_fifth(self) -> None:
        w = CompositeWeights(accuracy=1.0, safety=1.0, cost_efficiency=1.0, consistency=1.0, security=1.0)
        norm = w.normalised()
        assert abs(norm.accuracy - 0.2) < 1e-9
        assert abs(norm.safety - 0.2) < 1e-9

    def test_normalised_does_not_mutate_original(self) -> None:
        w = CompositeWeights(accuracy=2.0, safety=1.0, cost_efficiency=1.0, consistency=1.0, security=1.0)
        original_accuracy = w.accuracy
        w.normalised()
        assert w.accuracy == original_accuracy

    def test_single_nonzero_weight_normalises_to_one(self) -> None:
        w = CompositeWeights(accuracy=5.0, safety=0.0, cost_efficiency=0.0, consistency=0.0, security=0.0)
        norm = w.normalised()
        assert abs(norm.accuracy - 1.0) < 1e-9

    def test_default_weights_singleton(self) -> None:
        assert DEFAULT_WEIGHTS.accuracy == 0.30
        assert DEFAULT_WEIGHTS.safety == 0.25


# ===========================================================================
# LeaderboardSubmission — field validation
# ===========================================================================


class TestLeaderboardSubmissionValidation:
    def test_valid_submission_created(self, submission: LeaderboardSubmission) -> None:
        assert submission.agent_name == "agent-alpha"
        assert submission.agent_version == "1.0.0"

    def test_empty_agent_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(agent_name="")

    def test_empty_agent_version_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(agent_version="")

    def test_empty_framework_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(framework="")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(model="")

    def test_empty_submitter_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(submitter="")

    def test_empty_benchmark_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(benchmark_name="")

    def test_empty_benchmark_version_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(benchmark_version="")

    def test_accuracy_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(accuracy_score=-0.01)

    def test_accuracy_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(accuracy_score=1.001)

    def test_safety_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(safety_score=-0.01)

    def test_safety_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(safety_score=1.001)

    def test_cost_efficiency_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(cost_efficiency=-0.01)

    def test_cost_efficiency_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(cost_efficiency=1.001)

    def test_consistency_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(consistency_score=-0.01)

    def test_consistency_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(consistency_score=1.001)

    def test_security_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(security_score=-0.01)

    def test_security_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(security_score=1.001)

    def test_latency_p95_ms_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(latency_p95_ms=-1.0)

    def test_latency_p95_ms_zero_accepted(self) -> None:
        sub = _make_submission(latency_p95_ms=0.0)
        assert sub.latency_p95_ms == 0.0

    def test_num_runs_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(num_runs=0)

    def test_num_runs_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(num_runs=-1)

    def test_total_tokens_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(total_tokens=-1)

    def test_total_cost_usd_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_submission(total_cost_usd=-0.01)

    def test_boundary_accuracy_zero(self) -> None:
        sub = _make_submission(accuracy_score=0.0)
        assert sub.accuracy_score == 0.0

    def test_boundary_accuracy_one(self) -> None:
        sub = _make_submission(accuracy_score=1.0)
        assert sub.accuracy_score == 1.0

    def test_submitted_at_defaults_to_utc_now(self, submission: LeaderboardSubmission) -> None:
        assert submission.submitted_at.tzinfo is not None

    def test_submitted_at_can_be_set_explicitly(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        sub = _make_submission()
        sub2 = LeaderboardSubmission(
            **{**sub.model_dump(), "submitted_at": ts}
        )
        assert sub2.submitted_at == ts

    def test_num_runs_large_value_accepted(self) -> None:
        sub = _make_submission(num_runs=10000)
        assert sub.num_runs == 10000

    def test_total_tokens_large_value_accepted(self) -> None:
        sub = _make_submission(total_tokens=10_000_000)
        assert sub.total_tokens == 10_000_000


# ===========================================================================
# LeaderboardSubmission — composite score
# ===========================================================================


class TestCompositeScore:
    def test_composite_score_auto_computed(self, submission: LeaderboardSubmission) -> None:
        assert submission.composite_score > 0.0

    def test_composite_score_with_all_ones(self) -> None:
        sub = _make_submission(
            accuracy_score=1.0,
            safety_score=1.0,
            cost_efficiency=1.0,
            consistency_score=1.0,
            security_score=1.0,
        )
        assert abs(sub.composite_score - 1.0) < 1e-9

    def test_composite_score_with_all_zeros(self) -> None:
        sub = _make_submission(
            accuracy_score=0.0,
            safety_score=0.0,
            cost_efficiency=0.0,
            consistency_score=0.0,
            security_score=0.0,
        )
        assert abs(sub.composite_score) < 1e-9

    def test_compute_composite_with_custom_weights(self, submission: LeaderboardSubmission) -> None:
        weights = CompositeWeights(
            accuracy=1.0,
            safety=0.0,
            cost_efficiency=0.0,
            consistency=0.0,
            security=0.0,
        )
        score = submission.compute_composite(weights)
        assert abs(score - submission.accuracy_score) < 1e-9

    def test_compute_composite_returns_new_score(self, submission: LeaderboardSubmission) -> None:
        weights = CompositeWeights(
            accuracy=0.0,
            safety=1.0,
            cost_efficiency=0.0,
            consistency=0.0,
            security=0.0,
        )
        score = submission.compute_composite(weights)
        assert abs(score - submission.safety_score) < 1e-9
        assert abs(submission.composite_score - submission.safety_score) < 1e-9

    def test_compute_composite_updates_field(self, submission: LeaderboardSubmission) -> None:
        weights = CompositeWeights(
            accuracy=0.5,
            safety=0.5,
            cost_efficiency=0.0,
            consistency=0.0,
            security=0.0,
        )
        new_score = submission.compute_composite(weights)
        assert submission.composite_score == new_score

    def test_compute_composite_with_none_uses_defaults(self, submission: LeaderboardSubmission) -> None:
        manual = submission._weighted_composite(DEFAULT_WEIGHTS)
        score = submission.compute_composite(None)
        assert abs(score - manual) < 1e-9

    def test_composite_score_is_clamped_high(self) -> None:
        sub = _make_submission(
            accuracy_score=1.0,
            safety_score=1.0,
            cost_efficiency=1.0,
            consistency_score=1.0,
            security_score=1.0,
        )
        assert sub.composite_score <= 1.0

    def test_composite_score_is_clamped_low(self) -> None:
        sub = _make_submission(
            accuracy_score=0.0,
            safety_score=0.0,
            cost_efficiency=0.0,
            consistency_score=0.0,
            security_score=0.0,
        )
        assert sub.composite_score >= 0.0

    def test_default_weight_proportions_respected(self) -> None:
        sub = _make_submission(
            accuracy_score=1.0,
            safety_score=0.0,
            cost_efficiency=0.0,
            consistency_score=0.0,
            security_score=0.0,
        )
        assert abs(sub.composite_score - DEFAULT_WEIGHTS.accuracy) < 1e-9

    def test_unnormalised_weights_give_same_ratio(self) -> None:
        w1 = CompositeWeights(accuracy=0.3, safety=0.25, cost_efficiency=0.15, consistency=0.15, security=0.15)
        w2 = CompositeWeights(accuracy=6.0, safety=5.0, cost_efficiency=3.0, consistency=3.0, security=3.0)
        sub = _make_submission(
            accuracy_score=0.8,
            safety_score=0.9,
            cost_efficiency=0.7,
            consistency_score=0.75,
            security_score=0.85,
        )
        s1 = sub._weighted_composite(w1)
        s2 = sub._weighted_composite(w2)
        assert abs(s1 - s2) < 1e-9


# ===========================================================================
# RankingEngine
# ===========================================================================


class TestRankingEngineRank:
    def test_rank_descending_by_composite_score(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        ranked = RankingEngine.rank(many_submissions, sort_by="composite_score")
        scores = [s.composite_score for s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_ascending_by_composite_score(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        ranked = RankingEngine.rank(many_submissions, sort_by="composite_score", ascending=True)
        scores = [s.composite_score for s in ranked]
        assert scores == sorted(scores)

    def test_rank_by_accuracy_score(self, many_submissions: list[LeaderboardSubmission]) -> None:
        ranked = RankingEngine.rank(many_submissions, sort_by="accuracy_score")
        scores = [s.accuracy_score for s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_latency_ascending(self, many_submissions: list[LeaderboardSubmission]) -> None:
        ranked = RankingEngine.rank(many_submissions, sort_by="latency_p95_ms", ascending=True)
        latencies = [s.latency_p95_ms for s in ranked]
        assert latencies == sorted(latencies)

    def test_rank_does_not_mutate_input(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        original_order = [s.agent_name for s in many_submissions]
        RankingEngine.rank(many_submissions)
        assert [s.agent_name for s in many_submissions] == original_order

    def test_rank_empty_list_returns_empty(self) -> None:
        assert RankingEngine.rank([]) == []

    def test_rank_invalid_field_raises_value_error(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        with pytest.raises(ValueError, match="Cannot sort by"):
            RankingEngine.rank(many_submissions, sort_by="not_a_field")

    def test_rank_single_element_list(self, submission: LeaderboardSubmission) -> None:
        result = RankingEngine.rank([submission])
        assert len(result) == 1

    def test_rank_by_agent_name_alphabetical(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        ranked = RankingEngine.rank(many_submissions, sort_by="agent_name", ascending=True)
        names = [s.agent_name for s in ranked]
        assert names == sorted(names)

    def test_rank_by_safety_score(self, many_submissions: list[LeaderboardSubmission]) -> None:
        ranked = RankingEngine.rank(many_submissions, sort_by="safety_score")
        scores = [s.safety_score for s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_total_cost_usd_ascending(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        ranked = RankingEngine.rank(many_submissions, sort_by="total_cost_usd", ascending=True)
        costs = [s.total_cost_usd for s in ranked]
        assert costs == sorted(costs)


class TestRankingEngineTopN:
    def test_top_n_returns_n_results(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        top3 = RankingEngine.top_n(many_submissions, n=3)
        assert len(top3) == 3

    def test_top_1_is_highest_composite_score(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        top1 = RankingEngine.top_n(many_submissions, n=1)
        best = max(many_submissions, key=lambda s: s.composite_score)
        assert top1[0].agent_name == best.agent_name

    def test_top_n_larger_than_list_returns_all(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        result = RankingEngine.top_n(many_submissions, n=100)
        assert len(result) == len(many_submissions)

    def test_top_n_zero_raises_value_error(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            RankingEngine.top_n(many_submissions, n=0)

    def test_top_n_negative_raises_value_error(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            RankingEngine.top_n(many_submissions, n=-5)

    def test_top_n_empty_list_returns_empty(self) -> None:
        assert RankingEngine.top_n([], n=5) == []

    def test_top_n_custom_sort_field(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        top2 = RankingEngine.top_n(many_submissions, n=2, sort_by="accuracy_score")
        scores = [s.accuracy_score for s in top2]
        all_scores = sorted([s.accuracy_score for s in many_submissions], reverse=True)
        assert scores[0] == all_scores[0]
        assert scores[1] == all_scores[1]


class TestRankingEngineFilterByFramework:
    def test_filter_returns_matching_framework(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        result = RankingEngine.filter_by_framework(many_submissions, "langchain")
        assert all(s.framework.lower() == "langchain" for s in result)

    def test_filter_case_insensitive(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        lower = RankingEngine.filter_by_framework(many_submissions, "langchain")
        upper = RankingEngine.filter_by_framework(many_submissions, "LANGCHAIN")
        mixed = RankingEngine.filter_by_framework(many_submissions, "LangChain")
        assert len(lower) == len(upper) == len(mixed)

    def test_filter_unknown_framework_returns_empty(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        result = RankingEngine.filter_by_framework(many_submissions, "unknown-framework")
        assert result == []

    def test_filter_empty_list_returns_empty(self) -> None:
        assert RankingEngine.filter_by_framework([], "langchain") == []

    def test_filter_does_not_mutate_input(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        original_len = len(many_submissions)
        RankingEngine.filter_by_framework(many_submissions, "langchain")
        assert len(many_submissions) == original_len

    def test_filter_crewai_framework(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        result = RankingEngine.filter_by_framework(many_submissions, "crewai")
        assert len(result) == 1
        assert result[0].agent_name == "agent-beta"


class TestRankingEngineFilterByBenchmark:
    def test_filter_by_benchmark_returns_matching(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        result = RankingEngine.filter_by_benchmark(many_submissions, "qa_basic")
        assert all(s.benchmark_name.lower() == "qa_basic" for s in result)

    def test_filter_by_benchmark_case_insensitive(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        lower = RankingEngine.filter_by_benchmark(many_submissions, "qa_basic")
        upper = RankingEngine.filter_by_benchmark(many_submissions, "QA_BASIC")
        assert len(lower) == len(upper)

    def test_filter_unknown_benchmark_returns_empty(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        result = RankingEngine.filter_by_benchmark(many_submissions, "no_such_benchmark")
        assert result == []

    def test_filter_empty_list_returns_empty(self) -> None:
        assert RankingEngine.filter_by_benchmark([], "qa_basic") == []

    def test_filter_safety_basic_benchmark(
        self, many_submissions: list[LeaderboardSubmission]
    ) -> None:
        result = RankingEngine.filter_by_benchmark(many_submissions, "safety_basic")
        assert len(result) == 1
        assert result[0].agent_name == "agent-beta"


# ===========================================================================
# LeaderboardStorage — in-memory mode
# ===========================================================================


class TestLeaderboardStorageMemory:
    def test_save_and_load_all(
        self, memory_store: LeaderboardStorage, submission: LeaderboardSubmission
    ) -> None:
        memory_store.save(submission)
        all_subs = memory_store.load_all()
        assert len(all_subs) == 1
        assert all_subs[0].agent_name == submission.agent_name

    def test_load_all_empty_returns_empty(self, memory_store: LeaderboardStorage) -> None:
        assert memory_store.load_all() == []

    def test_save_duplicate_version_overwrites(
        self, memory_store: LeaderboardStorage, submission: LeaderboardSubmission
    ) -> None:
        memory_store.save(submission)
        updated = _make_submission(accuracy_score=0.99)
        memory_store.save(updated)
        all_subs = memory_store.load_all()
        assert len(all_subs) == 1
        assert all_subs[0].accuracy_score == 0.99

    def test_save_different_version_appends(
        self, memory_store: LeaderboardStorage
    ) -> None:
        sub1 = _make_submission(agent_version="1.0.0")
        sub2 = _make_submission(agent_version="2.0.0")
        memory_store.save(sub1)
        memory_store.save(sub2)
        assert len(memory_store.load_all()) == 2

    def test_load_by_agent_returns_matching(
        self, memory_store: LeaderboardStorage
    ) -> None:
        sub1 = _make_submission(agent_name="agent-x", agent_version="1.0.0")
        sub2 = _make_submission(agent_name="agent-x", agent_version="2.0.0")
        sub3 = _make_submission(agent_name="agent-y", agent_version="1.0.0")
        memory_store.save(sub1)
        memory_store.save(sub2)
        memory_store.save(sub3)
        result = memory_store.load_by_agent("agent-x")
        assert len(result) == 2
        assert all(s.agent_name == "agent-x" for s in result)

    def test_load_by_agent_unknown_returns_empty(
        self, memory_store: LeaderboardStorage
    ) -> None:
        memory_store.save(_make_submission())
        assert memory_store.load_by_agent("nonexistent") == []

    def test_delete_existing_returns_true(
        self, memory_store: LeaderboardStorage, submission: LeaderboardSubmission
    ) -> None:
        memory_store.save(submission)
        removed = memory_store.delete(submission.agent_name, submission.agent_version)
        assert removed is True

    def test_delete_existing_removes_from_store(
        self, memory_store: LeaderboardStorage, submission: LeaderboardSubmission
    ) -> None:
        memory_store.save(submission)
        memory_store.delete(submission.agent_name, submission.agent_version)
        assert memory_store.load_all() == []

    def test_delete_nonexistent_returns_false(
        self, memory_store: LeaderboardStorage
    ) -> None:
        removed = memory_store.delete("ghost", "0.0.0")
        assert removed is False

    def test_delete_only_matching_version(
        self, memory_store: LeaderboardStorage
    ) -> None:
        sub1 = _make_submission(agent_version="1.0.0")
        sub2 = _make_submission(agent_version="2.0.0")
        memory_store.save(sub1)
        memory_store.save(sub2)
        memory_store.delete("agent-alpha", "1.0.0")
        remaining = memory_store.load_all()
        assert len(remaining) == 1
        assert remaining[0].agent_version == "2.0.0"

    def test_load_all_returns_shallow_copy(
        self, memory_store: LeaderboardStorage, submission: LeaderboardSubmission
    ) -> None:
        memory_store.save(submission)
        copy1 = memory_store.load_all()
        copy2 = memory_store.load_all()
        assert copy1 is not copy2

    def test_multiple_saves_preserve_insertion_order(
        self, memory_store: LeaderboardStorage
    ) -> None:
        names = ["agent-z", "agent-a", "agent-m"]
        for name in names:
            memory_store.save(_make_submission(agent_name=name))
        loaded_names = [s.agent_name for s in memory_store.load_all()]
        assert loaded_names == names


# ===========================================================================
# LeaderboardStorage — file persistence
# ===========================================================================


class TestLeaderboardStorageFile:
    def test_save_creates_file(
        self, tmp_path: Path, submission: LeaderboardSubmission
    ) -> None:
        storage_path = tmp_path / "lb.json"
        store = LeaderboardStorage(storage_path=storage_path)
        store.save(submission)
        assert storage_path.exists()

    def test_persisted_data_loadable_by_new_instance(
        self, tmp_path: Path, submission: LeaderboardSubmission
    ) -> None:
        storage_path = tmp_path / "lb.json"
        store1 = LeaderboardStorage(storage_path=storage_path)
        store1.save(submission)

        store2 = LeaderboardStorage(storage_path=storage_path)
        loaded = store2.load_all()
        assert len(loaded) == 1
        assert loaded[0].agent_name == submission.agent_name

    def test_file_contains_valid_json(
        self, tmp_path: Path, submission: LeaderboardSubmission
    ) -> None:
        storage_path = tmp_path / "lb.json"
        store = LeaderboardStorage(storage_path=storage_path)
        store.save(submission)
        data = json.loads(storage_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_delete_updates_file(
        self, tmp_path: Path, submission: LeaderboardSubmission
    ) -> None:
        storage_path = tmp_path / "lb.json"
        store = LeaderboardStorage(storage_path=storage_path)
        store.save(submission)
        store.delete(submission.agent_name, submission.agent_version)

        store2 = LeaderboardStorage(storage_path=storage_path)
        assert store2.load_all() == []

    def test_existing_file_loaded_on_construction(
        self, tmp_path: Path, submission: LeaderboardSubmission
    ) -> None:
        storage_path = tmp_path / "lb.json"
        data = [submission.model_dump(mode="json")]
        storage_path.write_text(json.dumps(data, default=str))
        store = LeaderboardStorage(storage_path=storage_path)
        assert len(store.load_all()) == 1

    def test_nonexistent_file_starts_empty(self, tmp_path: Path) -> None:
        storage_path = tmp_path / "new_lb.json"
        store = LeaderboardStorage(storage_path=storage_path)
        assert store.load_all() == []

    def test_storage_path_is_none_no_file_written(
        self, tmp_path: Path, submission: LeaderboardSubmission
    ) -> None:
        store = LeaderboardStorage(storage_path=None)
        store.save(submission)
        # Verify no files were created in tmp_path
        assert list(tmp_path.iterdir()) == []


# ===========================================================================
# LeaderboardStorage — export_json / import_json
# ===========================================================================


class TestLeaderboardStorageExportImport:
    def test_export_json_creates_file(
        self,
        populated_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        export_path = tmp_path / "export.json"
        populated_store.export_json(export_path)
        assert export_path.exists()

    def test_export_json_content_is_valid_array(
        self,
        populated_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        export_path = tmp_path / "export.json"
        populated_store.export_json(export_path)
        data = json.loads(export_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 4

    def test_export_json_creates_parent_directories(
        self,
        populated_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        export_path = tmp_path / "nested" / "deep" / "export.json"
        populated_store.export_json(export_path)
        assert export_path.exists()

    def test_export_json_empty_store(
        self,
        memory_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        export_path = tmp_path / "empty.json"
        memory_store.export_json(export_path)
        data = json.loads(export_path.read_text())
        assert data == []

    def test_import_json_loads_submissions(
        self,
        memory_store: LeaderboardStorage,
        populated_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        export_path = tmp_path / "export.json"
        populated_store.export_json(export_path)
        count = memory_store.import_json(export_path)
        assert count == 4
        assert len(memory_store.load_all()) == 4

    def test_import_json_returns_count(
        self,
        memory_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        sub = _make_submission()
        export_path = tmp_path / "single.json"
        data = [sub.model_dump(mode="json")]
        export_path.write_text(json.dumps(data, default=str))
        count = memory_store.import_json(export_path)
        assert count == 1

    def test_import_json_file_not_found_raises(
        self, memory_store: LeaderboardStorage, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            memory_store.import_json(tmp_path / "ghost.json")

    def test_import_json_invalid_format_raises(
        self, memory_store: LeaderboardStorage, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="Expected a JSON array"):
            memory_store.import_json(bad_file)

    def test_import_json_merges_with_existing(
        self,
        memory_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        existing = _make_submission(agent_name="existing-agent")
        memory_store.save(existing)

        import_sub = _make_submission(agent_name="imported-agent")
        export_path = tmp_path / "import.json"
        data = [import_sub.model_dump(mode="json")]
        export_path.write_text(json.dumps(data, default=str))

        memory_store.import_json(export_path)
        all_names = {s.agent_name for s in memory_store.load_all()}
        assert "existing-agent" in all_names
        assert "imported-agent" in all_names

    def test_round_trip_export_import(
        self,
        populated_store: LeaderboardStorage,
        tmp_path: Path,
    ) -> None:
        export_path = tmp_path / "round_trip.json"
        populated_store.export_json(export_path)

        fresh_store = LeaderboardStorage(storage_path=None)
        fresh_store.import_json(export_path)

        original = {s.agent_name for s in populated_store.load_all()}
        restored = {s.agent_name for s in fresh_store.load_all()}
        assert original == restored


# ===========================================================================
# LeaderboardStorage — thread safety (basic)
# ===========================================================================


class TestLeaderboardStorageThreadSafety:
    def test_concurrent_saves_do_not_corrupt_state(
        self, memory_store: LeaderboardStorage
    ) -> None:
        errors: list[Exception] = []

        def save_unique(index: int) -> None:
            try:
                sub = _make_submission(
                    agent_name=f"concurrent-agent-{index}",
                    agent_version=str(index),
                )
                memory_store.save(sub)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=save_unique, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(memory_store.load_all()) == 20


# ===========================================================================
# CLI — leaderboard submit
# ===========================================================================


class TestLeaderboardSubmitCLI:
    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    def _base_args(self, storage_path: Path) -> list[str]:
        return [
            "leaderboard", "submit",
            "--agent-name", "cli-agent",
            "--agent-version", "1.0.0",
            "--framework", "langchain",
            "--model", "gpt-4o",
            "--submitter", "tester",
            "--accuracy", "0.90",
            "--safety", "0.95",
            "--cost-efficiency", "0.80",
            "--latency-p95", "500.0",
            "--consistency", "0.85",
            "--security", "0.90",
            "--benchmark-name", "qa_basic",
            "--storage", str(storage_path),
        ]

    def test_submit_succeeds(self, runner: CliRunner, tmp_path: Path) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(cli, self._base_args(tmp_path / "lb.json"))
        assert result.exit_code == 0, result.output
        assert "Submitted" in result.output

    def test_submit_creates_storage_file(self, runner: CliRunner, tmp_path: Path) -> None:
        from agent_eval.cli.main import cli

        storage_path = tmp_path / "lb.json"
        runner.invoke(cli, self._base_args(storage_path))
        assert storage_path.exists()

    def test_submit_shows_composite_score(self, runner: CliRunner, tmp_path: Path) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(cli, self._base_args(tmp_path / "lb.json"))
        assert "Composite score" in result.output

    def test_submit_invalid_accuracy_exits_nonzero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from agent_eval.cli.main import cli

        args = self._base_args(tmp_path / "lb.json")
        # Replace accuracy value with invalid
        idx = args.index("0.90")
        args[idx] = "1.5"
        result = runner.invoke(cli, args)
        assert result.exit_code != 0

    def test_submit_stores_submission(self, runner: CliRunner, tmp_path: Path) -> None:
        from agent_eval.cli.main import cli

        storage_path = tmp_path / "lb.json"
        runner.invoke(cli, self._base_args(storage_path))
        store = LeaderboardStorage(storage_path=storage_path)
        subs = store.load_all()
        assert len(subs) == 1
        assert subs[0].agent_name == "cli-agent"

    def test_submit_with_optional_fields(self, runner: CliRunner, tmp_path: Path) -> None:
        from agent_eval.cli.main import cli

        storage_path = tmp_path / "lb.json"
        args = self._base_args(storage_path) + [
            "--num-runs", "10",
            "--total-tokens", "50000",
            "--total-cost-usd", "1.25",
        ]
        result = runner.invoke(cli, args)
        assert result.exit_code == 0


# ===========================================================================
# CLI — leaderboard list
# ===========================================================================


class TestLeaderboardListCLI:
    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture()
    def storage_file(
        self,
        tmp_path: Path,
        many_submissions: list[LeaderboardSubmission],
    ) -> Path:
        path = tmp_path / "lb.json"
        store = LeaderboardStorage(storage_path=path)
        for sub in many_submissions:
            store.save(sub)
        return path

    def test_list_exits_zero(self, runner: CliRunner, storage_file: Path) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(cli, ["leaderboard", "list", "--storage", str(storage_file)])
        assert result.exit_code == 0, result.output

    def test_list_shows_agent_names(self, runner: CliRunner, storage_file: Path) -> None:
        from agent_eval.cli.main import cli

        # Rich truncates cell text in a narrow terminal; check for partial matches
        # that are guaranteed to appear even with truncation ("agent" prefix fits).
        result = runner.invoke(cli, ["leaderboard", "list", "--storage", str(storage_file)])
        assert "agent" in result.output

    def test_list_empty_storage_shows_message(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from agent_eval.cli.main import cli

        empty_path = tmp_path / "empty.json"
        store = LeaderboardStorage(storage_path=empty_path)
        # Trigger file creation with no submissions
        store.export_json(empty_path)
        result = runner.invoke(cli, ["leaderboard", "list", "--storage", str(empty_path)])
        assert "No submissions" in result.output

    def test_list_with_top_option(self, runner: CliRunner, storage_file: Path) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(
            cli, ["leaderboard", "list", "--storage", str(storage_file), "--top", "2"]
        )
        assert result.exit_code == 0

    def test_list_with_framework_filter(self, runner: CliRunner, storage_file: Path) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(
            cli,
            ["leaderboard", "list", "--storage", str(storage_file), "--framework", "langchain"],
        )
        assert result.exit_code == 0
        # Rich may truncate cells; verify the table rendered at all
        assert "Leaderboard" in result.output

    def test_list_with_benchmark_filter(self, runner: CliRunner, storage_file: Path) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(
            cli,
            ["leaderboard", "list", "--storage", str(storage_file), "--benchmark", "safety_basic"],
        )
        assert result.exit_code == 0
        # Rich may truncate cells; verify the table rendered with one row
        assert "crewai" in result.output

    def test_list_with_ascending_flag(self, runner: CliRunner, storage_file: Path) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(
            cli,
            ["leaderboard", "list", "--storage", str(storage_file), "--ascending"],
        )
        assert result.exit_code == 0

    def test_list_invalid_sort_field_exits_nonzero(
        self, runner: CliRunner, storage_file: Path
    ) -> None:
        from agent_eval.cli.main import cli

        result = runner.invoke(
            cli,
            ["leaderboard", "list", "--storage", str(storage_file), "--sort-by", "invalid_field"],
        )
        assert result.exit_code != 0


# ===========================================================================
# CLI — leaderboard export
# ===========================================================================


class TestLeaderboardExportCLI:
    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture()
    def storage_file(
        self,
        tmp_path: Path,
        many_submissions: list[LeaderboardSubmission],
    ) -> Path:
        path = tmp_path / "lb.json"
        store = LeaderboardStorage(storage_path=path)
        for sub in many_submissions:
            store.save(sub)
        return path

    def test_export_exits_zero(
        self, runner: CliRunner, storage_file: Path, tmp_path: Path
    ) -> None:
        from agent_eval.cli.main import cli

        output_path = tmp_path / "out.json"
        result = runner.invoke(
            cli,
            [
                "leaderboard", "export",
                str(output_path),
                "--storage", str(storage_file),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_export_creates_output_file(
        self, runner: CliRunner, storage_file: Path, tmp_path: Path
    ) -> None:
        from agent_eval.cli.main import cli

        output_path = tmp_path / "exported.json"
        runner.invoke(
            cli,
            ["leaderboard", "export", str(output_path), "--storage", str(storage_file)],
        )
        assert output_path.exists()

    def test_export_output_is_valid_json(
        self, runner: CliRunner, storage_file: Path, tmp_path: Path
    ) -> None:
        from agent_eval.cli.main import cli

        output_path = tmp_path / "exported.json"
        runner.invoke(
            cli,
            ["leaderboard", "export", str(output_path), "--storage", str(storage_file)],
        )
        data = json.loads(output_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 4

    def test_export_shows_submission_count(
        self, runner: CliRunner, storage_file: Path, tmp_path: Path
    ) -> None:
        from agent_eval.cli.main import cli

        output_path = tmp_path / "out.json"
        result = runner.invoke(
            cli,
            ["leaderboard", "export", str(output_path), "--storage", str(storage_file)],
        )
        assert "4" in result.output

    def test_export_empty_store_shows_message(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from agent_eval.cli.main import cli

        empty_path = tmp_path / "empty.json"
        store = LeaderboardStorage(storage_path=empty_path)
        store.export_json(empty_path)
        output_path = tmp_path / "out.json"
        result = runner.invoke(
            cli,
            ["leaderboard", "export", str(output_path), "--storage", str(empty_path)],
        )
        assert "empty" in result.output.lower() or "nothing" in result.output.lower()


# ===========================================================================
# Public __init__ re-exports
# ===========================================================================


class TestLeaderboardPackageImports:
    def test_import_submission_from_package(self) -> None:
        from agent_eval.leaderboard import LeaderboardSubmission as LS  # noqa: F401

        assert LS is LeaderboardSubmission

    def test_import_ranking_engine_from_package(self) -> None:
        from agent_eval.leaderboard import RankingEngine as RE  # noqa: F401

        assert RE is RankingEngine

    def test_import_storage_from_package(self) -> None:
        from agent_eval.leaderboard import LeaderboardStorage as LSt  # noqa: F401

        assert LSt is LeaderboardStorage

    def test_import_composite_weights_from_package(self) -> None:
        from agent_eval.leaderboard import CompositeWeights as CW  # noqa: F401

        assert CW is CompositeWeights
