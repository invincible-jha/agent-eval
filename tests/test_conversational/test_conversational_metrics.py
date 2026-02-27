"""Tests for ConversationEvaluator — multi-turn conversation quality metrics.

Covers:
- Turn dataclass construction and attributes
- coherence_score() — identical turns, distinct turns, single turn, empty
- relevance_score() — full match, partial match, no match, no assistant turns
- resolution_score() — various resolution phrases, no resolution, no assistant
- evaluate() — full report, auto-query from first user turn, explicit query
- ConversationReport fields and properties
- Edge cases — empty conversation, single turn, all user turns, all assistant
- Internal helpers — _tokenize, _bigrams, _jaccard, _check_resolution
"""
from __future__ import annotations

import pytest

from agent_eval.conversational import (
    ConversationEvaluator,
    ConversationReport,
    Turn,
)
from agent_eval.conversational.conversational_metrics import (
    _bigrams,
    _check_resolution,
    _jaccard,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turns(*args: tuple[str, str]) -> list[Turn]:
    """Build a list of Turn objects from (role, content) pairs."""
    return [Turn(role=role, content=content) for role, content in args]


def _user_assistant_pair(
    user_content: str, assistant_content: str
) -> list[Turn]:
    return [
        Turn(role="user", content=user_content),
        Turn(role="assistant", content=assistant_content),
    ]


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_sentence(self) -> None:
        tokens = _tokenize("Hello world!")
        assert tokens == ["hello", "world"]

    def test_punctuation_stripped(self) -> None:
        tokens = _tokenize("cats, dogs, and birds.")
        assert "cats" in tokens and "dogs" in tokens

    def test_lowercased(self) -> None:
        tokens = _tokenize("Python Is GREAT")
        assert tokens == ["python", "is", "great"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_numbers_included(self) -> None:
        tokens = _tokenize("answer is 42")
        assert "42" in tokens


# ---------------------------------------------------------------------------
# _bigrams
# ---------------------------------------------------------------------------


class TestBigrams:
    def test_simple_bigrams(self) -> None:
        bigrams = _bigrams(["a", "b", "c"])
        assert ("a", "b") in bigrams
        assert ("b", "c") in bigrams

    def test_single_token_returns_empty(self) -> None:
        assert _bigrams(["a"]) == set()

    def test_empty_returns_empty(self) -> None:
        assert _bigrams([]) == set()

    def test_duplicates_handled(self) -> None:
        bigrams = _bigrams(["a", "b", "a", "b"])
        # sets dedup — ("a","b") appears twice but is one set element
        assert ("a", "b") in bigrams


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_sets(self) -> None:
        s = {1, 2, 3}
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        assert _jaccard({1, 2}, {3, 4}) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = _jaccard({1, 2, 3}, {2, 3, 4})
        # intersection=2, union=4 → 0.5
        assert score == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        assert _jaccard(set(), set()) == pytest.approx(0.0)

    def test_one_empty(self) -> None:
        assert _jaccard({1}, set()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _check_resolution
# ---------------------------------------------------------------------------


class TestCheckResolution:
    def test_hope_that_helps(self) -> None:
        assert _check_resolution("I hope that helps!")

    def test_let_me_know_if(self) -> None:
        assert _check_resolution("Let me know if you need anything else.")

    def test_in_summary(self) -> None:
        assert _check_resolution("In summary, the answer is Paris.")

    def test_done(self) -> None:
        assert _check_resolution("Done.")

    def test_no_resolution_phrase(self) -> None:
        assert not _check_resolution("The capital of France is Paris.")

    def test_case_insensitive(self) -> None:
        assert _check_resolution("HOPE THAT HELPS!")

    def test_empty_string(self) -> None:
        assert not _check_resolution("")


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------


class TestTurn:
    def test_basic_construction(self) -> None:
        turn = Turn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.timestamp is None

    def test_with_timestamp(self) -> None:
        turn = Turn(role="assistant", content="Hi", timestamp="2026-01-01T00:00:00Z")
        assert turn.timestamp == "2026-01-01T00:00:00Z"

    def test_any_role_accepted(self) -> None:
        turn = Turn(role="system", content="You are a helpful assistant.")
        assert turn.role == "system"


# ---------------------------------------------------------------------------
# coherence_score()
# ---------------------------------------------------------------------------


class TestCoherenceScore:
    def test_identical_turns_high_score(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "The weather today is sunny and warm."),
            Turn("assistant", "The weather today is sunny and warm."),
        ]
        score = evaluator.coherence_score(turns)
        assert score > 0.8

    def test_unrelated_turns_low_score(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "What is quantum mechanics?"),
            Turn("assistant", "Pizza delivery in New York costs ten dollars."),
        ]
        score = evaluator.coherence_score(turns)
        assert score < 0.3

    def test_single_turn_returns_zero(self) -> None:
        evaluator = ConversationEvaluator()
        score = evaluator.coherence_score([Turn("user", "Hello!")])
        assert score == 0.0

    def test_empty_turns_returns_zero(self) -> None:
        evaluator = ConversationEvaluator()
        assert evaluator.coherence_score([]) == 0.0

    def test_score_in_range(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(
            ("user", "How do I reset my password?"),
            ("assistant", "To reset your password, click 'Forgot Password'."),
            ("user", "Where is the forgot password link?"),
            ("assistant", "The link is on the login page below the form."),
        )
        score = evaluator.coherence_score(turns)
        assert 0.0 <= score <= 1.0

    def test_multi_turn_averages_pairs(self) -> None:
        evaluator = ConversationEvaluator()
        # 3 turns → 2 pairs
        turns = [
            Turn("user", "cat sat mat"),
            Turn("assistant", "cat sat mat"),
            Turn("user", "dog ran park"),
        ]
        # pair 1: identical → 1.0; pair 2: unrelated → ~0.0
        score = evaluator.coherence_score(turns)
        assert score > 0.0
        assert score < 1.0


# ---------------------------------------------------------------------------
# relevance_score()
# ---------------------------------------------------------------------------


class TestRelevanceScore:
    def test_full_keyword_match(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "How do I reset my password?"),
            Turn("assistant", "You can reset your password from the settings page."),
        ]
        score = evaluator.relevance_score(turns, query="reset password")
        assert score > 0.5

    def test_no_keyword_match(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [Turn("assistant", "Elephants are large mammals.")]
        score = evaluator.relevance_score(turns, query="quantum physics")
        assert score == pytest.approx(0.0)

    def test_no_assistant_turns_returns_zero(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [Turn("user", "What is the weather?")]
        score = evaluator.relevance_score(turns, query="weather")
        assert score == 0.0

    def test_empty_query_returns_zero(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [Turn("assistant", "Something helpful.")]
        score = evaluator.relevance_score(turns, query="")
        assert score == 0.0

    def test_score_in_range(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "Tell me about France."),
            Turn("assistant", "France is a country in Western Europe. Paris is its capital."),
        ]
        score = evaluator.relevance_score(turns, query="France Europe")
        assert 0.0 <= score <= 1.0

    def test_multiple_assistant_turns_averaged(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "Tell me about dogs."),
            Turn("assistant", "Dogs are loyal animals."),  # matches "dogs"
            Turn("user", "What about cats?"),
            Turn("assistant", "Pizza is delicious."),  # no match
        ]
        score = evaluator.relevance_score(turns, query="dogs")
        # First assistant has match, second does not → average > 0, < 1
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# resolution_score()
# ---------------------------------------------------------------------------


class TestResolutionScore:
    def test_clear_resolution_in_final_turn(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "Can you help me?"),
            Turn("assistant", "Of course! I hope that helps."),
        ]
        assert evaluator.resolution_score(turns) == 1.0

    def test_no_resolution_phrase(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "What is 2+2?"),
            Turn("assistant", "It equals four."),
        ]
        assert evaluator.resolution_score(turns) == 0.0

    def test_resolution_in_non_final_turn_ignored(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [
            Turn("user", "First question?"),
            Turn("assistant", "I hope that helps."),  # resolution in middle
            Turn("user", "Follow-up question?"),
            Turn("assistant", "Here is more info."),  # no resolution
        ]
        # Only final assistant turn counts
        assert evaluator.resolution_score(turns) == 0.0

    def test_no_assistant_turns_returns_zero(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [Turn("user", "Hello")]
        assert evaluator.resolution_score(turns) == 0.0

    def test_empty_turns_returns_zero(self) -> None:
        evaluator = ConversationEvaluator()
        assert evaluator.resolution_score([]) == 0.0

    def test_multiple_resolution_phrases(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [Turn("assistant", "Done. Let me know if you need more.")]
        assert evaluator.resolution_score(turns) == 1.0


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_conversation_report(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(
            ("user", "What is Python?"),
            ("assistant", "Python is a programming language. Hope that helps!"),
        )
        report = evaluator.evaluate(turns)
        assert isinstance(report, ConversationReport)

    def test_report_has_all_scores(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(
            ("user", "How does Python work?"),
            ("assistant", "Python is interpreted and dynamically typed."),
        )
        report = evaluator.evaluate(turns)
        assert 0.0 <= report.coherence_score <= 1.0
        assert 0.0 <= report.relevance_score <= 1.0
        assert report.resolution_score in (0.0, 1.0)

    def test_turn_count_correct(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(("user", "q1"), ("assistant", "a1"), ("user", "q2"))
        report = evaluator.evaluate(turns)
        assert report.turn_count == 3

    def test_auto_query_from_first_user_turn(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(
            ("user", "Tell me about Paris"),
            ("assistant", "Paris is the capital of France."),
        )
        report = evaluator.evaluate(turns)
        # query should be derived from first user turn
        assert report.query is not None
        assert "paris" in report.query.lower() or "tell" in report.query.lower()

    def test_explicit_query_used(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(
            ("user", "Hello"),
            ("assistant", "Hi there!"),
        )
        report = evaluator.evaluate(turns, query="custom query")
        assert report.query == "custom query"

    def test_per_turn_coherence_length(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(
            ("user", "one"), ("assistant", "two"), ("user", "three")
        )
        report = evaluator.evaluate(turns)
        # 3 turns → 2 pairs
        assert len(report.per_turn_coherence) == 2

    def test_empty_conversation(self) -> None:
        evaluator = ConversationEvaluator()
        report = evaluator.evaluate([])
        assert report.coherence_score == 0.0
        assert report.relevance_score == 0.0
        assert report.resolution_score == 0.0
        assert report.turn_count == 0

    def test_single_turn(self) -> None:
        evaluator = ConversationEvaluator()
        report = evaluator.evaluate([Turn("user", "Hello?")])
        assert report.coherence_score == 0.0
        assert report.turn_count == 1

    def test_all_user_turns(self) -> None:
        evaluator = ConversationEvaluator()
        turns = [Turn("user", f"message {i}") for i in range(4)]
        report = evaluator.evaluate(turns)
        # No assistant turns → relevance and resolution must be 0
        assert report.relevance_score == 0.0
        assert report.resolution_score == 0.0

    def test_resolution_detected_in_report(self) -> None:
        evaluator = ConversationEvaluator()
        turns = _turns(
            ("user", "Can you help?"),
            ("assistant", "Sure! Let me know if you need more help."),
        )
        report = evaluator.evaluate(turns)
        assert report.resolution_score == 1.0


# ---------------------------------------------------------------------------
# ConversationReport
# ---------------------------------------------------------------------------


class TestConversationReport:
    def test_construction(self) -> None:
        report = ConversationReport(
            coherence_score=0.8,
            relevance_score=0.7,
            resolution_score=1.0,
            turn_count=4,
            query="test query",
            per_turn_coherence=[0.9, 0.7, 0.8],
        )
        assert report.coherence_score == 0.8
        assert report.relevance_score == 0.7
        assert report.resolution_score == 1.0
        assert report.turn_count == 4

    def test_default_per_turn_coherence(self) -> None:
        report = ConversationReport(
            coherence_score=0.0,
            relevance_score=0.0,
            resolution_score=0.0,
            turn_count=0,
        )
        assert report.per_turn_coherence == []

    def test_default_query_is_none(self) -> None:
        report = ConversationReport(
            coherence_score=0.0,
            relevance_score=0.0,
            resolution_score=0.0,
            turn_count=0,
        )
        assert report.query is None
