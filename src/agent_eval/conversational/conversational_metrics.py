"""Conversational metrics — multi-turn conversation quality evaluation.

Implements commodity-algorithm metrics that assess the quality of a
multi-turn agent conversation across three dimensions:

coherence
    Measures topical consistency between consecutive turns using
    normalised bigram-overlap (Jaccard coefficient).  Score in [0, 1].

relevance
    Measures how well assistant responses address a reference query
    using keyword-overlap between each assistant turn and the query.
    Score in [0, 1].

resolution
    Measures whether the conversation reached a clear resolution by
    checking the final assistant turn for resolution indicator phrases.
    Score is 0.0 or 1.0.

All algorithms are commodity / well-known NLP heuristics.  No ML models,
no embeddings, no proprietary scoring logic.

Classes
-------
Turn
    A single conversational exchange.
ConversationReport
    Aggregate evaluation result for a full conversation.
ConversationEvaluator
    Runs all three metrics and assembles a ConversationReport.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single turn in a multi-turn conversation.

    Attributes
    ----------
    role:
        Speaker role, conventionally ``"user"`` or ``"assistant"``.
    content:
        The text content of this turn.
    timestamp:
        Optional ISO-8601 timestamp string (informational only; not used
        in scoring calculations).
    """

    role: str
    content: str
    timestamp: str | None = None


@dataclass
class ConversationReport:
    """Aggregate evaluation result for a full conversation.

    Attributes
    ----------
    coherence_score:
        Bigram-overlap coherence across consecutive turns.  Range [0, 1].
        Higher is more coherent.
    relevance_score:
        Keyword-overlap relevance of assistant turns to the reference
        query.  Range [0, 1].  Higher is more on-topic.
    resolution_score:
        Whether the conversation ended with a resolution indicator.
        Either 0.0 (unresolved) or 1.0 (resolved).
    turn_count:
        Total number of turns in the evaluated conversation.
    query:
        The reference query used for relevance scoring, or ``None`` if
        relevance was not computed.
    per_turn_coherence:
        Coherence score for each consecutive pair of turns.
    """

    coherence_score: float
    relevance_score: float
    resolution_score: float
    turn_count: int
    query: str | None = None
    per_turn_coherence: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers — text normalisation
# ---------------------------------------------------------------------------

_WORD_SPLITTER = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-tokenize *text*, ignoring punctuation.

    Parameters
    ----------
    text:
        Raw text to tokenize.

    Returns
    -------
    list[str]
        List of lowercase word tokens.
    """
    return _WORD_SPLITTER.findall(text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Return the set of consecutive bigrams from a token list.

    Parameters
    ----------
    tokens:
        Ordered list of word tokens.

    Returns
    -------
    set[tuple[str, str]]
        All consecutive two-word pairs.  Empty set if fewer than 2 tokens.
    """
    if len(tokens) < 2:
        return set()
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _jaccard(set_a: set[object], set_b: set[object]) -> float:
    """Compute the Jaccard similarity coefficient between two sets.

    Returns 0.0 when both sets are empty (avoids zero-division).

    Parameters
    ----------
    set_a:
        First set.
    set_b:
        Second set.

    Returns
    -------
    float
        Intersection / Union.  Range [0, 1].
    """
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Resolution keywords
# ---------------------------------------------------------------------------

_RESOLUTION_PHRASES: list[str] = [
    "hope that helps",
    "let me know if",
    "feel free to ask",
    "is there anything else",
    "happy to help",
    "glad i could help",
    "you're welcome",
    "problem solved",
    "that should do it",
    "resolved",
    "done",
    "completed",
    "in summary",
    "to summarize",
    "in conclusion",
    "that answers",
    "hope this answers",
    "does that answer",
    "please let me know",
]


def _check_resolution(text: str) -> bool:
    """Return True if *text* contains at least one resolution indicator phrase.

    Parameters
    ----------
    text:
        The text of the final assistant turn (lowercased internally).

    Returns
    -------
    bool
        True when at least one resolution phrase is found.
    """
    lowered = text.lower()
    return any(phrase in lowered for phrase in _RESOLUTION_PHRASES)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ConversationEvaluator:
    """Evaluate multi-turn conversation quality across three dimensions.

    All algorithms are commodity (bigram Jaccard, keyword overlap, phrase
    matching).  No model calls are made; scoring is fully deterministic.

    Example
    -------
    ::

        evaluator = ConversationEvaluator()
        report = evaluator.evaluate(turns, query="How do I reset my password?")
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        turns: Sequence[Turn],
        query: str | None = None,
    ) -> ConversationReport:
        """Run all metrics and return an aggregate ConversationReport.

        Parameters
        ----------
        turns:
            Ordered list of conversation turns, oldest first.
        query:
            Optional reference query for relevance scoring.  When ``None``
            the relevance score is computed against the first user turn,
            or 0.0 if no user turns exist.

        Returns
        -------
        ConversationReport
            Scores for coherence, relevance, and resolution, plus metadata.
        """
        turn_list = list(turns)

        per_turn_coherence, coherence = self._compute_coherence(turn_list)
        effective_query = query or self._first_user_content(turn_list) or ""
        relevance = self.relevance_score(turn_list, effective_query)
        resolution = self.resolution_score(turn_list)

        return ConversationReport(
            coherence_score=coherence,
            relevance_score=relevance,
            resolution_score=resolution,
            turn_count=len(turn_list),
            query=effective_query if effective_query else None,
            per_turn_coherence=per_turn_coherence,
        )

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def coherence_score(self, turns: Sequence[Turn]) -> float:
        """Measure topic consistency across consecutive turns.

        Algorithm: compute bigram-Jaccard similarity between each pair of
        adjacent turns, then return the mean.  A single-turn or empty
        conversation returns 0.0.

        Parameters
        ----------
        turns:
            Ordered list of turns.

        Returns
        -------
        float
            Mean bigram Jaccard similarity.  Range [0, 1].
        """
        _, score = self._compute_coherence(list(turns))
        return score

    def relevance_score(self, turns: Sequence[Turn], query: str) -> float:
        """Measure how well assistant responses address *query*.

        Algorithm: for each assistant turn compute the token-level recall
        of *query* keywords appearing in that turn, then return the mean
        across all assistant turns.  Returns 0.0 when there are no
        assistant turns or no query tokens.

        Parameters
        ----------
        turns:
            Ordered list of turns.
        query:
            Reference query or topic string.

        Returns
        -------
        float
            Mean keyword recall across assistant turns.  Range [0, 1].
        """
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return 0.0

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return 0.0

        scores: list[float] = []
        for turn in assistant_turns:
            turn_tokens = set(_tokenize(turn.content))
            if not turn_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & turn_tokens) / len(query_tokens)
            scores.append(min(overlap, 1.0))

        return sum(scores) / len(scores)

    def resolution_score(self, turns: Sequence[Turn]) -> float:
        """Measure whether the conversation reached a clear resolution.

        Algorithm: scan the final assistant turn for resolution indicator
        phrases.  Score is 1.0 if resolved, 0.0 otherwise.

        Parameters
        ----------
        turns:
            Ordered list of turns.

        Returns
        -------
        float
            1.0 if the final assistant turn contains a resolution phrase,
            0.0 otherwise.
        """
        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return 0.0
        final_turn = assistant_turns[-1]
        return 1.0 if _check_resolution(final_turn.content) else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_coherence(
        self,
        turns: list[Turn],
    ) -> tuple[list[float], float]:
        """Compute per-turn pair coherence scores and their mean.

        Parameters
        ----------
        turns:
            List of turns.

        Returns
        -------
        tuple[list[float], float]
            A list of per-consecutive-pair scores and the overall mean.
        """
        if len(turns) < 2:
            return [], 0.0

        pair_scores: list[float] = []
        for i in range(len(turns) - 1):
            tokens_a = _tokenize(turns[i].content)
            tokens_b = _tokenize(turns[i + 1].content)
            bigrams_a = _bigrams(tokens_a)
            bigrams_b = _bigrams(tokens_b)
            score = _jaccard(bigrams_a, bigrams_b)  # type: ignore[arg-type]
            pair_scores.append(score)

        mean_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        return pair_scores, mean_score

    @staticmethod
    def _first_user_content(turns: list[Turn]) -> str | None:
        """Return the content of the first user turn, or None."""
        for turn in turns:
            if turn.role == "user":
                return turn.content
        return None
