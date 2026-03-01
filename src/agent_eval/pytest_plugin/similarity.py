"""Multi-strategy text similarity scoring for pytest-agent-eval.

Provides commodity NLP similarity metrics (Jaccard token overlap,
SequenceMatcher fuzzy ratio, character n-gram Jaccard) combined into a
weighted ensemble. All implementations use the Python standard library only —
no third-party dependencies required.

Strategies
----------
token_overlap
    Jaccard index over word tokens. Fast; handles synonym misses.
fuzzy
    SequenceMatcher ratio from :mod:`difflib`. Character-level edit similarity.
ngram
    Character tri-gram Jaccard. Robust to spelling variations and partial matches.
combined
    Weighted ensemble of all three, with a bonus when the expected phrase is
    present as an exact substring.
"""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

_WORD_BOUNDARY: re.Pattern[str] = re.compile(r"\b\w+\b")

_TOKEN_OVERLAP_WEIGHT: float = 0.4
_FUZZY_WEIGHT: float = 0.3
_NGRAM_WEIGHT: float = 0.3
_EXACT_SUBSTRING_BONUS: float = 0.2
# Minimum score guaranteed when the expected phrase is an exact substring of result.
# Preserves backward-compatible behaviour: exact keyword presence always passes the
# default threshold of 0.8, matching the prior binary scorer's semantics.
_EXACT_SUBSTRING_FLOOR: float = 0.85
_DEFAULT_NGRAM_SIZE: int = 3


class SimilarityScorer:
    """Multi-strategy text similarity scoring.

    Each strategy returns a float in [0.0, 1.0] where 1.0 indicates
    identical text and 0.0 indicates no detectable similarity.
    """

    def score(
        self,
        result: str,
        expected: str,
        strategy: str = "combined",
    ) -> float:
        """Score similarity between *result* and *expected*.

        Parameters
        ----------
        result:
            The agent output string.
        expected:
            The reference string (intent keyword or phrase).
        strategy:
            One of ``"token_overlap"``, ``"fuzzy"``, ``"ngram"``,
            or ``"combined"``. Default ``"combined"``.

        Returns
        -------
        float
            Similarity in [0.0, 1.0].

        Raises
        ------
        ValueError
            If *strategy* is not one of the recognised strategy names.
        """
        if not result or not expected:
            return 0.0

        if strategy == "token_overlap":
            return self._token_overlap(result, expected)
        elif strategy == "fuzzy":
            return self._fuzzy_match(result, expected)
        elif strategy == "ngram":
            return self._ngram_similarity(result, expected)
        elif strategy == "combined":
            return self._combined(result, expected)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                "Valid options: 'token_overlap', 'fuzzy', 'ngram', 'combined'."
            )

    def _tokenize(self, text: str) -> set[str]:
        """Return the set of lowercase word tokens from *text*."""
        return set(_WORD_BOUNDARY.findall(text.lower()))

    def _token_overlap(self, result: str, expected: str) -> float:
        """Jaccard index of word token sets."""
        tokens_a = self._tokenize(result)
        tokens_b = self._tokenize(expected)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _fuzzy_match(self, result: str, expected: str) -> float:
        """SequenceMatcher ratio (character-level edit similarity)."""
        return SequenceMatcher(None, result.lower(), expected.lower()).ratio()

    def _ngram_similarity(
        self,
        result: str,
        expected: str,
        n: int = _DEFAULT_NGRAM_SIZE,
    ) -> float:
        """Character n-gram Jaccard similarity."""

        def get_ngrams(text: str, size: int) -> Counter[str]:
            normalized = text.lower()
            return Counter(
                normalized[i : i + size] for i in range(len(normalized) - size + 1)
            )

        ngrams_a = get_ngrams(result, n)
        ngrams_b = get_ngrams(expected, n)
        if not ngrams_a or not ngrams_b:
            return 0.0

        intersection = sum((ngrams_a & ngrams_b).values())
        union = sum((ngrams_a | ngrams_b).values())
        return intersection / union if union > 0 else 0.0

    def _combined(self, result: str, expected: str) -> float:
        """Weighted ensemble of token_overlap, fuzzy, and ngram strategies.

        When the expected phrase is found as an exact case-insensitive substring
        of result, the score is guaranteed to be at least ``_EXACT_SUBSTRING_FLOOR``
        (0.85). This preserves backward-compatible semantics: a keyword that is
        literally present always passes the default accuracy threshold (0.8).
        An additional bonus of +0.2 is also added before capping at 1.0, so
        higher-quality matches (where the sentences are very similar) can still
        score above the floor.
        """
        token_score = self._token_overlap(result, expected)
        fuzzy_score = self._fuzzy_match(result, expected)
        ngram_score = self._ngram_similarity(result, expected)

        combined = (
            token_score * _TOKEN_OVERLAP_WEIGHT
            + fuzzy_score * _FUZZY_WEIGHT
            + ngram_score * _NGRAM_WEIGHT
        )

        if expected.lower() in result.lower():
            combined = max(_EXACT_SUBSTRING_FLOOR, min(1.0, combined + _EXACT_SUBSTRING_BONUS))

        return round(combined, 4)
