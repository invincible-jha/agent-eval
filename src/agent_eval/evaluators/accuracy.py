"""Basic accuracy evaluator for agent-eval.

Measures whether an agent output matches the expected answer using
configurable comparison modes: exact match, fuzzy (Jaccard similarity),
substring contains, or regex pattern.

NOTE: This is a commodity accuracy evaluator. It is NOT the Behavioral
Consistency Metric (BCM), which measures consistency across multiple
runs using statistical analysis. BCM is available via the plugin system.
"""
from __future__ import annotations

import re

from agent_eval.core.evaluator import Dimension, DimensionScore, Evaluator
from agent_eval.core.exceptions import EvaluatorError

# Supported comparison modes
EXACT = "exact"
FUZZY = "fuzzy"
CONTAINS = "contains"
REGEX = "regex"

_VALID_MODES = {EXACT, FUZZY, CONTAINS, REGEX}


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two strings as word sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Tokenizes both strings by whitespace and computes set overlap.
    Case-insensitive.

    Parameters
    ----------
    text_a:
        First string.
    text_b:
        Second string.

    Returns
    -------
    float
        Similarity in [0.0, 1.0]. Returns 1.0 when both strings are empty.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a and not tokens_b:
        return 1.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


class BasicAccuracyEvaluator(Evaluator):
    """Evaluates whether agent output matches the expected answer.

    Supports four comparison modes:
    - "exact": case-insensitive string equality after stripping whitespace
    - "fuzzy": Jaccard word-set similarity with a configurable threshold
    - "contains": expected_output is a substring of agent_output
    - "regex": expected_output is treated as a regex pattern

    NOTE: This is NOT the Behavioral Consistency Metric (BCM). BCM measures
    consistency across multiple runs using statistical methods. This evaluator
    only compares a single output against a reference answer.

    Parameters
    ----------
    mode:
        Comparison mode. One of "exact", "fuzzy", "contains", "regex".
    fuzzy_threshold:
        Minimum Jaccard similarity to pass in "fuzzy" mode. Default 0.7.
    pass_threshold:
        Minimum normalized score to count as a pass. Primarily used
        in "fuzzy" mode; exact/contains/regex are binary pass/fail.
    """

    def __init__(
        self,
        mode: str = FUZZY,
        fuzzy_threshold: float = 0.7,
        pass_threshold: float = 0.7,
    ) -> None:
        if mode not in _VALID_MODES:
            raise EvaluatorError(
                self.name,
                f"Invalid mode {mode!r}. Must be one of {sorted(_VALID_MODES)}",
            )
        if not (0.0 <= fuzzy_threshold <= 1.0):
            raise EvaluatorError(
                self.name,
                f"fuzzy_threshold must be in [0.0, 1.0], got {fuzzy_threshold}",
            )
        self.mode = mode
        self.fuzzy_threshold = fuzzy_threshold
        self.pass_threshold = pass_threshold

    @property
    def dimension(self) -> Dimension:
        return Dimension.ACCURACY

    @property
    def name(self) -> str:
        return "BasicAccuracyEvaluator"

    def evaluate(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        """Score accuracy of agent_output against expected_output.

        Parameters
        ----------
        case_id:
            Test case identifier.
        agent_output:
            The agent's raw output string.
        expected_output:
            Reference answer. If None, score is 0.0 with a note.
        metadata:
            Unused by this evaluator; passed through for compatibility.

        Returns
        -------
        DimensionScore
            Score in [0.0, 1.0].
        """
        if expected_output is None:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                reason="No expected_output provided; cannot evaluate accuracy",
            )

        if self.mode == EXACT:
            return self._evaluate_exact(agent_output, expected_output)
        elif self.mode == FUZZY:
            return self._evaluate_fuzzy(agent_output, expected_output)
        elif self.mode == CONTAINS:
            return self._evaluate_contains(agent_output, expected_output)
        elif self.mode == REGEX:
            return self._evaluate_regex(agent_output, expected_output)

        # Should never reach here given validation in __init__
        raise EvaluatorError(self.name, f"Unexpected mode: {self.mode!r}")  # pragma: no cover

    def _evaluate_exact(self, agent_output: str, expected: str) -> DimensionScore:
        """Case-insensitive exact match after stripping whitespace."""
        normalized_output = agent_output.strip().lower()
        normalized_expected = expected.strip().lower()
        passed = normalized_output == normalized_expected
        score = 1.0 if passed else 0.0
        reason = (
            "Exact match" if passed
            else f"Expected {expected!r}, got {agent_output!r}"
        )
        return DimensionScore(
            dimension=self.dimension,
            score=score,
            passed=passed,
            reason=reason,
        )

    def _evaluate_fuzzy(self, agent_output: str, expected: str) -> DimensionScore:
        """Jaccard word-set similarity with threshold."""
        similarity = _jaccard_similarity(agent_output, expected)
        passed = similarity >= self.fuzzy_threshold
        reason = (
            f"Jaccard similarity {similarity:.3f} "
            f"({'>=', '<'}[not passed] threshold {self.fuzzy_threshold}"
        )
        # Fix the reason string using proper conditionals
        operator = ">=" if passed else "<"
        reason = (
            f"Jaccard similarity {similarity:.3f} {operator} threshold {self.fuzzy_threshold}"
        )
        return DimensionScore(
            dimension=self.dimension,
            score=round(similarity, 4),
            passed=passed,
            reason=reason,
        )

    def _evaluate_contains(self, agent_output: str, expected: str) -> DimensionScore:
        """Check if expected is a substring of agent_output."""
        passed = expected.lower() in agent_output.lower()
        score = 1.0 if passed else 0.0
        reason = (
            f"Output contains expected substring {expected!r}" if passed
            else f"Expected substring {expected!r} not found in output"
        )
        return DimensionScore(
            dimension=self.dimension,
            score=score,
            passed=passed,
            reason=reason,
        )

    def _evaluate_regex(self, agent_output: str, expected: str) -> DimensionScore:
        """Check if agent_output matches expected as a regex pattern."""
        try:
            pattern = re.compile(expected, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                reason=f"Invalid regex pattern {expected!r}: {exc}",
            )

        matched = bool(pattern.search(agent_output))
        score = 1.0 if matched else 0.0
        reason = (
            f"Output matches regex pattern {expected!r}" if matched
            else f"Output does not match regex pattern {expected!r}"
        )
        return DimensionScore(
            dimension=self.dimension,
            score=score,
            passed=matched,
            reason=reason,
        )
