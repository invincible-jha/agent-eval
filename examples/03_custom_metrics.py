#!/usr/bin/env python3
"""Example: Custom Metrics

Demonstrates how to define a custom Evaluator with domain-specific
scoring logic, then integrate it into the runner pipeline.

Usage:
    python examples/03_custom_metrics.py

Requirements:
    pip install agent-eval
"""
from __future__ import annotations

from agent_eval import (
    Dimension,
    DimensionScore,
    EvalResult,
    Evaluator,
    TestCase,
    AgentUnderTest,
)


class KeywordRecallEvaluator(Evaluator):
    """Scores how many required keywords appear in the agent response."""

    def __init__(self, required_keywords: list[str]) -> None:
        self._keywords = [kw.lower() for kw in required_keywords]

    @property
    def dimension(self) -> Dimension:
        return Dimension.ACCURACY

    def score(self, case: TestCase, response: str) -> DimensionScore:
        response_lower = response.lower()
        hits = sum(1 for kw in self._keywords if kw in response_lower)
        ratio = hits / len(self._keywords) if self._keywords else 1.0
        return DimensionScore(
            dimension=self.dimension,
            score=ratio,
            explanation=f"Found {hits}/{len(self._keywords)} required keywords",
        )


class LengthEvaluator(Evaluator):
    """Scores responses that fall within an acceptable length range."""

    def __init__(self, min_words: int = 10, max_words: int = 100) -> None:
        self._min = min_words
        self._max = max_words

    @property
    def dimension(self) -> Dimension:
        return Dimension.COHERENCE

    def score(self, case: TestCase, response: str) -> DimensionScore:
        word_count = len(response.split())
        if self._min <= word_count <= self._max:
            score = 1.0
            explanation = f"Length OK ({word_count} words)"
        elif word_count < self._min:
            score = word_count / self._min
            explanation = f"Too short ({word_count} words, min {self._min})"
        else:
            score = self._max / word_count
            explanation = f"Too long ({word_count} words, max {self._max})"
        return DimensionScore(
            dimension=self.dimension,
            score=score,
            explanation=explanation,
        )


def main() -> None:
    # Step 1: Define custom evaluators
    keyword_evaluator = KeywordRecallEvaluator(
        required_keywords=["neural", "network", "layers", "training"]
    )
    length_evaluator = LengthEvaluator(min_words=5, max_words=50)

    # Step 2: Create a test case
    case = TestCase(
        id="custom-01",
        input="Describe a neural network in one sentence.",
        expected_output="",
        dimensions=[Dimension.ACCURACY, Dimension.COHERENCE],
    )

    # Step 3: Simulate agent response
    response = "A neural network consists of layers of interconnected nodes that learn by adjusting weights during training."

    # Step 4: Score with custom evaluators
    keyword_score = keyword_evaluator.score(case, response)
    length_score = length_evaluator.score(case, response)

    print("Custom metric results:")
    print(f"  Keyword recall: {keyword_score.score:.2f} — {keyword_score.explanation}")
    print(f"  Length quality: {length_score.score:.2f} — {length_score.explanation}")

    # Step 5: Build EvalResult manually
    result = EvalResult(
        case_id=case.id,
        input=case.input,
        response=response,
        scores=[keyword_score, length_score],
    )
    overall = sum(s.score for s in result.scores) / len(result.scores)
    print(f"\nOverall average score: {overall:.2f}")
    print(f"Case: {result.case_id} | Response preview: {result.response[:60]}...")


if __name__ == "__main__":
    main()
