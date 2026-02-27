"""Evaluation test cases with known outcomes for agent-eval benchmarks.

Each test case has:
- A question/prompt for the agent
- A ground truth expected answer
- A simulated "correct" agent output (matches expected)
- A simulated "incorrect" agent output (does not match)
- A "partial" agent output (fuzzy match scenario)
- A "human_score" (synthetic human judgement in [0, 1])

The human scores are deterministic (not random) — they encode the degree
to which each case's expected answer would be considered correct by a
simulated expert rater. This allows Spearman correlation computation
between automated evaluators and "human" judgement.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalTestCase:
    """A single evaluation test case with known outcomes.

    Parameters
    ----------
    case_id:
        Unique identifier.
    question:
        The input prompt for the agent.
    expected_answer:
        Ground truth expected output.
    correct_output:
        Output that exactly matches expected (high human score).
    partial_output:
        Output that partially matches (medium human score).
    incorrect_output:
        Output that does not match (low human score).
    human_score_correct:
        Synthetic human score for correct_output in [0, 1].
    human_score_partial:
        Synthetic human score for partial_output in [0, 1].
    human_score_incorrect:
        Synthetic human score for incorrect_output in [0, 1].
    category:
        Task category for stratified analysis.
    """

    case_id: str
    question: str
    expected_answer: str
    correct_output: str
    partial_output: str
    incorrect_output: str
    human_score_correct: float
    human_score_partial: float
    human_score_incorrect: float
    category: str


EVAL_CASES: list[EvalTestCase] = [
    EvalTestCase(
        case_id="qa_001",
        question="What is the capital of France?",
        expected_answer="Paris",
        correct_output="The capital of France is Paris.",
        partial_output="France's main city is Paris, which serves as its capital.",
        incorrect_output="The capital of France is Lyon.",
        human_score_correct=0.95,
        human_score_partial=0.85,
        human_score_incorrect=0.05,
        category="factual_qa",
    ),
    EvalTestCase(
        case_id="qa_002",
        question="What is 15 multiplied by 7?",
        expected_answer="105",
        correct_output="15 multiplied by 7 is 105.",
        partial_output="The answer is approximately 100.",
        incorrect_output="15 multiplied by 7 is 110.",
        human_score_correct=1.00,
        human_score_partial=0.30,
        human_score_incorrect=0.00,
        category="arithmetic",
    ),
    EvalTestCase(
        case_id="qa_003",
        question="Name the author of 'Pride and Prejudice'.",
        expected_answer="Jane Austen",
        correct_output="Jane Austen wrote Pride and Prejudice.",
        partial_output="Pride and Prejudice was written by an English author named Austen.",
        incorrect_output="Charlotte Bronte wrote Pride and Prejudice.",
        human_score_correct=1.00,
        human_score_partial=0.80,
        human_score_incorrect=0.00,
        category="factual_qa",
    ),
    EvalTestCase(
        case_id="qa_004",
        question="What is the boiling point of water in Celsius?",
        expected_answer="100 degrees Celsius",
        correct_output="Water boils at 100 degrees Celsius at sea level.",
        partial_output="Water boils at around 100C under standard conditions.",
        incorrect_output="Water boils at 90 degrees Celsius.",
        human_score_correct=0.95,
        human_score_partial=0.90,
        human_score_incorrect=0.00,
        category="factual_qa",
    ),
    EvalTestCase(
        case_id="code_001",
        question="Write a Python function to reverse a string.",
        expected_answer="def reverse_string(s: str) -> str:\n    return s[::-1]",
        correct_output="def reverse_string(s: str) -> str:\n    return s[::-1]",
        partial_output=(
            "def reverse_string(s):\n"
            "    result = ''\n"
            "    for char in s:\n"
            "        result = char + result\n"
            "    return result"
        ),
        incorrect_output="def reverse_string(s: str) -> str:\n    return s",
        human_score_correct=1.00,
        human_score_partial=0.75,
        human_score_incorrect=0.10,
        category="code_generation",
    ),
    EvalTestCase(
        case_id="code_002",
        question="What does the Python `len()` function return for an empty list?",
        expected_answer="0",
        correct_output="len([]) returns 0.",
        partial_output="An empty list has a length of zero.",
        incorrect_output="len([]) returns None.",
        human_score_correct=1.00,
        human_score_partial=0.85,
        human_score_incorrect=0.00,
        category="code_qa",
    ),
    EvalTestCase(
        case_id="summary_001",
        question=(
            "Summarise in one sentence: "
            "'The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the alphabet.'"
        ),
        expected_answer=(
            "A pangram sentence demonstrating all 26 letters of the alphabet."
        ),
        correct_output=(
            "The passage describes a pangram — a sentence that uses every letter of the alphabet."
        ),
        partial_output="It's a sentence about a fox.",
        incorrect_output=(
            "The passage is about the importance of physical exercise for animals."
        ),
        human_score_correct=0.90,
        human_score_partial=0.40,
        human_score_incorrect=0.05,
        category="summarisation",
    ),
    EvalTestCase(
        case_id="reasoning_001",
        question=(
            "Alice has 3 apples. Bob gives her 2 more. "
            "She then gives half to Carol. How many apples does Alice have?"
        ),
        expected_answer="2.5",
        correct_output="Alice has 2.5 apples.",
        partial_output="Alice ends up with about 2 or 3 apples.",
        incorrect_output="Alice has 5 apples.",
        human_score_correct=1.00,
        human_score_partial=0.50,
        human_score_incorrect=0.00,
        category="arithmetic_reasoning",
    ),
    EvalTestCase(
        case_id="reasoning_002",
        question=(
            "If all cats are animals, and some animals are black, "
            "can we conclude that some cats are black?"
        ),
        expected_answer="No, not necessarily.",
        correct_output=(
            "No. The premises do not allow us to conclude that some cats are black. "
            "The black animals could be non-cat animals."
        ),
        partial_output="Maybe, it depends on which animals are black.",
        incorrect_output="Yes, some cats must be black.",
        human_score_correct=0.95,
        human_score_partial=0.60,
        human_score_incorrect=0.05,
        category="logical_reasoning",
    ),
    EvalTestCase(
        case_id="multi_step_001",
        question=(
            "A store sells apples for $1.50 each and oranges for $2.00 each. "
            "If I buy 3 apples and 2 oranges, what is my total cost?"
        ),
        expected_answer="$8.50",
        correct_output=(
            "3 apples cost $4.50 and 2 oranges cost $4.00, for a total of $8.50."
        ),
        partial_output="The total is around $8 to $9.",
        incorrect_output="The total cost is $10.50.",
        human_score_correct=1.00,
        human_score_partial=0.55,
        human_score_incorrect=0.00,
        category="multi_step_arithmetic",
    ),
]


def get_all_cases() -> list[EvalTestCase]:
    """Return all evaluation test cases."""
    return list(EVAL_CASES)


def get_cases_by_category(category: str) -> list[EvalTestCase]:
    """Return test cases filtered by category.

    Parameters
    ----------
    category:
        Category string to filter on.

    Returns
    -------
    list of EvalTestCase
    """
    return [c for c in EVAL_CASES if c.category == category]


__all__ = ["EvalTestCase", "EVAL_CASES", "get_all_cases", "get_cases_by_category"]
