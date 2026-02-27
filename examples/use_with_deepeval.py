"""Example: Using agent-eval with DeepEval.

Install with:
    pip install "aumos-agent-eval[deepeval]"

This example demonstrates importing DeepEval test cases into
agent-eval's statistical evaluation framework. You get pass@k
metrics, confidence intervals, and multi-run statistical analysis
on top of your existing DeepEval test suite.
"""
from __future__ import annotations

# from agent_eval.integrations.deepeval_adapter import (
#     DeepEvalImporter,
#     EvalCase,
#     run_with_statistics,
# )

# --- Import DeepEval test cases ---
# importer = DeepEvalImporter()
#
# # From DeepEval LLMTestCase objects:
# from deepeval.test_case import LLMTestCase
# test_case = LLMTestCase(
#     input="What is the capital of France?",
#     actual_output="The capital of France is Paris.",
#     expected_output="Paris",
# )
# eval_case = importer.from_test_case(test_case)
# print(f"Imported: {eval_case.input_text[:50]}...")

# --- Or create EvalCase directly ---
# case = EvalCase(
#     input_text="What is 2 + 2?",
#     actual_output="4",
#     expected_output="4",
# )

# --- Run with statistical analysis ---
# def my_eval() -> EvalResult:
#     """Your evaluation function."""
#     output = my_agent.call(case.input_text)
#     return evaluator.evaluate(output, expected=case.expected_output)
#
# result = run_with_statistics(my_eval, n_runs=10)
# print(f"Pass rate: {result.pass_rate:.1%}")
# print(f"Pass@3: {result.get_pass_at_k(3).value:.3f}")
# print(f"95% CI: [{result.ci_95.lower:.3f}, {result.ci_95.upper:.3f}]")

print("Example: use_with_deepeval.py")
print("Uncomment the code above and install deepeval to run.")
