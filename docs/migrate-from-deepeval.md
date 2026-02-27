# Migrating from DeepEval to agent-eval

agent-eval is a statistical evaluation framework built for multi-run, multi-agent benchmarking.
It goes beyond per-prompt pass/fail decisions by measuring evaluation reliability over N runs,
computing pass@k, confidence intervals, and cascade failure analysis across agent pipelines. If
you are using DeepEval today, this guide shows how to migrate or run both frameworks in
coexistence.

---

## Feature Comparison

| Capability | DeepEval | agent-eval |
|---|---|---|
| LLM test metrics (faithfulness, relevancy, etc.) | Yes — first-class | Via custom `Evaluator` implementations |
| Pytest integration | Yes — `deepeval` pytest plugin | Yes — `pytest-agent-eval` plugin |
| Single-run pass/fail | Yes | Yes — `EvalResult.passed` |
| Multi-run statistical evaluation | No | Yes — `StatisticalRunner` with N runs |
| pass@k metric | No | Yes — precomputed for k in {1, 3, 5} |
| Wilson confidence intervals | No | Yes — 95% CI on pass rate |
| Score variance / std dev | No | Yes — `StatisticalResult.score_std` |
| Cascade failure analysis | No | Yes — `CascadeAnalyzer` maps failures through pipeline DAGs |
| Deployment gates | No | Yes — `BasicThresholdGate`, `CompositeGate` |
| Leaderboard / ranking | No | Yes — `EvalLeaderboard` with version tracking |
| BSL behavioral spec integration | No | Yes — run BSL policy assertions as eval suites |
| Evaluation suite builder | No | Yes — `SuiteBuilder`, `SuiteLoader` |
| HTTP agent adapter | No | Yes — `HTTPAdapter` for black-box agent eval |
| Multiple evaluator composition | Limited | Yes — `CompositeGate` chains multiple gates |

---

## Installation

```bash
# Core install
pip install agent-eval

# With pytest plugin
pip install "agent-eval[pytest]"

# With DeepEval adapter for importing existing test cases
pip install "agent-eval[deepeval]"
```

---

## Step 1 — Replace DeepEval Test Cases with agent-eval TestCase

**Before (DeepEval):**

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

def test_rag_agent():
    test_case = LLMTestCase(
        input="What causes climate change?",
        actual_output=agent.run("What causes climate change?"),
        retrieval_context=["Greenhouse gas emissions trap heat in the atmosphere."],
    )
    assert_test(test_case, [AnswerRelevancyMetric(), FaithfulnessMetric()])
```

**After (agent-eval):**

```python
from agent_eval import TestCase, BenchmarkSuite, EvalRunner, RunnerOptions

# Define a test case
test_case = TestCase(
    id="climate-change-rag",
    input_data={"question": "What causes climate change?"},
    expected_output="Greenhouse gas emissions trap heat in the atmosphere.",
    tags=["rag", "factual"],
)

# Wrap your agent
def evaluate_agent(test_case: TestCase) -> bool:
    response = agent.run(test_case.input_data["question"])
    # Your evaluation logic — exact match, LLM-as-judge, etc.
    return "greenhouse" in response.lower()

# Build and run a suite
suite = BenchmarkSuite(name="rag-factual", test_cases=[test_case])
runner = EvalRunner(options=RunnerOptions(timeout_seconds=30))
report = runner.run(suite=suite, eval_fn=evaluate_agent)
print(report.summary())
```

---

## Step 2 — Add Multi-Run Statistical Evaluation

This is the primary advantage of agent-eval over single-run frameworks. Stochastic agents can
pass on lucky runs and fail on others — multi-run statistics reveal actual reliability.

**DeepEval approach (single run, binary):**

```python
# One run — may not represent typical agent behavior
assert_test(test_case, [AnswerRelevancyMetric()])
```

**agent-eval approach (10 runs, statistical):**

```python
from agent_eval import EvalResult, Evaluator, Dimension, DimensionScore
from agent_eval.statistical.runner import StatisticalRunner

# Define your evaluator
class RelevancyEvaluator(Evaluator):
    def evaluate(self, agent_output: str, expected: str) -> EvalResult:
        score = compute_relevancy(agent_output, expected)
        dim_score = DimensionScore(
            dimension=Dimension(name="relevancy", weight=1.0),
            score=score,
        )
        return EvalResult(
            dimension_scores=[dim_score],
            passed=score >= 0.7,
        )

evaluator = RelevancyEvaluator()

# Run 10 times and collect statistics
def eval_fn() -> EvalResult:
    output = agent.run("What causes climate change?")
    return evaluator.evaluate(output, expected="Greenhouse gas emissions...")

runner = StatisticalRunner(n_runs=10)
result = runner.run(eval_fn)

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Mean score: {result.mean_score:.3f} ± {result.score_std:.3f}")
print(f"pass@1: {result.get_pass_at_k(1).value:.3f}")
print(f"pass@3: {result.get_pass_at_k(3).value:.3f}")
print(f"95% CI: [{result.ci_95.lower:.3f}, {result.ci_95.upper:.3f}]")
```

pass@k answers the question: "If I run this agent k times and take the best result, what is the
probability of at least one pass?" This is the right metric for agentic tasks where retry is
acceptable.

---

## Step 3 — Replace DeepEval Pytest Plugin

**Before (DeepEval pytest):**

```python
# test_agent.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

@pytest.mark.parametrize("question,expected", [
    ("What is photosynthesis?", "Plants convert light to energy"),
    ("What is osmosis?", "Movement of water through a membrane"),
])
def test_biology_agent(question, expected):
    test_case = LLMTestCase(
        input=question,
        actual_output=bio_agent.run(question),
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])
```

**After (agent-eval pytest plugin):**

```python
# test_agent.py
import pytest
from agent_eval import TestCase

# The pytest plugin automatically discovers TestCase fixtures and runs them
@pytest.fixture
def biology_suite():
    return [
        TestCase(
            id="photosynthesis",
            input_data={"question": "What is photosynthesis?"},
            expected_output="Plants convert light to energy",
        ),
        TestCase(
            id="osmosis",
            input_data={"question": "What is osmosis?"},
            expected_output="Movement of water through a membrane",
        ),
    ]

def test_biology_agent_suite(biology_suite):
    from agent_eval import EvalRunner, BenchmarkSuite, RunnerOptions

    def eval_fn(tc: TestCase) -> bool:
        output = bio_agent.run(tc.input_data["question"])
        return tc.expected_output.lower() in output.lower()

    suite = BenchmarkSuite(name="biology", test_cases=biology_suite)
    runner = EvalRunner(options=RunnerOptions(n_runs=5))
    report = runner.run(suite=suite, eval_fn=eval_fn)

    assert report.overall_pass_rate >= 0.8, f"Pass rate {report.overall_pass_rate:.1%} below threshold"
```

---

## Step 4 — Add Deployment Gates

agent-eval provides deployment gates that block a release if evaluation scores fall below
configurable thresholds. DeepEval does not have this concept.

```python
from agent_eval import BasicThresholdGate, CompositeGate, GateResult
from agent_eval.statistical.runner import StatisticalRunner, StatisticalResult

# Define pass criteria
accuracy_gate = BasicThresholdGate(
    name="accuracy",
    min_pass_rate=0.85,
    min_mean_score=0.75,
)
latency_gate = BasicThresholdGate(
    name="latency-p95",
    max_latency_ms=2000,
)
gate = CompositeGate(gates=[accuracy_gate, latency_gate], require_all=True)

runner = StatisticalRunner(n_runs=20)
result = runner.run(eval_fn)

gate_result: GateResult = gate.evaluate(result)
if not gate_result.passed:
    print(f"Deployment blocked: {gate_result.reason}")
    raise SystemExit(1)
```

---

## Step 5 — Cascade Failure Analysis

For multi-agent pipelines, a failure in an upstream agent can cascade into failures in downstream
agents. agent-eval's `CascadeAnalyzer` maps these dependencies so you can identify the root cause
rather than treating each failure independently.

```python
from agent_eval.cascade.analyzer import CascadeAnalyzer
from agent_eval.cascade.dependency_graph import DependencyGraph

# Define the pipeline topology
graph = DependencyGraph()
graph.add_edge(upstream="retrieval-agent", downstream="reasoning-agent")
graph.add_edge(upstream="reasoning-agent", downstream="response-agent")

# Collect eval results per agent
results_by_agent = {
    "retrieval-agent": retrieval_result,
    "reasoning-agent": reasoning_result,
    "response-agent": response_result,
}

analyzer = CascadeAnalyzer(graph=graph)
cascade_report = analyzer.analyze(results_by_agent)

for root_cause in cascade_report.root_causes:
    print(f"Root failure: {root_cause.agent_id} — {root_cause.failure_rate:.1%} fail rate")
    print(f"  Cascades into: {[d.agent_id for d in root_cause.downstream_affected]}")
```

---

## Coexistence: Importing DeepEval Test Cases into agent-eval

If you have an existing DeepEval test suite and want to migrate it incrementally, use the
`[deepeval]` adapter to import `LLMTestCase` objects directly.

```bash
pip install "agent-eval[deepeval]"
```

```python
from agent_eval.adapters.deepeval import DeepEvalAdapter
from agent_eval import BenchmarkSuite, EvalRunner, RunnerOptions

# Your existing DeepEval test cases
from deepeval.test_case import LLMTestCase

existing_cases = [
    LLMTestCase(
        input="What is photosynthesis?",
        actual_output="",  # will be populated by the adapter at eval time
        expected_output="Plants convert light to energy",
    ),
]

# Convert to agent-eval TestCase objects
adapter = DeepEvalAdapter()
test_cases = adapter.import_test_cases(existing_cases)

# Run with statistical rigor
suite = BenchmarkSuite(name="migrated-from-deepeval", test_cases=test_cases)
runner = EvalRunner(options=RunnerOptions(n_runs=5))
report = runner.run(suite=suite, eval_fn=lambda tc: your_evaluator(tc))
print(report.summary())
```

You can migrate test cases file-by-file while keeping the rest of your DeepEval suite intact.

---

## What You Gain by Switching

1. **Statistical rigor** — single-run pass/fail is insufficient for stochastic agents. pass@k,
   confidence intervals, and score variance give you a statistically defensible picture of agent
   reliability.
2. **Cascade analysis** — failures in multi-agent pipelines often originate upstream. The
   `CascadeAnalyzer` identifies root causes rather than just reporting that every agent failed.
3. **Deployment gates** — `BasicThresholdGate` and `CompositeGate` block releases automatically
   when evaluation criteria are not met, integrating directly into CI pipelines.
4. **Leaderboard tracking** — `EvalLeaderboard` tracks performance across agent versions over time,
   so you can quantify whether a model or prompt change actually improved quality.
5. **BSL integration** — if you write behavioral policy specifications in BSL, agent-eval can
   execute them as typed evaluation suites without any glue code.

## What You Keep

- Existing DeepEval `LLMTestCase` objects can be imported directly via the adapter, so your test
  content does not need to be rewritten.
- The pytest plugin is additive — you can run `pytest-agent-eval` and DeepEval tests in the same
  CI job while migrating incrementally.
- Any LLM-as-judge evaluators you built for DeepEval can be wrapped as `Evaluator` implementations
  and composed into `CompositeGate` objects.
