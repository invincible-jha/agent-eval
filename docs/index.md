# agent-eval

**Agent Evaluation Framework** — multi-dimensional quality assessment for AI agents.

[![CI](https://github.com/invincible-jha/agent-eval/actions/workflows/ci.yaml/badge.svg)](https://github.com/invincible-jha/agent-eval/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/aumos-agent-eval.svg)](https://pypi.org/project/aumos-agent-eval/)
[![Python versions](https://img.shields.io/pypi/pyversions/aumos-agent-eval.svg)](https://pypi.org/project/aumos-agent-eval/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/invincible-jha/agent-eval/blob/main/LICENSE)

agent-eval is a framework for evaluating AI agents across six quality dimensions: accuracy, latency, cost, safety, format, and custom metrics. It wraps any agent — LangChain, CrewAI, AutoGen, OpenAI Agents, or a plain callable — and produces structured reports with CI quality gates.

## Installation

```bash
pip install aumos-agent-eval
```

Verify the installation:

```bash
agent-eval version
```

## Quick Start

```python
import agent_eval
from agent_eval.runner import EvalRunner
from agent_eval.suite import SuiteBuilder
from agent_eval.gates import ThresholdGate

# Define a benchmark suite programmatically
suite = (
    SuiteBuilder()
    .add_case(
        input="What is the capital of France?",
        expected_output="Paris",
        latency_budget_ms=2000,
        cost_cap_usd=0.01,
    )
    .add_case(
        input="Summarize the French Revolution in one sentence.",
        expected_output=None,  # LLM-judge will evaluate
        latency_budget_ms=5000,
        cost_cap_usd=0.05,
    )
    .build()
)

# Wrap your agent
def my_agent(input_text: str) -> str:
    return "Paris"  # replace with your agent call

# Run evaluation
runner = EvalRunner(concurrency=4, timeout_seconds=30)
report = runner.run(agent=my_agent, suite=suite)

# Apply a quality gate
gate = ThresholdGate(accuracy=0.9, safety=1.0, latency=0.8)
gate.assert_pass(report)  # raises if thresholds not met

# Print the report
print(report.to_markdown())
```

## Key Features

- **EvalRunner** — multi-run evaluation with configurable concurrency, per-case timeouts, retry logic, and fail-fast mode
- **Six evaluation dimensions** — `ACCURACY`, `LATENCY`, `COST`, `SAFETY`, `FORMAT`, and `CUSTOM`, each producing a normalized `[0.0, 1.0]` score with pass/fail determination
- **YAML benchmark suites** — per-case expected outputs, latency budgets, and cost caps; `SuiteBuilder` for programmatic construction
- **LLM-judge evaluator** — alongside deterministic accuracy, latency, cost, and format evaluators, all implementing the `Evaluator` ABC
- **Universal agent adapters** — LangChain, CrewAI, AutoGen, OpenAI Agents, and plain callables supported without code changes
- **Flexible reporting** — JSON, Markdown, HTML, and rich console output with per-dimension aggregate statistics
- **Quality gates** — `ThresholdGate` and `CompositeGate` turn evaluation results into CI pass/fail signals

## Links

- [GitHub Repository](https://github.com/invincible-jha/agent-eval)
- [PyPI Package](https://pypi.org/project/aumos-agent-eval/)
- [Architecture](architecture.md)
- [Migration Guide (from DeepEval)](migrate-from-deepeval.md)
- [Contributing](https://github.com/invincible-jha/agent-eval/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/invincible-jha/agent-eval/blob/main/CHANGELOG.md)

---

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.
