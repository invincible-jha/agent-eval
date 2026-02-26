# agent-eval

Framework for evaluating AI agents across multiple quality dimensions

[![CI](https://github.com/aumos-ai/agent-eval/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-eval/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-eval.svg)](https://pypi.org/project/agent-eval/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-eval.svg)](https://pypi.org/project/agent-eval/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- `EvalRunner` orchestrates multi-run evaluation with configurable concurrency, per-case timeouts, retry logic, and fail-fast mode
- Six evaluation dimensions — `ACCURACY`, `LATENCY`, `COST`, `SAFETY`, `FORMAT`, and `CUSTOM` — each producing a normalized `[0.0, 1.0]` score with a pass/fail determination
- `BenchmarkSuite` YAML loader with per-case expected outputs, latency budgets, and cost caps; `SuiteBuilder` for programmatic construction
- LLM-judge evaluator alongside deterministic accuracy, latency, cost, and format evaluators — all implementing the `Evaluator` ABC
- Agent adapters for LangChain, CrewAI, AutoGen, OpenAI Agents, and plain callables so any agent can be wrapped without code changes
- Reporting in JSON, Markdown, HTML, and rich console output with per-dimension aggregate statistics
- Quality gates (`ThresholdGate`, `CompositeGate`) that turn evaluation results into CI pass/fail signals

## Quick Start

Install from PyPI:

```bash
pip install agent-eval
```

Verify the installation:

```bash
agent-eval version
```

Basic usage:

```python
import agent_eval

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

The open-source edition provides the core foundation. For production
deployments requiring SLA-backed support, advanced integrations, and the full
AgentEval platform, see [docs/UPGRADE_TO_AgentEval.md](docs/UPGRADE_TO_AgentEval.md).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
