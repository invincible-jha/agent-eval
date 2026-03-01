# Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | [Quickstart](01_quickstart.py) | Minimal working example with the convenience API |
| 02 | [Configuration](02_configuration.py) | Advanced configuration: RunnerOptions, thresholds, gates |
| 03 | [Custom Metrics](03_custom_metrics.py) | Domain-specific evaluators with custom scoring logic |
| 04 | [Baseline Comparison](04_baseline_comparison.py) | Compare two agents against the same suite |
| 05 | [agentcore-sdk Integration](05_integration_agentcore.py) | Evaluate an EventBus-driven agentcore agent |
| 06 | [LangChain Eval](06_langchain_eval.py) | Evaluate a LangChain chain with agent-eval |
| 07 | [CrewAI Eval](07_crewai_eval.py) | Evaluate a CrewAI crew with composite gates |

## Running the examples

```bash
pip install agent-eval
python examples/01_quickstart.py
```

For framework integrations, install the optional dependency:

```bash
pip install langchain langchain-openai   # for example 06
pip install crewai                       # for example 07
pip install agentcore-sdk               # for example 05
```
