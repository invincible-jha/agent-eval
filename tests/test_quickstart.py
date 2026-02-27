"""Test that the 3-line quickstart API works for agent-eval."""
from __future__ import annotations


def test_quickstart_import() -> None:
    from agent_eval import evaluate

    assert callable(evaluate)


def test_quickstart_basic_usage() -> None:
    from agent_eval import evaluate

    def simple_agent(prompt: str) -> str:
        return f"Answer: {prompt}"

    report = evaluate(simple_agent, tasks=["Hello?"], n=1)
    assert report is not None


def test_quickstart_multiple_tasks() -> None:
    from agent_eval import evaluate

    def echo_agent(prompt: str) -> str:
        return prompt

    report = evaluate(echo_agent, tasks=["Task one", "Task two"], n=1)
    assert report is not None


def test_quickstart_report_has_results() -> None:
    from agent_eval import evaluate

    def my_agent(prompt: str) -> str:
        return "response"

    report = evaluate(my_agent, tasks=["test"], n=1)
    assert hasattr(report, "results")
    assert len(report.results) >= 1


def test_quickstart_agent_name() -> None:
    from agent_eval import evaluate

    def named_agent(prompt: str) -> str:
        return "ok"

    report = evaluate(named_agent, tasks=["hi"], n=1, agent_name="MyTestAgent")
    assert report.agent_name == "MyTestAgent"
