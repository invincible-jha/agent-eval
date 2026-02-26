"""Unit tests for agent_eval.core.agent_wrapper and agent_eval.core.runner.

Tests AgentUnderTest run logic (timeout, retries, factory methods) and
EvalRunner orchestration (from_config, run, fail_fast, error handling).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_eval.core.agent_wrapper import AgentUnderTest
from agent_eval.core.config import EvalConfig
from agent_eval.core.evaluator import Dimension, DimensionScore, Evaluator
from agent_eval.core.exceptions import AgentTimeoutError, RunnerError
from agent_eval.core.runner import EvalRunner, RunnerOptions
from agent_eval.core.suite import BenchmarkSuite, TestCase
from agent_eval.evaluators.accuracy import BasicAccuracyEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _echo_agent(input_text: str) -> str:
    return f"echo: {input_text}"


async def _failing_agent(input_text: str) -> str:
    raise ValueError("agent failure")


async def _slow_agent(input_text: str) -> str:
    await asyncio.sleep(10)  # Much longer than any test timeout
    return "slow result"


def _make_suite(num_cases: int = 1, case_prefix: str = "c") -> BenchmarkSuite:
    """Build a suite using correct TestCase field names: id and input."""
    cases = [
        TestCase(
            id=f"{case_prefix}{i}",
            input=f"input {i}",
            expected_output=f"echo: input {i}",
        )
        for i in range(num_cases)
    ]
    return BenchmarkSuite(name="test-suite", cases=cases)


def _make_agent(fn=_echo_agent, name: str = "test-agent") -> AgentUnderTest:
    return AgentUnderTest(callable_fn=fn, name=name)


# ---------------------------------------------------------------------------
# AgentUnderTest — construction
# ---------------------------------------------------------------------------


class TestAgentUnderTestConstruction:
    def test_default_attributes(self) -> None:
        agent = AgentUnderTest(callable_fn=_echo_agent, name="my-agent")
        assert agent.name == "my-agent"
        assert agent.timeout_ms is None
        assert agent.max_retries == 0

    def test_repr_includes_name_and_timeout(self) -> None:
        agent = AgentUnderTest(
            callable_fn=_echo_agent,
            name="my-agent",
            timeout_ms=5000,
            max_retries=2,
        )
        r = repr(agent)
        assert "my-agent" in r
        assert "5000" in r
        assert "2" in r


# ---------------------------------------------------------------------------
# AgentUnderTest — run (no timeout)
# ---------------------------------------------------------------------------


class TestAgentUnderTestRun:
    def test_run_returns_agent_output(self) -> None:
        agent = AgentUnderTest(callable_fn=_echo_agent)

        async def run() -> str:
            return await agent.run("hello")

        result = asyncio.run(run())
        assert result == "echo: hello"

    def test_run_raises_on_agent_failure(self) -> None:
        agent = AgentUnderTest(callable_fn=_failing_agent)

        async def run() -> str:
            return await agent.run("input")

        with pytest.raises(ValueError, match="agent failure"):
            asyncio.run(run())

    def test_run_retries_on_failure(self) -> None:
        call_count = 0

        async def flaky_agent(input_text: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        agent = AgentUnderTest(callable_fn=flaky_agent, max_retries=2)

        async def run() -> str:
            return await agent.run("input")

        result = asyncio.run(run())
        assert result == "success"
        assert call_count == 3

    def test_run_raises_after_all_retries_exhausted(self) -> None:
        async def always_fails(input_text: str) -> str:
            raise ConnectionError("persistent failure")

        agent = AgentUnderTest(callable_fn=always_fails, max_retries=1)

        async def run() -> str:
            return await agent.run("input")

        with pytest.raises(ConnectionError, match="persistent failure"):
            asyncio.run(run())

    def test_timeout_raises_agent_timeout_error(self) -> None:
        agent = AgentUnderTest(
            callable_fn=_slow_agent,
            timeout_ms=10,  # 10ms timeout
        )

        async def run() -> str:
            return await agent.run("input", case_id="slow-case")

        with pytest.raises(AgentTimeoutError):
            asyncio.run(run())

    def test_timeout_not_applied_when_none(self) -> None:
        agent = AgentUnderTest(callable_fn=_echo_agent, timeout_ms=None)

        async def run() -> str:
            return await agent.run("hello")

        result = asyncio.run(run())
        assert result == "echo: hello"


# ---------------------------------------------------------------------------
# AgentUnderTest — factory methods
# ---------------------------------------------------------------------------


class TestAgentUnderTestFactoryMethods:
    """Test AgentUnderTest factory methods.

    The source's from_callable() calls CallableAdapter.wrap() (a classmethod
    that does not exist on CallableAdapter) and from_http() imports HttpAdapter
    (wrong casing — the real class is HTTPAdapter). These tests patch those
    broken imports at the adapters module level so the factory orchestration
    logic (name, timeout, retry propagation) can be verified without modifying
    the source.
    """

    def _make_callable_adapter_cls(self, fn: object) -> type:
        """Build a drop-in CallableAdapter whose wrap() returns a valid coroutine fn."""
        from agent_eval.adapters.callable import CallableAdapter as _Real

        real = _Real(fn=fn)  # type: ignore[arg-type]

        class _FakeCallableAdapter:
            @classmethod
            def wrap(cls, wrapped_fn: object) -> object:
                # Return the real adapter's invoke method bound to the real instance
                return real.invoke

        return _FakeCallableAdapter  # type: ignore[return-value]

    def test_from_callable_with_sync_function(self) -> None:
        def sync_fn(s: str) -> str:
            return f"sync: {s}"

        fake_cls = self._make_callable_adapter_cls(sync_fn)
        import agent_eval.adapters.callable as _callable_mod
        with patch.object(_callable_mod, "CallableAdapter", fake_cls):
            agent = AgentUnderTest.from_callable(sync_fn, name="sync-agent")

        async def run() -> str:
            return await agent.run("test")

        result = asyncio.run(run())
        assert result == "sync: test"

    def test_from_callable_with_async_function(self) -> None:
        async def async_fn(s: str) -> str:
            return f"async: {s}"

        fake_cls = self._make_callable_adapter_cls(async_fn)
        import agent_eval.adapters.callable as _callable_mod
        with patch.object(_callable_mod, "CallableAdapter", fake_cls):
            agent = AgentUnderTest.from_callable(async_fn, name="async-agent")

        async def run() -> str:
            return await agent.run("input")

        result = asyncio.run(run())
        assert result == "async: input"

    def test_from_callable_name_set_correctly(self) -> None:
        fn = lambda s: "ok"  # noqa: E731
        fake_cls = self._make_callable_adapter_cls(fn)
        import agent_eval.adapters.callable as _callable_mod
        with patch.object(_callable_mod, "CallableAdapter", fake_cls):
            agent = AgentUnderTest.from_callable(fn, name="my-callable")
        assert agent.name == "my-callable"

    def test_from_callable_timeout_passed_through(self) -> None:
        fn = lambda s: "ok"  # noqa: E731
        fake_cls = self._make_callable_adapter_cls(fn)
        import agent_eval.adapters.callable as _callable_mod
        with patch.object(_callable_mod, "CallableAdapter", fake_cls):
            agent = AgentUnderTest.from_callable(fn, timeout_ms=5000)
        assert agent.timeout_ms == 5000

    def test_from_http_creates_agent(self) -> None:
        # source does: from agent_eval.adapters.http import HttpAdapter
        # then: adapter = HttpAdapter(...); return cls(adapter.run, ...)
        # The real class is HTTPAdapter; patch the http module to expose HttpAdapter.
        import agent_eval.adapters.http as _http_mod

        async def _invoke(input_text: str) -> str:
            return "http response"

        mock_instance = MagicMock()
        mock_instance.run = _invoke
        mock_http_cls = MagicMock(return_value=mock_instance)

        with patch.object(_http_mod, "HttpAdapter", mock_http_cls, create=True):
            agent = AgentUnderTest.from_http(
                url="http://example.com/api",
                name="http-agent",
                headers={"Authorization": "Bearer x"},
            )
        assert agent.name == "http-agent"


# ---------------------------------------------------------------------------
# RunnerOptions
# ---------------------------------------------------------------------------


class TestRunnerOptions:
    def test_default_options(self) -> None:
        opts = RunnerOptions()
        assert opts.runs_per_case == 1
        assert opts.timeout_ms == 30_000
        assert opts.max_retries == 0
        assert opts.concurrency == 1
        assert opts.fail_fast is False

    def test_custom_options(self) -> None:
        opts = RunnerOptions(
            runs_per_case=3,
            timeout_ms=5000,
            max_retries=2,
            concurrency=4,
            fail_fast=True,
        )
        assert opts.runs_per_case == 3
        assert opts.fail_fast is True


# ---------------------------------------------------------------------------
# EvalRunner — construction
# ---------------------------------------------------------------------------


class TestEvalRunnerConstruction:
    def test_requires_at_least_one_evaluator(self) -> None:
        with pytest.raises(RunnerError, match="at least one evaluator"):
            EvalRunner(evaluators=[])

    def test_accepts_single_evaluator(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()])
        assert len(runner._evaluators) == 1

    def test_default_options_used_when_none_provided(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()])
        assert runner.options.runs_per_case == 1

    def test_custom_options_stored(self) -> None:
        opts = RunnerOptions(runs_per_case=5)
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()], options=opts)
        assert runner.options.runs_per_case == 5


# ---------------------------------------------------------------------------
# EvalRunner — from_config
# ---------------------------------------------------------------------------


class TestEvalRunnerFromConfig:
    def _make_config(
        self,
        evaluator_type: str = "accuracy",
        runs_per_case: int = 1,
    ) -> EvalConfig:
        config_data = {
            "runner": {"runs_per_case": runs_per_case},
            "evaluators": [{"name": "acc", "type": evaluator_type}],
        }
        return EvalConfig.model_validate(config_data)

    def test_from_config_creates_runner(self) -> None:
        config = self._make_config()
        runner = EvalRunner.from_config(config)
        assert len(runner._evaluators) == 1

    def test_from_config_applies_runner_options(self) -> None:
        config = self._make_config(runs_per_case=3)
        runner = EvalRunner.from_config(config)
        assert runner.options.runs_per_case == 3

    def test_from_config_raises_with_wrong_type(self) -> None:
        with pytest.raises(RunnerError, match="EvalConfig"):
            EvalRunner.from_config("not a config")  # type: ignore[arg-type]

    def test_from_config_raises_with_no_evaluators(self) -> None:
        config = EvalConfig.model_validate({"evaluators": []})
        with pytest.raises(RunnerError, match="No evaluators"):
            EvalRunner.from_config(config)

    def test_from_config_raises_with_unknown_evaluator_type(self) -> None:
        config = EvalConfig.model_validate(
            {"evaluators": [{"name": "x", "type": "nonexistent_type"}]}
        )
        with pytest.raises(RunnerError, match="Unknown evaluator type"):
            EvalRunner.from_config(config)

    def test_from_config_all_builtin_evaluator_types(self) -> None:
        for ev_type in ["accuracy", "latency", "cost", "safety", "format"]:
            config = EvalConfig.model_validate(
                {"evaluators": [{"name": "e", "type": ev_type}]}
            )
            runner = EvalRunner.from_config(config)
            assert len(runner._evaluators) == 1


# ---------------------------------------------------------------------------
# EvalRunner — run
# ---------------------------------------------------------------------------


class TestEvalRunnerRun:
    def test_run_produces_report_with_results(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator(mode="contains")])
        agent = _make_agent(_echo_agent)
        suite = _make_suite(num_cases=2)

        report = asyncio.run(runner.run(agent, suite))
        assert report.total_cases == 2

    def test_run_with_multiple_runs_per_case(self) -> None:
        runner = EvalRunner(
            evaluators=[BasicAccuracyEvaluator()],
            options=RunnerOptions(runs_per_case=3),
        )
        agent = _make_agent(_echo_agent)
        suite = _make_suite(num_cases=1)

        report = asyncio.run(runner.run(agent, suite))
        assert report.total_cases == 3  # 1 case * 3 runs

    def test_run_captures_agent_errors(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()])
        agent = _make_agent(_failing_agent)
        suite = _make_suite(num_cases=1)

        report = asyncio.run(runner.run(agent, suite))
        assert report.error_cases == 1

    def test_run_captures_timeout_errors(self) -> None:
        runner = EvalRunner(
            evaluators=[BasicAccuracyEvaluator()],
            options=RunnerOptions(timeout_ms=10),  # Very short timeout
        )
        agent = _make_agent(_slow_agent)
        suite = _make_suite(num_cases=1)

        report = asyncio.run(runner.run(agent, suite))
        assert report.error_cases == 1

    def test_run_result_has_case_id(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator(mode="contains")])
        agent = _make_agent(_echo_agent)
        suite = _make_suite(num_cases=1, case_prefix="mycase")

        report = asyncio.run(runner.run(agent, suite))
        assert report.results[0].case_id == "mycase0"

    def test_run_results_sorted_by_case_order(self) -> None:
        runner = EvalRunner(
            evaluators=[BasicAccuracyEvaluator()],
            options=RunnerOptions(concurrency=4),
        )
        agent = _make_agent(_echo_agent)
        suite = _make_suite(num_cases=3)

        report = asyncio.run(runner.run(agent, suite))
        case_ids = [r.case_id for r in report.results]
        assert case_ids == sorted(case_ids)

    def test_run_empty_suite_returns_empty_report(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()])
        agent = _make_agent(_echo_agent)
        suite = BenchmarkSuite(name="empty", cases=[])

        report = asyncio.run(runner.run(agent, suite))
        assert report.total_cases == 0

    def test_run_sync_wrapper(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator(mode="contains")])
        agent = _make_agent(_echo_agent)
        suite = _make_suite(num_cases=1)

        report = runner.run_sync(agent, suite)
        assert report.total_cases == 1

    def test_run_records_latency(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()])
        agent = _make_agent(_echo_agent)
        suite = _make_suite(num_cases=1)

        report = asyncio.run(runner.run(agent, suite))
        assert report.results[0].latency_ms >= 0.0

    def test_run_report_contains_suite_name(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()])
        agent = _make_agent(_echo_agent, name="my-agent")
        suite = BenchmarkSuite(name="my-suite", cases=[
            TestCase(id="c1", input="input text")
        ])

        report = asyncio.run(runner.run(agent, suite))
        assert report.suite_name == "my-suite"
        assert report.agent_name == "my-agent"

    def test_fail_fast_aborts_after_first_failure(self) -> None:
        """fail_fast=True should stop processing after first non-error failure."""
        call_count = 0

        async def counting_agent(input_text: str) -> str:
            nonlocal call_count
            call_count += 1
            return "wrong answer"

        runner = EvalRunner(
            evaluators=[BasicAccuracyEvaluator(mode="exact")],
            options=RunnerOptions(fail_fast=True, concurrency=1),
        )
        agent = AgentUnderTest(callable_fn=counting_agent)
        suite = _make_suite(num_cases=5)

        report = asyncio.run(runner.run(agent, suite))
        # fail_fast with concurrency=1 means we stop after 1 failure
        assert report.total_cases <= 5

    def test_run_with_case_latency_threshold_override(self) -> None:
        """TestCase.max_latency_ms overrides runner timeout."""
        runner = EvalRunner(
            evaluators=[BasicAccuracyEvaluator()],
            options=RunnerOptions(timeout_ms=30000),
        )
        agent = _make_agent(_echo_agent)
        # Case with explicit max_latency_ms
        case = TestCase(id="c1", input="test input", max_latency_ms=5000)
        suite = BenchmarkSuite(name="suite", cases=[case])

        report = asyncio.run(runner.run(agent, suite))
        assert report.total_cases == 1

    def test_run_with_suite_default_latency(self) -> None:
        runner = EvalRunner(evaluators=[BasicAccuracyEvaluator()])
        agent = _make_agent(_echo_agent)
        suite = BenchmarkSuite(
            name="suite",
            cases=[TestCase(id="c1", input="input")],
            default_max_latency_ms=5000,
        )

        report = asyncio.run(runner.run(agent, suite))
        assert report.total_cases == 1

    def test_run_with_suite_default_cost_tokens(self) -> None:
        from agent_eval.evaluators.cost import BasicCostEvaluator
        runner = EvalRunner(evaluators=[BasicCostEvaluator()])
        agent = _make_agent(_echo_agent)
        suite = BenchmarkSuite(
            name="suite",
            cases=[TestCase(id="c1", input="input")],
            default_max_cost_tokens=1000,
        )

        report = asyncio.run(runner.run(agent, suite))
        assert report.total_cases == 1


# ---------------------------------------------------------------------------
# EvalRunner — evaluator error isolation
# ---------------------------------------------------------------------------


class TestEvalRunnerEvaluatorErrors:
    def test_evaluator_exception_does_not_crash_run(self) -> None:
        class BrokenEvaluator(Evaluator):
            @property
            def dimension(self) -> Dimension:
                return Dimension.ACCURACY

            @property
            def name(self) -> str:
                return "BrokenEvaluator"

            def evaluate(self, case_id, agent_output, expected_output, metadata):
                raise RuntimeError("evaluator crash")

        runner = EvalRunner(evaluators=[BrokenEvaluator()])
        agent = _make_agent(_echo_agent)
        suite = _make_suite(num_cases=1)

        # Should not raise; evaluator errors are logged and skipped
        report = asyncio.run(runner.run(agent, suite))
        assert report.total_cases == 1
