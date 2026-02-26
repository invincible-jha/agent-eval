"""EvalRunner — orchestrates the evaluation of an agent against a suite.

The runner coordinates:
1. Loading evaluators from config
2. Running the agent against each test case (with retries and timeout)
3. Scoring outputs with each evaluator
4. Aggregating all results into an EvalReport

NOTE: runs_per_case provides simple reruns to observe variance in
non-deterministic agents. This is NOT statistical rigor, bootstrap
sampling, or confidence interval calculation.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

from agent_eval.core.agent_wrapper import AgentUnderTest
from agent_eval.core.evaluator import EvalResult, Evaluator
from agent_eval.core.exceptions import AgentTimeoutError, RunnerError
from agent_eval.core.report import EvalReport
from agent_eval.core.suite import BenchmarkSuite, TestCase

logger = logging.getLogger(__name__)


@dataclass
class RunnerOptions:
    """Runtime options for EvalRunner.

    Parameters
    ----------
    runs_per_case:
        Number of independent runs per test case. Useful for observing
        variance in non-deterministic agent responses.
        NOT statistical analysis.
    timeout_ms:
        Default per-case timeout. Overridden by TestCase.max_latency_ms
        when present.
    max_retries:
        Retry attempts on transient failures.
    concurrency:
        Maximum parallel agent calls. Use 1 for sequential execution.
    fail_fast:
        Abort after the first case-level failure.
    """

    runs_per_case: int = 1
    timeout_ms: int = 30_000
    max_retries: int = 0
    concurrency: int = 1
    fail_fast: bool = False


class EvalRunner:
    """Runs an agent against a benchmark suite and returns an EvalReport.

    Parameters
    ----------
    evaluators:
        List of Evaluator instances to apply to each agent output.
    options:
        Runner configuration. Defaults to RunnerOptions().

    Example
    -------
    ::

        runner = EvalRunner(
            evaluators=[BasicAccuracyEvaluator(), BasicLatencyEvaluator(max_ms=5000)],
            options=RunnerOptions(runs_per_case=3),
        )
        report = asyncio.run(runner.run(agent, suite))
    """

    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        options: RunnerOptions | None = None,
    ) -> None:
        if not evaluators:
            raise RunnerError("EvalRunner requires at least one evaluator")
        self._evaluators: list[Evaluator] = list(evaluators)
        self.options = options or RunnerOptions()

    @classmethod
    def from_config(cls, config: object) -> "EvalRunner":
        """Build an EvalRunner from an EvalConfig object.

        Parameters
        ----------
        config:
            An EvalConfig instance with evaluator and runner settings.

        Returns
        -------
        EvalRunner

        Raises
        ------
        RunnerError
            If no evaluators are configured or a type is not found.
        """
        from agent_eval.core.config import EvalConfig
        from agent_eval.core.config import RunnerConfig
        from agent_eval.evaluators import EVALUATOR_REGISTRY

        if not isinstance(config, EvalConfig):
            raise RunnerError(f"Expected EvalConfig, got {type(config).__name__}")

        active = config.active_evaluators()
        if not active:
            raise RunnerError(
                "No evaluators configured. Add at least one evaluator to eval.yaml."
            )

        evaluators: list[Evaluator] = []
        for eval_config in active:
            if eval_config.type not in EVALUATOR_REGISTRY:
                raise RunnerError(
                    f"Unknown evaluator type {eval_config.type!r}. "
                    f"Available: {sorted(EVALUATOR_REGISTRY.keys())}"
                )
            evaluator_cls = EVALUATOR_REGISTRY[eval_config.type]
            try:
                evaluators.append(evaluator_cls(**eval_config.settings))
            except TypeError as exc:
                raise RunnerError(
                    f"Cannot instantiate evaluator {eval_config.type!r}: {exc}"
                ) from exc

        runner_cfg: RunnerConfig = config.runner
        options = RunnerOptions(
            runs_per_case=runner_cfg.runs_per_case,
            timeout_ms=runner_cfg.timeout_ms,
            max_retries=runner_cfg.max_retries,
            concurrency=runner_cfg.concurrency,
            fail_fast=runner_cfg.fail_fast,
        )
        return cls(evaluators=evaluators, options=options)

    # ------------------------------------------------------------------
    # Public run interface
    # ------------------------------------------------------------------

    async def run(
        self,
        agent: AgentUnderTest,
        suite: BenchmarkSuite,
    ) -> EvalReport:
        """Run the agent against all cases in the suite.

        Parameters
        ----------
        agent:
            The agent under test.
        suite:
            The benchmark suite of test cases.

        Returns
        -------
        EvalReport
            Aggregated results with per-dimension statistics.
        """
        if not suite.cases:
            logger.warning("BenchmarkSuite %r has no cases", suite.name)

        all_results: list[EvalResult] = []
        semaphore = asyncio.Semaphore(self.options.concurrency)

        async def run_case_bounded(case: TestCase, run_index: int) -> EvalResult:
            async with semaphore:
                return await self._run_single(agent, case, run_index, suite)

        tasks = [
            run_case_bounded(case, run_index)
            for case in suite.cases
            for run_index in range(self.options.runs_per_case)
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            all_results.append(result)

            if self.options.fail_fast and not result.passed and result.error is None:
                logger.warning(
                    "fail_fast=True: aborting after failure on case %r", result.case_id
                )
                break

        # Sort by case order then run_index for deterministic reports
        case_order = {case.id: idx for idx, case in enumerate(suite.cases)}
        all_results.sort(
            key=lambda r: (case_order.get(r.case_id, 999999), r.run_index)
        )

        run_config: dict[str, str | int | float | bool] = {
            "runs_per_case": self.options.runs_per_case,
            "timeout_ms": self.options.timeout_ms,
            "max_retries": self.options.max_retries,
            "concurrency": self.options.concurrency,
            "evaluators": ", ".join(e.name for e in self._evaluators),
        }

        return EvalReport.from_results(
            results=all_results,
            suite_name=suite.name,
            agent_name=agent.name,
            run_config=run_config,
        )

    def run_sync(
        self,
        agent: AgentUnderTest,
        suite: BenchmarkSuite,
    ) -> EvalReport:
        """Synchronous wrapper around ``run()``.

        Parameters
        ----------
        agent:
            The agent under test.
        suite:
            The benchmark suite.

        Returns
        -------
        EvalReport
        """
        return asyncio.run(self.run(agent, suite))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_single(
        self,
        agent: AgentUnderTest,
        case: TestCase,
        run_index: int,
        suite: BenchmarkSuite,
    ) -> EvalResult:
        """Run the agent on a single case and score it with all evaluators."""
        # Determine effective timeout: case-level overrides suite/runner default
        effective_timeout_ms: int
        if case.max_latency_ms is not None:
            effective_timeout_ms = case.max_latency_ms
        elif suite.default_max_latency_ms is not None:
            effective_timeout_ms = suite.default_max_latency_ms
        else:
            effective_timeout_ms = self.options.timeout_ms

        # Build a temporary wrapper with case-specific timeout
        scoped_agent = AgentUnderTest(
            callable_fn=agent._callable,
            name=agent.name,
            timeout_ms=effective_timeout_ms,
            max_retries=self.options.max_retries,
        )

        start = time.perf_counter()
        agent_output = ""
        error: str | None = None

        try:
            agent_output = await scoped_agent.run(case.input, case_id=case.id)
        except AgentTimeoutError as exc:
            error = str(exc)
            logger.warning("Timeout on case %r (run %d): %s", case.id, run_index, exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Agent error on case %r (run %d): %s", case.id, run_index, exc
            )

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Evaluate output with all evaluators
        dimension_scores = []
        if error is None:
            metadata: dict[str, str | int | float | bool] = dict(case.metadata)
            metadata["latency_ms"] = round(latency_ms, 2)
            if case.max_latency_ms is not None:
                metadata["max_latency_ms"] = case.max_latency_ms
            elif suite.default_max_latency_ms is not None:
                metadata["max_latency_ms"] = suite.default_max_latency_ms
            if case.max_cost_tokens is not None:
                metadata["max_cost_tokens"] = case.max_cost_tokens
            elif suite.default_max_cost_tokens is not None:
                metadata["max_cost_tokens"] = suite.default_max_cost_tokens

            for evaluator in self._evaluators:
                try:
                    score = evaluator.evaluate(
                        case_id=case.id,
                        agent_output=agent_output,
                        expected_output=case.expected_output,
                        metadata=metadata,
                    )
                    dimension_scores.append(score)
                except Exception as eval_exc:
                    logger.error(
                        "Evaluator %r failed on case %r: %s",
                        evaluator.name,
                        case.id,
                        eval_exc,
                    )

        return EvalResult(
            case_id=case.id,
            run_index=run_index,
            agent_output=agent_output,
            dimension_scores=dimension_scores,
            latency_ms=latency_ms,
            error=error,
        )
