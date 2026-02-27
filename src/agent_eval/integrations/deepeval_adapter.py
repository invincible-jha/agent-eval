"""DeepEval adapter — imports DeepEval test cases and adds pass@k + statistics.

Converts DeepEval ``LLMTestCase`` objects to agent-eval ``EvalCase`` dicts,
then runs multi-sample statistical evaluation using the existing statistical
metrics module. Bridges DeepEval's dataset format into agent-eval's framework.

Install the extra to use this module::

    pip install aumos-agent-eval[deepeval]

Usage
-----
::

    from agent_eval.integrations.deepeval_adapter import DeepEvalImporter

    importer = DeepEvalImporter()
    cases = importer.import_test_cases(deepeval_dataset)

    def my_agent(input_text: str) -> str:
        ...

    result = importer.run_with_statistics(cases, my_agent, n_runs=5, k=3)
    print(f"pass@3 = {result.pass_at_k_value:.3f}")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent_eval.statistical.metrics import (
    ConfidenceInterval,
    PassAtKResult,
    confidence_interval,
    pass_at_k_result,
    score_stddev,
    score_variance,
)

try:
    import deepeval  # type: ignore[import-untyped]
    from deepeval.test_case import LLMTestCase  # type: ignore[import-untyped]
except ImportError as _import_error:
    raise ImportError(
        "DeepEval is required for this adapter. "
        "Install it with: pip install aumos-agent-eval[deepeval]"
    ) from _import_error

logger = logging.getLogger(__name__)

# Type alias — agent callable signature
AgentCallable = Callable[[str], str]


@dataclass
class EvalCase:
    """An agent-eval test case imported from a DeepEval dataset.

    Parameters
    ----------
    case_id:
        Unique identifier for this test case.
    input_text:
        The input prompt or question for the agent.
    expected_output:
        The reference answer or expected agent response. None when
        the DeepEval case has no expected output.
    context:
        Context strings from the DeepEval case (e.g. RAG documents).
    metadata:
        Additional key-value data from the DeepEval case.
    """

    case_id: str
    input_text: str
    expected_output: Optional[str]
    context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a single agent run against one EvalCase.

    Parameters
    ----------
    case_id:
        The test case identifier.
    run_index:
        Zero-based run index within the n_runs repetitions.
    agent_output:
        Raw string returned by the agent.
    passed:
        Whether this run's output matches the expected output.
    latency_ms:
        Wall-clock time from request to response, in milliseconds.
    error:
        Exception string if the agent raised; None on success.
    """

    case_id: str
    run_index: int
    agent_output: str
    passed: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class StatisticalResult:
    """Aggregate statistical result across all runs and cases.

    Parameters
    ----------
    cases:
        The EvalCase list that was evaluated.
    n_runs:
        Number of runs per case.
    k:
        The k value used for pass@k.
    pass_at_k_value:
        The estimated pass@k probability across all cases.
    pass_at_k_detail:
        Full PassAtKResult with per-field breakdown.
    confidence_interval_result:
    Wilson score confidence interval for the overall pass rate.
    variance:
        Score variance across all runs.
    std_dev:
        Score standard deviation across all runs.
    run_results:
        Flat list of all individual RunResult records.
    total_runs:
        Total number of agent invocations executed.
    successful_runs:
        Number of runs that passed.
    mean_latency_ms:
        Average latency per run in milliseconds.
    """

    cases: list[EvalCase]
    n_runs: int
    k: int
    pass_at_k_value: float
    pass_at_k_detail: PassAtKResult
    confidence_interval_result: ConfidenceInterval
    variance: float
    std_dev: float
    run_results: list[RunResult]
    total_runs: int
    successful_runs: int
    mean_latency_ms: float


class DeepEvalImporter:
    """Converts DeepEval test cases to agent-eval format and runs statistical evaluation.

    Parameters
    ----------
    pass_threshold:
        Proportion of matching words required to count a run as passing
        when no explicit checker is provided. Range [0.0, 1.0]. Default: 0.7.
    pass_checker:
        Optional custom function ``(expected, actual) -> bool`` that
        determines whether a run passed. When provided, overrides the
        default word-overlap heuristic.

    Examples
    --------
    ::

        importer = DeepEvalImporter()
        cases = importer.import_test_cases(dataset)
        result = importer.run_with_statistics(cases, agent_fn, n_runs=10, k=5)
        print(f"pass@5 = {result.pass_at_k_value:.3f}")
        print(f"95% CI: [{result.confidence_interval_result.lower:.3f}, "
              f"{result.confidence_interval_result.upper:.3f}]")
    """

    def __init__(
        self,
        pass_threshold: float = 0.7,
        pass_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    ) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError(
                f"pass_threshold must be in [0.0, 1.0], got {pass_threshold}"
            )
        self._pass_threshold = pass_threshold
        self._pass_checker = pass_checker

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_test_cases(
        self,
        deepeval_dataset: Any,
    ) -> list[EvalCase]:
        """Convert DeepEval test cases to agent-eval EvalCase objects.

        Accepts a DeepEval ``EvaluationDataset`` (which exposes
        ``test_cases``) or a plain list of ``LLMTestCase`` objects.

        Parameters
        ----------
        deepeval_dataset:
            A ``deepeval.dataset.EvaluationDataset`` or ``list[LLMTestCase]``.

        Returns
        -------
        list[EvalCase]
            Agent-eval format test cases.

        Raises
        ------
        TypeError
            If the dataset contains objects that are not ``LLMTestCase``
            instances.
        """
        raw_cases: list[Any]
        if isinstance(deepeval_dataset, list):
            raw_cases = deepeval_dataset
        else:
            # EvaluationDataset exposes .test_cases
            raw_cases = list(getattr(deepeval_dataset, "test_cases", deepeval_dataset))

        eval_cases: list[EvalCase] = []
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, LLMTestCase):
                raise TypeError(
                    f"Expected LLMTestCase at index {index}, got {type(raw_case).__name__}"
                )
            eval_cases.append(self._convert_case(raw_case, index))

        logger.info("Imported %d test cases from DeepEval dataset", len(eval_cases))
        return eval_cases

    def import_single_case(
        self,
        case: Any,
        case_id: Optional[str] = None,
    ) -> EvalCase:
        """Convert a single DeepEval LLMTestCase to an EvalCase.

        Parameters
        ----------
        case:
            A ``deepeval.test_case.LLMTestCase`` instance.
        case_id:
            Optional override for the case identifier.

        Returns
        -------
        EvalCase
        """
        if not isinstance(case, LLMTestCase):
            raise TypeError(
                f"Expected LLMTestCase, got {type(case).__name__}"
            )
        result = self._convert_case(case, 0)
        if case_id is not None:
            return EvalCase(
                case_id=case_id,
                input_text=result.input_text,
                expected_output=result.expected_output,
                context=result.context,
                metadata=result.metadata,
            )
        return result

    # ------------------------------------------------------------------
    # Statistical evaluation
    # ------------------------------------------------------------------

    def run_with_statistics(
        self,
        cases: list[EvalCase],
        agent_fn: AgentCallable,
        n_runs: int = 10,
        k: int = 1,
        confidence_level: float = 0.95,
    ) -> StatisticalResult:
        """Run each test case n_runs times and compute pass@k and CI statistics.

        Parameters
        ----------
        cases:
            Test cases from ``import_test_cases()``.
        agent_fn:
            Callable ``(input_text: str) -> str`` representing the agent under
            evaluation.
        n_runs:
            Number of independent agent invocations per case.
        k:
            The k value for pass@k (must be <= n_runs).
        confidence_level:
            Confidence level for the Wilson score CI. Default: 0.95.

        Returns
        -------
        StatisticalResult
            Full statistical analysis including pass@k, CI, variance, and all
            individual run records.

        Raises
        ------
        ValueError
            If ``n_runs < 1`` or ``k < 1`` or ``cases`` is empty.
        """
        if not cases:
            raise ValueError("cases must not be empty")
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        all_run_results: list[RunResult] = []

        for case in cases:
            for run_index in range(n_runs):
                run_result = self._execute_single_run(
                    case=case,
                    agent_fn=agent_fn,
                    run_index=run_index,
                )
                all_run_results.append(run_result)

        total_runs = len(all_run_results)
        successful_runs = sum(1 for r in all_run_results if r.passed and r.error is None)

        pass_at_k_detail = pass_at_k_result(
            n_correct=successful_runs,
            n_total=total_runs,
            k=k,
        )

        ci_result = confidence_interval(
            n_successes=successful_runs,
            n_trials=total_runs,
            confidence=confidence_level,
        )

        scores = [1.0 if r.passed else 0.0 for r in all_run_results]
        var = score_variance(scores)
        std = score_stddev(scores)

        latencies = [r.latency_ms for r in all_run_results]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

        logger.info(
            "Statistical evaluation complete: %d/%d passed, pass@%d=%.4f",
            successful_runs,
            total_runs,
            k,
            pass_at_k_detail.value,
        )

        return StatisticalResult(
            cases=cases,
            n_runs=n_runs,
            k=k,
            pass_at_k_value=pass_at_k_detail.value,
            pass_at_k_detail=pass_at_k_detail,
            confidence_interval_result=ci_result,
            variance=var,
            std_dev=std,
            run_results=all_run_results,
            total_runs=total_runs,
            successful_runs=successful_runs,
            mean_latency_ms=round(mean_latency, 3),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_case(self, raw_case: Any, index: int) -> EvalCase:
        """Extract fields from an LLMTestCase into an EvalCase."""
        case_id = str(getattr(raw_case, "id", None) or f"case-{index:04d}")
        input_text = str(getattr(raw_case, "input", "") or "")
        expected = getattr(raw_case, "expected_output", None)
        expected_output: Optional[str] = str(expected) if expected is not None else None

        # DeepEval context may be a list[str] or None
        raw_context = getattr(raw_case, "context", None) or []
        context = [str(c) for c in raw_context] if raw_context else []

        # Capture additional DeepEval fields as metadata
        metadata: dict[str, Any] = {}
        for attr in ("retrieval_context", "additional_metadata", "tags"):
            value = getattr(raw_case, attr, None)
            if value is not None:
                metadata[attr] = value

        return EvalCase(
            case_id=case_id,
            input_text=input_text,
            expected_output=expected_output,
            context=context,
            metadata=metadata,
        )

    def _execute_single_run(
        self,
        case: EvalCase,
        agent_fn: AgentCallable,
        run_index: int,
    ) -> RunResult:
        """Invoke the agent for one run and evaluate the output."""
        start_time = time.perf_counter()
        agent_output = ""
        error_msg: Optional[str] = None

        try:
            agent_output = agent_fn(case.input_text)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.debug(
                "Agent raised during case %s run %d: %s",
                case.case_id,
                run_index,
                error_msg,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        passed = self._check_pass(case.expected_output, agent_output) if not error_msg else False

        return RunResult(
            case_id=case.case_id,
            run_index=run_index,
            agent_output=agent_output,
            passed=passed,
            latency_ms=round(latency_ms, 3),
            error=error_msg,
        )

    def _check_pass(
        self,
        expected: Optional[str],
        actual: str,
    ) -> bool:
        """Determine whether an agent output passes for a given expected output."""
        if self._pass_checker is not None:
            return self._pass_checker(expected, actual)

        if expected is None:
            # No reference answer — treat any non-empty output as passing
            return bool(actual.strip())

        if not actual.strip():
            return False

        # Default: word-overlap heuristic
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        if not expected_words:
            return bool(actual.strip())
        overlap = len(expected_words & actual_words) / len(expected_words)
        return overlap >= self._pass_threshold

    def __repr__(self) -> str:
        return (
            f"DeepEvalImporter("
            f"pass_threshold={self._pass_threshold!r}, "
            f"custom_checker={self._pass_checker is not None})"
        )
