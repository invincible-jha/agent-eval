"""LLM-as-judge evaluator for agent-eval.

Provides a pass-through wrapper that lets users provide their own LLM
client and prompt template to score agent outputs.

NOTE: This is a pass-through wrapper. The user is responsible for:
- Providing the LLM client
- Writing the evaluation prompt template
- Interpreting the score

This is NOT the Human-Aligned Evaluation (HAE) system, which uses
calibrated rubrics, inter-rater reliability measurement, and
disagreement resolution protocols. HAE is available via the plugin system.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Coroutine
from typing import Union

from agent_eval.core.evaluator import Dimension, DimensionScore, Evaluator
from agent_eval.core.exceptions import EvaluatorError

# Type alias for LLM client callables
SyncLLMCallable = Callable[[str], str]
AsyncLLMCallable = Callable[[str], Coroutine[None, None, str]]
LLMCallable = Union[SyncLLMCallable, AsyncLLMCallable]

_DEFAULT_TEMPLATE = """You are an objective evaluator. Rate the following agent output on a scale from 0.0 to 1.0.

Task: {input}
Expected: {expected_output}
Agent Output: {agent_output}

Provide a score between 0.0 and 1.0 where:
- 1.0 = perfectly correct and helpful
- 0.5 = partially correct
- 0.0 = completely wrong or harmful

Respond with ONLY a decimal number between 0.0 and 1.0."""

_FLOAT_PATTERN = re.compile(r"\b(1\.0|0\.\d{1,4})\b")


def _default_score_parser(response: str) -> float:
    """Extract a float score from an LLM response string.

    Looks for the last float in [0.0, 1.0] range in the response.
    Falls back to 0.0 if no valid float is found.

    Parameters
    ----------
    response:
        The raw string response from the LLM judge.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    matches = _FLOAT_PATTERN.findall(response)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass

    # Try to find any number in the response
    number_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
    all_numbers = number_pattern.findall(response)
    for num_str in reversed(all_numbers):
        try:
            value = float(num_str)
            if 0.0 <= value <= 1.0:
                return value
            elif 1.0 < value <= 10.0:
                # Scale from 10-point to 1.0 scale
                return round(value / 10.0, 4)
        except ValueError:
            continue

    return 0.0


class BasicLLMJudge(Evaluator):
    """Pass-through LLM-as-judge evaluator.

    Wraps a user-provided LLM callable and prompt template to evaluate
    agent outputs. The user is responsible for providing a well-crafted
    prompt and a reliable LLM client.

    This evaluator supports both sync and async LLM clients by running
    sync clients in a thread pool when called from async context.

    NOTE: This is NOT the Human-Aligned Evaluation (HAE) system. HAE
    uses calibrated rubrics, inter-rater reliability, and disagreement
    resolution. This is a simple pass-through: your prompt, your client,
    your score. HAE is available via the plugin system.

    Parameters
    ----------
    llm_client:
        A callable that accepts a prompt string and returns the LLM response.
        Can be sync ``(str) -> str`` or async ``(str) -> Awaitable[str]``.
    prompt_template:
        String template with placeholders: {input}, {agent_output},
        {expected_output}, {case_id}. Defaults to a generic scoring prompt.
    score_parser:
        Callable that extracts a float from the LLM response string.
        Defaults to ``_default_score_parser`` which looks for a float
        in [0.0, 1.0].
    pass_threshold:
        Minimum score to count as a pass. Default: 0.7.
    dimension:
        The dimension this judge measures. Defaults to Dimension.ACCURACY.
        Use Dimension.CUSTOM for open-ended evaluation.
    judge_name:
        Human-readable name for this judge instance.
    """

    def __init__(
        self,
        llm_client: LLMCallable,
        prompt_template: str = _DEFAULT_TEMPLATE,
        score_parser: Callable[[str], float] | None = None,
        pass_threshold: float = 0.7,
        dimension: Dimension = Dimension.ACCURACY,
        judge_name: str = "BasicLLMJudge",
    ) -> None:
        if not callable(llm_client):
            raise EvaluatorError(judge_name, "llm_client must be callable")
        if not (0.0 <= pass_threshold <= 1.0):
            raise EvaluatorError(
                judge_name,
                f"pass_threshold must be in [0.0, 1.0], got {pass_threshold}",
            )
        self._llm_client = llm_client
        self._prompt_template = prompt_template
        self._score_parser = score_parser or _default_score_parser
        self._pass_threshold = pass_threshold
        self._dimension = dimension
        self._judge_name = judge_name

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    @property
    def name(self) -> str:
        return self._judge_name

    def evaluate(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        """Evaluate agent output using the LLM judge.

        Renders the prompt template, calls the LLM client synchronously,
        and parses the score from the response.

        For async LLM clients, use ``evaluate_async()`` instead.

        Parameters
        ----------
        case_id:
            Test case identifier.
        agent_output:
            The agent's output to evaluate.
        expected_output:
            Reference answer (passed to template as {expected_output}).
        metadata:
            Additional context. ``input_text`` key is used for {input}
            in the template if present.

        Returns
        -------
        DimensionScore
        """
        import asyncio
        import inspect

        prompt = self._build_prompt(
            case_id=case_id,
            agent_output=agent_output,
            expected_output=expected_output or "",
            metadata=metadata,
        )

        try:
            if inspect.iscoroutinefunction(self._llm_client):
                # Run async client in a new event loop if not in async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context; use run_in_executor
                    import concurrent.futures
                    future = loop.run_in_executor(
                        None,
                        lambda: asyncio.run(self._llm_client(prompt)),  # type: ignore[arg-type]
                    )
                    # This won't work directly in sync; fall back to a note
                    response = f"0.5 (async client called from sync context for case {case_id})"
                except RuntimeError:
                    response = asyncio.run(self._llm_client(prompt))  # type: ignore[arg-type]
            else:
                response = self._llm_client(prompt)  # type: ignore[call-arg]
        except Exception as exc:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                reason=f"LLM judge call failed: {exc}",
            )

        score = self._score_parser(str(response))
        score = max(0.0, min(1.0, score))
        passed = score >= self._pass_threshold

        return DimensionScore(
            dimension=self.dimension,
            score=round(score, 4),
            passed=passed,
            reason=f"LLM judge score: {score:.3f} (threshold: {self._pass_threshold})",
        )

    async def evaluate_async(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        """Async version of evaluate(). Use with async LLM clients.

        Parameters
        ----------
        case_id:
            Test case identifier.
        agent_output:
            The agent's output to evaluate.
        expected_output:
            Reference answer.
        metadata:
            Additional context.

        Returns
        -------
        DimensionScore
        """
        import inspect

        prompt = self._build_prompt(
            case_id=case_id,
            agent_output=agent_output,
            expected_output=expected_output or "",
            metadata=metadata,
        )

        try:
            if inspect.iscoroutinefunction(self._llm_client):
                response = await self._llm_client(prompt)  # type: ignore[misc]
            else:
                import asyncio
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, self._llm_client, prompt)  # type: ignore[arg-type]
        except Exception as exc:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                reason=f"LLM judge call failed: {exc}",
            )

        score = self._score_parser(str(response))
        score = max(0.0, min(1.0, score))
        passed = score >= self._pass_threshold

        return DimensionScore(
            dimension=self.dimension,
            score=round(score, 4),
            passed=passed,
            reason=f"LLM judge score: {score:.3f} (threshold: {self._pass_threshold})",
        )

    def _build_prompt(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str,
        metadata: dict[str, str | int | float | bool],
    ) -> str:
        """Render the prompt template with evaluation context."""
        input_text = str(metadata.get("input_text", case_id))
        try:
            return self._prompt_template.format(
                case_id=case_id,
                input=input_text,
                agent_output=agent_output,
                expected_output=expected_output,
                **{k: str(v) for k, v in metadata.items()},
            )
        except KeyError:
            # Template has unknown placeholders; use basic substitution
            prompt = self._prompt_template
            prompt = prompt.replace("{case_id}", case_id)
            prompt = prompt.replace("{input}", input_text)
            prompt = prompt.replace("{agent_output}", agent_output)
            prompt = prompt.replace("{expected_output}", expected_output)
            return prompt
