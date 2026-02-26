"""AgentUnderTest — adapts diverse agent implementations to a common interface.

The AgentUnderTest wraps any callable, LangChain agent, CrewAI crew, or
HTTP endpoint behind a single async interface: ``run(input_text) -> str``.

Adapters for specific frameworks live in agent_eval.adapters.* and are
imported lazily to avoid hard dependencies on optional packages.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING

from agent_eval.core.exceptions import AgentTimeoutError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type alias: any async callable that maps a string to a string
AsyncAgentCallable = Callable[[str], Coroutine[None, None, str]]


class AgentUnderTest:
    """Wraps an agent implementation behind a single async interface.

    All framework adapters (LangChain, CrewAI, AutoGen, HTTP) are accessed
    via class methods. The core async run loop lives here so that timeout
    and retry logic is implemented once.

    Parameters
    ----------
    callable_fn:
        An async callable ``(input: str) -> str``.
    name:
        Human-readable name for this agent (used in reports).
    timeout_ms:
        Per-invocation timeout in milliseconds. None means no timeout.
    max_retries:
        Number of additional attempts after the first failure.
    """

    def __init__(
        self,
        callable_fn: AsyncAgentCallable,
        name: str = "agent",
        timeout_ms: int | None = None,
        max_retries: int = 0,
    ) -> None:
        self._callable = callable_fn
        self.name = name
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[str], str] | AsyncAgentCallable,
        name: str = "callable_agent",
        timeout_ms: int | None = None,
        max_retries: int = 0,
    ) -> "AgentUnderTest":
        """Wrap a plain Python callable (sync or async).

        Parameters
        ----------
        fn:
            A function ``(input: str) -> str`` or an async version.
        name:
            Label for reporting.
        timeout_ms:
            Per-invocation timeout in milliseconds.
        max_retries:
            Retry count on failure.

        Returns
        -------
        AgentUnderTest
        """
        from agent_eval.adapters.callable import CallableAdapter
        async_fn = CallableAdapter.wrap(fn)
        return cls(async_fn, name=name, timeout_ms=timeout_ms, max_retries=max_retries)

    @classmethod
    def from_langchain(
        cls,
        agent: object,
        name: str = "langchain_agent",
        timeout_ms: int | None = None,
        max_retries: int = 0,
    ) -> "AgentUnderTest":
        """Wrap a LangChain agent or chain.

        The agent must implement ``ainvoke({"input": str}) -> dict``
        or ``ainvoke(str) -> str``.

        Parameters
        ----------
        agent:
            LangChain agent/chain with ainvoke method.
        name:
            Label for reporting.
        timeout_ms:
            Per-invocation timeout.
        max_retries:
            Retry count.
        """
        from agent_eval.adapters.langchain import LangChainAdapter
        async_fn = LangChainAdapter.wrap(agent)
        return cls(async_fn, name=name, timeout_ms=timeout_ms, max_retries=max_retries)

    @classmethod
    def from_crewai(
        cls,
        crew: object,
        name: str = "crewai_crew",
        timeout_ms: int | None = None,
        max_retries: int = 0,
    ) -> "AgentUnderTest":
        """Wrap a CrewAI Crew object.

        The crew must implement ``kickoff(inputs: dict) -> str``.

        Parameters
        ----------
        crew:
            CrewAI Crew instance.
        name:
            Label for reporting.
        timeout_ms:
            Per-invocation timeout.
        max_retries:
            Retry count.
        """
        from agent_eval.adapters.crewai import CrewAIAdapter
        async_fn = CrewAIAdapter.wrap(crew)
        return cls(async_fn, name=name, timeout_ms=timeout_ms, max_retries=max_retries)

    @classmethod
    def from_http(
        cls,
        url: str,
        name: str = "http_agent",
        timeout_ms: int | None = None,
        max_retries: int = 0,
        headers: dict[str, str] | None = None,
        input_key: str = "input",
        output_key: str = "output",
    ) -> "AgentUnderTest":
        """Wrap an HTTP endpoint that accepts POST requests.

        Parameters
        ----------
        url:
            Full URL of the agent endpoint.
        name:
            Label for reporting.
        timeout_ms:
            Per-invocation timeout.
        max_retries:
            Retry count.
        headers:
            Optional request headers (e.g., Authorization).
        input_key:
            JSON body key for the input text.
        output_key:
            JSON response key containing the output text.
        """
        from agent_eval.adapters.http import HttpAdapter
        adapter = HttpAdapter(
            url=url,
            headers=headers or {},
            input_key=input_key,
            output_key=output_key,
        )
        return cls(adapter.run, name=name, timeout_ms=timeout_ms, max_retries=max_retries)

    # ------------------------------------------------------------------
    # Core run method
    # ------------------------------------------------------------------

    async def run(self, input_text: str, case_id: str = "") -> str:
        """Invoke the agent with timeout and retry logic.

        Parameters
        ----------
        input_text:
            The prompt or task input.
        case_id:
            The test case identifier, used in timeout error messages.

        Returns
        -------
        str
            The agent's string output.

        Raises
        ------
        AgentTimeoutError
            If the agent does not respond within timeout_ms.
        Exception
            Any exception from the agent after all retries are exhausted.
        """
        attempts = self.max_retries + 1
        last_exception: Exception | None = None

        for attempt in range(attempts):
            try:
                return await self._run_once(input_text, case_id)
            except AgentTimeoutError:
                raise
            except Exception as exc:
                last_exception = exc
                if attempt < attempts - 1:
                    logger.warning(
                        "Agent %r failed on attempt %d/%d for case %r: %s",
                        self.name,
                        attempt + 1,
                        attempts,
                        case_id,
                        exc,
                    )

        assert last_exception is not None
        raise last_exception

    async def _run_once(self, input_text: str, case_id: str) -> str:
        """Run the agent once, applying timeout if configured."""
        if self.timeout_ms is None:
            return await self._callable(input_text)

        timeout_seconds = self.timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(
                self._callable(input_text),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise AgentTimeoutError(
                timeout_ms=self.timeout_ms,
                case_id=case_id or "unknown",
            ) from None

    def __repr__(self) -> str:
        return (
            f"AgentUnderTest(name={self.name!r}, "
            f"timeout_ms={self.timeout_ms}, "
            f"max_retries={self.max_retries})"
        )
