"""Generic callable adapter for agent-eval.

Wraps any Python callable (sync or async) as an agent for evaluation.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Callable


class CallableAdapter:
    """Wraps a Python callable as an evaluatable agent.

    Parameters
    ----------
    fn:
        The callable to wrap. Can be sync or async. Must accept a
        string input and return a string output.
    name:
        Human-readable name for the agent.
    timeout_seconds:
        Maximum time to wait for the callable. Defaults to 30.

    Examples
    --------
    ::

        def my_agent(prompt: str) -> str:
            return "Hello!"

        adapter = CallableAdapter(fn=my_agent, name="my-agent")
        result = await adapter.invoke("Hi")
    """

    def __init__(
        self,
        fn: Callable[[str], str],
        name: str = "callable-agent",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._fn = fn
        self._name = name
        self._timeout = timeout_seconds
        self._is_async = inspect.iscoroutinefunction(fn)

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    async def invoke(self, input_text: str) -> str:
        """Invoke the wrapped callable with the given input.

        Parameters
        ----------
        input_text:
            The input prompt for the agent.

        Returns
        -------
        str
            The agent's output.

        Raises
        ------
        TimeoutError
            If the callable exceeds the timeout.
        """
        if self._is_async:
            result = await asyncio.wait_for(
                self._fn(input_text),  # type: ignore[arg-type]
                timeout=self._timeout,
            )
        else:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._fn, input_text),
                timeout=self._timeout,
            )
        return str(result)
