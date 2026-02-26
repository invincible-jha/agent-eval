"""LangChain agent adapter for agent-eval.

Wraps LangChain Agent or Chain instances as evaluatable agents.
Requires ``langchain`` to be installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class LangChainAdapter:
    """Wraps a LangChain agent or chain for evaluation.

    Parameters
    ----------
    agent:
        A LangChain Agent, Chain, or Runnable instance.
    name:
        Human-readable agent name.
    input_key:
        Key to use for the input in the invoke dict. Defaults to "input".
    output_key:
        Key to extract from the output dict. Defaults to "output".

    Raises
    ------
    ImportError
        If LangChain is not installed.
    """

    def __init__(
        self,
        agent: object,
        name: str = "langchain-agent",
        input_key: str = "input",
        output_key: str = "output",
    ) -> None:
        try:
            from langchain_core.runnables import Runnable  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LangChain is required for LangChainAdapter. "
                "Install with: pip install langchain-core"
            ) from exc

        self._agent = agent
        self._name = name
        self._input_key = input_key
        self._output_key = output_key

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    async def invoke(self, input_text: str) -> str:
        """Invoke the LangChain agent.

        Parameters
        ----------
        input_text:
            The input prompt.

        Returns
        -------
        str
            The agent's output.
        """
        input_dict = {self._input_key: input_text}

        if hasattr(self._agent, "ainvoke"):
            result = await self._agent.ainvoke(input_dict)  # type: ignore[union-attr]
        elif hasattr(self._agent, "invoke"):
            result = self._agent.invoke(input_dict)  # type: ignore[union-attr]
        else:
            raise TypeError(
                f"Agent {type(self._agent).__name__} does not have invoke() or ainvoke()"
            )

        if isinstance(result, dict):
            return str(result.get(self._output_key, result))
        return str(result)
