"""OpenAI Agents SDK adapter for agent-eval.

Wraps OpenAI Agents SDK agents as evaluatable agents.
Requires ``openai-agents`` to be installed.
"""
from __future__ import annotations


class OpenAIAgentsAdapter:
    """Wraps an OpenAI Agents SDK agent for evaluation.

    Parameters
    ----------
    agent:
        An OpenAI Agents SDK Agent instance.
    name:
        Human-readable agent name.

    Raises
    ------
    ImportError
        If the OpenAI Agents SDK is not installed.
    """

    def __init__(
        self,
        agent: object,
        name: str = "openai-agent",
    ) -> None:
        try:
            from agents import Agent  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAI Agents SDK is required for OpenAIAgentsAdapter. "
                "Install with: pip install openai-agents"
            ) from exc

        self._agent = agent
        self._name = name

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    async def invoke(self, input_text: str) -> str:
        """Run the OpenAI agent.

        Parameters
        ----------
        input_text:
            The input prompt.

        Returns
        -------
        str
            The agent's response.
        """
        try:
            from agents import Runner

            result = await Runner.run(self._agent, input_text)  # type: ignore[arg-type]
            if hasattr(result, "final_output"):
                return str(result.final_output)
            return str(result)
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK Runner not available. "
                "Install with: pip install openai-agents"
            )
