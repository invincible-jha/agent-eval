"""AutoGen agent adapter for agent-eval.

Wraps AutoGen agent instances as evaluatable agents.
Requires ``autogen`` to be installed.
"""
from __future__ import annotations


class AutoGenAdapter:
    """Wraps an AutoGen agent for evaluation.

    Parameters
    ----------
    agent:
        An AutoGen AssistantAgent or ConversableAgent instance.
    name:
        Human-readable agent name.

    Raises
    ------
    ImportError
        If AutoGen is not installed.
    """

    def __init__(
        self,
        agent: object,
        name: str = "autogen-agent",
    ) -> None:
        try:
            from autogen import ConversableAgent  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AutoGen is required for AutoGenAdapter. "
                "Install with: pip install autogen-agentchat"
            ) from exc

        self._agent = agent
        self._name = name

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    async def invoke(self, input_text: str) -> str:
        """Invoke the AutoGen agent.

        Parameters
        ----------
        input_text:
            The input prompt.

        Returns
        -------
        str
            The agent's response.
        """
        if hasattr(self._agent, "generate_reply"):
            messages = [{"role": "user", "content": input_text}]
            result = self._agent.generate_reply(messages=messages)  # type: ignore[union-attr]
            return str(result) if result else ""

        raise TypeError(
            f"Agent {type(self._agent).__name__} does not have generate_reply()"
        )
