"""Microsoft Agents SDK agent adapter for agent-eval.

Wraps a Microsoft Agents SDK ``ActivityHandler`` (or compatible bot) as an
evaluatable agent.  Requires ``microsoft-agents`` to be installed.
"""
from __future__ import annotations

from typing import Any


class MicrosoftAgentAdapter:
    """Wraps a Microsoft Agents SDK bot for evaluation.

    The adapter drives a single-turn conversation by constructing a minimal
    ``TurnContext``-like object and invoking the bot's ``on_message_activity``
    or ``on_turn`` handler, then capturing any ``send_activity`` calls as the
    response text.

    Parameters
    ----------
    agent:
        A Microsoft Agents SDK ``ActivityHandler`` or compatible bot instance.
    name:
        Human-readable agent name.

    Raises
    ------
    ImportError
        If the Microsoft Agents SDK is not installed.
    """

    def __init__(
        self,
        agent: object,
        name: str = "microsoft-agent",
    ) -> None:
        try:
            import microsoft.agents  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Microsoft Agents SDK is required for MicrosoftAgentAdapter. "
                "Install with: pip install microsoft-agents"
            ) from exc

        self._agent = agent
        self._name = name

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    async def invoke(self, input_text: str) -> str:
        """Drive a single-turn conversation with the bot.

        Parameters
        ----------
        input_text:
            The user's message text.

        Returns
        -------
        str
            The bot's response text extracted from sent activities.

        Raises
        ------
        TypeError
            If the agent has neither ``on_message_activity`` nor ``on_turn``.
        """
        responses: list[str] = []

        # Build a minimal turn context shim that captures send_activity calls.
        context = _MinimalTurnContext(input_text=input_text, responses=responses)

        if hasattr(self._agent, "on_message_activity"):
            await self._agent.on_message_activity(context)  # type: ignore[union-attr]
        elif hasattr(self._agent, "on_turn"):
            await self._agent.on_turn(context)  # type: ignore[union-attr]
        else:
            raise TypeError(
                f"Microsoft agent {type(self._agent).__name__} has neither "
                "on_message_activity() nor on_turn(). "
                "Pass a valid ActivityHandler instance."
            )

        if responses:
            return " ".join(responses)
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalActivity:
    """Minimal Activity shim used by :class:`_MinimalTurnContext`."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "message"
        self.name: str | None = None


class _MinimalTurnContext:
    """Minimal TurnContext shim that captures ``send_activity`` calls.

    Attributes
    ----------
    activity:
        A minimal Activity object carrying the user's input text.
    """

    def __init__(self, input_text: str, responses: list[str]) -> None:
        self.activity = _MinimalActivity(text=input_text)
        self._responses = responses

    async def send_activity(self, activity: Any) -> None:  # noqa: ANN401
        """Capture the bot's outgoing activity as a text string.

        Parameters
        ----------
        activity:
            A string or object with a ``text`` attribute sent by the bot.
        """
        if isinstance(activity, str):
            self._responses.append(activity)
        else:
            text = getattr(activity, "text", None)
            if isinstance(text, str):
                self._responses.append(text)
            else:
                self._responses.append(str(activity))
