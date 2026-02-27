"""Anthropic SDK agent adapter for agent-eval.

Wraps an Anthropic Messages API client as an evaluatable agent.
Requires ``anthropic`` to be installed.
"""
from __future__ import annotations

from typing import Any


class AnthropicAdapter:
    """Wraps an Anthropic Messages API client for evaluation.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic`` client.
    model:
        The model identifier to use for message creation.
    name:
        Human-readable agent name.
    system_prompt:
        Optional system prompt prepended to every conversation.
    max_tokens:
        Maximum tokens in the response.  Defaults to 1024.

    Raises
    ------
    ImportError
        If the Anthropic SDK is not installed.
    """

    def __init__(
        self,
        client: object,
        model: str = "claude-opus-4-6",
        name: str = "anthropic-agent",
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Anthropic SDK is required for AnthropicAdapter. "
                "Install with: pip install anthropic"
            ) from exc

        self._client = client
        self._model = model
        self._name = name
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    async def invoke(self, input_text: str) -> str:
        """Send a message to the Anthropic API and return the response text.

        Parameters
        ----------
        input_text:
            The user message to send.

        Returns
        -------
        str
            The assistant's response text.

        Raises
        ------
        TypeError
            If the client does not expose a ``messages.create`` method.
        """
        messages_api = getattr(self._client, "messages", None)
        if messages_api is None or not hasattr(messages_api, "create"):
            raise TypeError(
                f"Anthropic client {type(self._client).__name__} does not have "
                "messages.create(). Pass a valid anthropic.Anthropic() instance."
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": input_text}],
        }
        if self._system_prompt is not None:
            kwargs["system"] = self._system_prompt

        response = messages_api.create(**kwargs)
        return _extract_response_text(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_response_text(response: object) -> str:
    """Best-effort text extraction from an Anthropic ``Message`` response."""
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return " ".join(parts)
    if hasattr(response, "completion"):
        completion = response.completion  # type: ignore[union-attr]
        if isinstance(completion, str):
            return completion
    return str(response)
