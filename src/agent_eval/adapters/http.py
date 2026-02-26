"""HTTP endpoint adapter for agent-eval.

Wraps an HTTP API endpoint as an evaluatable agent.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen
from urllib.error import URLError


class HTTPAdapter:
    """Wraps an HTTP endpoint as an evaluatable agent.

    Sends POST requests with JSON body and parses the response.

    Parameters
    ----------
    url:
        The HTTP endpoint URL.
    name:
        Human-readable agent name.
    headers:
        Additional HTTP headers (e.g., Authorization).
    timeout_seconds:
        Request timeout in seconds.
    input_field:
        JSON field name for the input text in the request body.
    output_field:
        JSON field name to extract from the response. Supports
        dot-notation for nested fields (e.g., "data.output").

    Examples
    --------
    ::

        adapter = HTTPAdapter(
            url="https://api.example.com/agent",
            headers={"Authorization": "Bearer token"},
            input_field="prompt",
            output_field="response.text",
        )
        result = await adapter.invoke("What is 2+2?")
    """

    def __init__(
        self,
        url: str,
        name: str = "http-agent",
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
        input_field: str = "input",
        output_field: str = "output",
    ) -> None:
        self._url = url
        self._name = name
        self._headers = headers or {}
        self._timeout = timeout_seconds
        self._input_field = input_field
        self._output_field = output_field

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    def _extract_field(self, data: dict[str, object], field_path: str) -> str:
        """Extract a nested field using dot notation."""
        current: object = data
        for part in field_path.split("."):
            if isinstance(current, dict):
                current = current.get(part, "")
            else:
                return str(current)
        return str(current)

    async def invoke(self, input_text: str) -> str:
        """Send a request to the HTTP endpoint.

        Parameters
        ----------
        input_text:
            The input prompt for the agent.

        Returns
        -------
        str
            The agent's response text.

        Raises
        ------
        ConnectionError
            If the HTTP request fails.
        """
        body = json.dumps({self._input_field: input_text}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }
        req = Request(self._url, data=body, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=self._timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except URLError as exc:
            raise ConnectionError(f"HTTP request failed: {exc}") from exc

        if isinstance(response_data, dict):
            return self._extract_field(response_data, self._output_field)
        return str(response_data)
