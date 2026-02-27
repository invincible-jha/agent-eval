"""Load production traces from JSON Lines format.

Each line of a .jsonl file represents one agent interaction trace.
Required fields:
  - ``input``: the user/system input string
  - ``output``: the agent's response string

Optional fields:
  - ``trace_id``: unique identifier
  - ``timestamp``: ISO-8601 timestamp
  - ``tool_calls``: list of tool call records
  - ``latency_ms``: response latency in milliseconds
  - ``metadata``: arbitrary key-value metadata
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProductionTrace:
    """A single production agent interaction trace.

    Attributes
    ----------
    trace_id:
        Unique identifier for this trace.
    input_text:
        The input prompt or query sent to the agent.
    output_text:
        The agent's response.
    tool_calls:
        List of tool call records, if any.
    latency_ms:
        Response latency in milliseconds, if recorded.
    timestamp:
        ISO-8601 timestamp of the interaction.
    metadata:
        Arbitrary key-value pairs attached to this trace.
    """

    trace_id: str
    input_text: str
    output_text: str
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    latency_ms: float | None = None
    timestamp: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def input_length(self) -> int:
        """Character length of the input text."""
        return len(self.input_text)

    @property
    def output_length(self) -> int:
        """Character length of the output text."""
        return len(self.output_text)

    @property
    def tool_call_count(self) -> int:
        """Number of tool calls made during this trace."""
        return len(self.tool_calls)


def _parse_trace_record(
    record: dict[str, object],
    index: int,
) -> ProductionTrace | None:
    """Parse a single JSONL record into a ProductionTrace.

    Parameters
    ----------
    record:
        Parsed JSON dict from a single JSONL line.
    index:
        Line index, used for error messages and auto-generated IDs.

    Returns
    -------
    ProductionTrace | None
        None if required fields are missing.
    """
    input_text = record.get("input") or record.get("input_text", "")
    output_text = record.get("output") or record.get("output_text", "")

    if not isinstance(input_text, str) or not isinstance(output_text, str):
        logger.warning(
            "Skipping trace at line %d: 'input' and 'output' must be strings.",
            index,
        )
        return None

    trace_id = str(
        record.get("trace_id") or record.get("id") or f"trace_{index}"
    )

    tool_calls_raw = record.get("tool_calls", [])
    tool_calls = tool_calls_raw if isinstance(tool_calls_raw, list) else []

    latency_raw = record.get("latency_ms")
    latency_ms: float | None = None
    if latency_raw is not None:
        try:
            latency_ms = float(latency_raw)
        except (ValueError, TypeError):
            logger.debug("Could not parse latency_ms at line %d", index)

    metadata_raw = record.get("metadata", {})
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    return ProductionTrace(
        trace_id=trace_id,
        input_text=input_text,
        output_text=output_text,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        timestamp=str(record.get("timestamp", "")),
        metadata=metadata,
    )


def load_traces_from_jsonl(
    content: str,
    *,
    max_traces: int | None = None,
) -> list[ProductionTrace]:
    """Parse JSON Lines content and return a list of ProductionTrace objects.

    Parameters
    ----------
    content:
        Raw string content of a .jsonl file.
    max_traces:
        Optional limit on number of traces returned.

    Returns
    -------
    list[ProductionTrace]
        Parsed trace records, skipping invalid lines.
    """
    traces: list[ProductionTrace] = []

    for index, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON at line %d: %s", index, exc)
            continue

        if not isinstance(record, dict):
            logger.warning("Expected dict at line %d, got %s", index, type(record).__name__)
            continue

        trace = _parse_trace_record(record, index)
        if trace is not None:
            traces.append(trace)

        if max_traces is not None and len(traces) >= max_traces:
            break

    logger.info("Loaded %d production traces from JSONL content", len(traces))
    return traces


class TraceLoader:
    """High-level loader for production traces from .jsonl files.

    Usage
    -----
    ::

        loader = TraceLoader()
        traces = loader.load_file(Path("production_traces.jsonl"))
    """

    def __init__(self, max_traces: int | None = None) -> None:
        """Initialise the loader.

        Parameters
        ----------
        max_traces:
            Optional maximum number of traces to load.
        """
        self.max_traces = max_traces

    def load_file(self, path: Path) -> list[ProductionTrace]:
        """Load traces from a .jsonl file on disk.

        Parameters
        ----------
        path:
            Filesystem path to the JSONL file.

        Returns
        -------
        list[ProductionTrace]

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        content = path.read_text(encoding="utf-8")
        return load_traces_from_jsonl(content, max_traces=self.max_traces)

    def load_string(self, content: str) -> list[ProductionTrace]:
        """Load traces from a JSONL string.

        Parameters
        ----------
        content:
            Raw JSONL content string.

        Returns
        -------
        list[ProductionTrace]
        """
        return load_traces_from_jsonl(content, max_traces=self.max_traces)
