"""Tests for agent_eval.gap.trace_loader."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent_eval.gap.trace_loader import (
    ProductionTrace,
    TraceLoader,
    load_traces_from_jsonl,
)


VALID_JSONL = "\n".join([
    json.dumps({"trace_id": "t1", "input": "Hello world", "output": "Hi there"}),
    json.dumps({"input": "Question?", "output": "Answer.", "latency_ms": 120.5}),
    json.dumps({
        "id": "custom_id",
        "input": "Use tool",
        "output": "Done",
        "tool_calls": [{"name": "search", "args": {}}],
        "metadata": {"user_id": "u123"},
    }),
])

EMPTY_JSONL = ""
PARTIAL_JSONL = "\n".join([
    json.dumps({"input": "valid", "output": "response"}),
    "not valid json {{{{",
    json.dumps({"input": "also valid", "output": "also good"}),
])


class TestLoadTracesFromJsonl:
    def test_valid_jsonl_parsed(self) -> None:
        traces = load_traces_from_jsonl(VALID_JSONL)
        assert len(traces) == 3

    def test_trace_ids_set(self) -> None:
        traces = load_traces_from_jsonl(VALID_JSONL)
        assert traces[0].trace_id == "t1"

    def test_auto_id_when_missing(self) -> None:
        traces = load_traces_from_jsonl(VALID_JSONL)
        assert traces[1].trace_id.startswith("trace_")

    def test_latency_ms_parsed(self) -> None:
        traces = load_traces_from_jsonl(VALID_JSONL)
        assert traces[1].latency_ms == 120.5

    def test_tool_calls_parsed(self) -> None:
        traces = load_traces_from_jsonl(VALID_JSONL)
        assert traces[2].tool_call_count == 1

    def test_metadata_parsed(self) -> None:
        traces = load_traces_from_jsonl(VALID_JSONL)
        assert traces[2].metadata.get("user_id") == "u123"

    def test_empty_returns_empty_list(self) -> None:
        traces = load_traces_from_jsonl(EMPTY_JSONL)
        assert traces == []

    def test_invalid_lines_skipped(self) -> None:
        traces = load_traces_from_jsonl(PARTIAL_JSONL)
        assert len(traces) == 2

    def test_max_traces_respected(self) -> None:
        traces = load_traces_from_jsonl(VALID_JSONL, max_traces=2)
        assert len(traces) == 2

    def test_blank_lines_skipped(self) -> None:
        content = json.dumps({"input": "hi", "output": "bye"}) + "\n\n\n"
        traces = load_traces_from_jsonl(content)
        assert len(traces) == 1


class TestProductionTraceProperties:
    def test_input_length(self) -> None:
        trace = ProductionTrace(trace_id="t", input_text="hello", output_text="world")
        assert trace.input_length == 5

    def test_output_length(self) -> None:
        trace = ProductionTrace(trace_id="t", input_text="hi", output_text="hello world")
        assert trace.output_length == 11

    def test_tool_call_count(self) -> None:
        trace = ProductionTrace(
            trace_id="t",
            input_text="q",
            output_text="a",
            tool_calls=[{"name": "t1"}, {"name": "t2"}],
        )
        assert trace.tool_call_count == 2


class TestTraceLoader:
    def test_load_string(self) -> None:
        loader = TraceLoader()
        traces = loader.load_string(VALID_JSONL)
        assert len(traces) == 3

    def test_load_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(VALID_JSONL)
            tmp_path = Path(tmp.name)
        try:
            loader = TraceLoader()
            traces = loader.load_file(tmp_path)
            assert len(traces) == 3
        finally:
            tmp_path.unlink()

    def test_load_file_not_found(self) -> None:
        loader = TraceLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file(Path("/nonexistent/traces.jsonl"))

    def test_max_traces_limit(self) -> None:
        loader = TraceLoader(max_traces=1)
        traces = loader.load_string(VALID_JSONL)
        assert len(traces) == 1
