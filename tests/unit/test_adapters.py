"""Unit tests for agent_eval.adapters.

Tests CallableAdapter, HTTPAdapter, and the import-error paths for
AutoGenAdapter, CrewAIAdapter, LangChainAdapter, and OpenAIAgentsAdapter.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from agent_eval.adapters.callable import CallableAdapter
from agent_eval.adapters.http import HTTPAdapter

# Pre-import adapter modules so they are registered in sys.modules.
# This prevents importlib.reload() from failing when a previous test
# used patch.dict to temporarily remove the module from sys.modules.
import agent_eval.adapters.autogen  # noqa: E402
import agent_eval.adapters.crewai   # noqa: E402
import agent_eval.adapters.langchain  # noqa: E402


# ---------------------------------------------------------------------------
# CallableAdapter
# ---------------------------------------------------------------------------


class TestCallableAdapterConstruction:
    def test_sync_function_accepted(self) -> None:
        adapter = CallableAdapter(fn=lambda s: "result", name="test")
        assert adapter.name == "test"
        assert adapter._is_async is False

    def test_async_function_detected(self) -> None:
        async def async_fn(s: str) -> str:
            return "async result"

        adapter = CallableAdapter(fn=async_fn, name="async-agent")  # type: ignore[arg-type]
        assert adapter._is_async is True

    def test_default_name(self) -> None:
        adapter = CallableAdapter(fn=lambda s: "ok")
        assert adapter.name == "callable-agent"

    def test_name_property(self) -> None:
        adapter = CallableAdapter(fn=lambda s: "ok", name="my-agent")
        assert adapter.name == "my-agent"


class TestCallableAdapterInvoke:
    def test_invoke_sync_function(self) -> None:
        adapter = CallableAdapter(fn=lambda s: f"echo: {s}")

        async def run() -> str:
            return await adapter.invoke("hello")

        result = asyncio.run(run())
        assert result == "echo: hello"

    def test_invoke_async_function(self) -> None:
        async def async_fn(s: str) -> str:
            return f"async: {s}"

        adapter = CallableAdapter(fn=async_fn, name="async")  # type: ignore[arg-type]

        async def run() -> str:
            return await adapter.invoke("world")

        result = asyncio.run(run())
        assert result == "async: world"

    def test_invoke_returns_string(self) -> None:
        adapter = CallableAdapter(fn=lambda s: 42)  # type: ignore[arg-type,return-value]

        async def run() -> str:
            return await adapter.invoke("input")

        result = asyncio.run(run())
        assert result == "42"

    def test_invoke_passes_input_to_function(self) -> None:
        received: list[str] = []

        def capture(s: str) -> str:
            received.append(s)
            return "ok"

        adapter = CallableAdapter(fn=capture)

        async def run() -> None:
            await adapter.invoke("test input")

        asyncio.run(run())
        assert received == ["test input"]


# ---------------------------------------------------------------------------
# HTTPAdapter
# ---------------------------------------------------------------------------


class TestHTTPAdapterConstruction:
    def test_default_name(self) -> None:
        adapter = HTTPAdapter(url="http://example.com")
        assert adapter.name == "http-agent"

    def test_custom_name(self) -> None:
        adapter = HTTPAdapter(url="http://example.com", name="my-http-agent")
        assert adapter.name == "my-http-agent"

    def test_headers_stored(self) -> None:
        headers = {"Authorization": "Bearer token"}
        adapter = HTTPAdapter(url="http://example.com", headers=headers)
        assert adapter._headers == headers

    def test_default_input_output_fields(self) -> None:
        adapter = HTTPAdapter(url="http://example.com")
        assert adapter._input_field == "input"
        assert adapter._output_field == "output"

    def test_custom_input_output_fields(self) -> None:
        adapter = HTTPAdapter(
            url="http://example.com",
            input_field="prompt",
            output_field="response",
        )
        assert adapter._input_field == "prompt"
        assert adapter._output_field == "response"


class TestHTTPAdapterExtractField:
    def test_extract_simple_field(self) -> None:
        adapter = HTTPAdapter(url="http://example.com")
        result = adapter._extract_field({"output": "hello"}, "output")
        assert result == "hello"

    def test_extract_nested_field_with_dot_notation(self) -> None:
        adapter = HTTPAdapter(url="http://example.com")
        result = adapter._extract_field(
            {"data": {"output": "nested value"}}, "data.output"
        )
        assert result == "nested value"

    def test_extract_missing_field_returns_empty_string(self) -> None:
        adapter = HTTPAdapter(url="http://example.com")
        result = adapter._extract_field({}, "missing")
        assert result == ""

    def test_extract_non_dict_intermediate_returns_str(self) -> None:
        adapter = HTTPAdapter(url="http://example.com")
        # data.extra where data is a string, not dict
        result = adapter._extract_field({"data": "not_a_dict"}, "data.extra")
        assert result == "not_a_dict"


class TestHTTPAdapterInvoke:
    def _mock_urlopen(self, response_data: dict) -> MagicMock:
        """Create a mock for urlopen that returns JSON response."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_invoke_extracts_output_field(self) -> None:
        adapter = HTTPAdapter(url="http://example.com/agent")
        mock_resp = self._mock_urlopen({"output": "agent response"})

        with patch("agent_eval.adapters.http.urlopen", return_value=mock_resp):
            result = asyncio.run(adapter.invoke("test input"))

        assert result == "agent response"

    def test_invoke_sends_input_in_request_body(self) -> None:
        adapter = HTTPAdapter(url="http://example.com/agent", input_field="prompt")
        mock_resp = self._mock_urlopen({"output": "response"})
        captured_requests: list = []

        def mock_urlopen_fn(req, timeout=None):
            captured_requests.append(req)
            return mock_resp

        with patch("agent_eval.adapters.http.urlopen", side_effect=mock_urlopen_fn):
            asyncio.run(adapter.invoke("my prompt"))

        assert len(captured_requests) == 1
        body = json.loads(captured_requests[0].data.decode("utf-8"))
        assert body["prompt"] == "my prompt"

    def test_invoke_url_error_raises_connection_error(self) -> None:
        adapter = HTTPAdapter(url="http://nonexistent.example.com")

        with patch(
            "agent_eval.adapters.http.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with pytest.raises(ConnectionError, match="HTTP request failed"):
                asyncio.run(adapter.invoke("input"))

    def test_invoke_non_dict_response_returns_str(self) -> None:
        adapter = HTTPAdapter(url="http://example.com/agent")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps("plain string response").encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("agent_eval.adapters.http.urlopen", return_value=mock_resp):
            result = asyncio.run(adapter.invoke("input"))

        assert result == "plain string response"

    def test_invoke_includes_custom_headers(self) -> None:
        adapter = HTTPAdapter(
            url="http://example.com",
            headers={"Authorization": "Bearer token123"},
        )
        mock_resp = self._mock_urlopen({"output": "ok"})
        captured_requests: list = []

        def mock_urlopen_fn(req, timeout=None):
            captured_requests.append(req)
            return mock_resp

        with patch("agent_eval.adapters.http.urlopen", side_effect=mock_urlopen_fn):
            asyncio.run(adapter.invoke("test"))

        req = captured_requests[0]
        assert req.get_header("Authorization") == "Bearer token123"

    def test_invoke_nested_output_field(self) -> None:
        adapter = HTTPAdapter(
            url="http://example.com",
            output_field="data.response",
        )
        mock_resp = self._mock_urlopen({"data": {"response": "deep answer"}})

        with patch("agent_eval.adapters.http.urlopen", return_value=mock_resp):
            result = asyncio.run(adapter.invoke("input"))

        assert result == "deep answer"


# ---------------------------------------------------------------------------
# AutoGenAdapter — ImportError path
# ---------------------------------------------------------------------------


class TestAutoGenAdapterImportError:
    def test_import_error_raised_when_autogen_not_installed(self) -> None:
        with patch.dict("sys.modules", {"autogen": None}):
            from agent_eval.adapters.autogen import AutoGenAdapter
            with pytest.raises(ImportError, match="AutoGen"):
                AutoGenAdapter(agent=MagicMock())

    def test_adapter_name_default(self) -> None:
        import agent_eval.adapters.autogen as autogen_module
        mock_autogen = MagicMock()
        mock_autogen.ConversableAgent = MagicMock
        # Include the adapter module in patch.dict so reload() can find it.
        with patch.dict(
            "sys.modules",
            {"autogen": mock_autogen, "agent_eval.adapters.autogen": autogen_module},
        ):
            importlib.reload(autogen_module)
            adapter = autogen_module.AutoGenAdapter(agent=MagicMock())
            assert adapter.name == "autogen-agent"

    def test_adapter_invoke_with_generate_reply(self) -> None:
        import agent_eval.adapters.autogen as autogen_module
        mock_autogen = MagicMock()
        mock_agent = MagicMock()
        mock_agent.generate_reply.return_value = "agent response"
        mock_autogen.ConversableAgent = MagicMock

        with patch.dict(
            "sys.modules",
            {"autogen": mock_autogen, "agent_eval.adapters.autogen": autogen_module},
        ):
            importlib.reload(autogen_module)
            adapter = autogen_module.AutoGenAdapter(agent=mock_agent, name="test-autogen")

            async def run() -> str:
                return await adapter.invoke("hello")

            result = asyncio.run(run())
            assert result == "agent response"


# ---------------------------------------------------------------------------
# CrewAIAdapter — ImportError path
# ---------------------------------------------------------------------------


class TestCrewAIAdapterImportError:
    def test_import_error_raised_when_crewai_not_installed(self) -> None:
        with patch.dict("sys.modules", {"crewai": None}):
            from agent_eval.adapters.crewai import CrewAIAdapter
            with pytest.raises(ImportError, match="CrewAI"):
                CrewAIAdapter(crew=MagicMock())

    def test_adapter_invoke_with_kickoff(self) -> None:
        import agent_eval.adapters.crewai as crewai_module
        mock_crewai = MagicMock()
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = "crew output"
        mock_crew.kickoff.return_value = mock_result
        mock_crewai.Crew = MagicMock

        with patch.dict(
            "sys.modules",
            {"crewai": mock_crewai, "agent_eval.adapters.crewai": crewai_module},
        ):
            importlib.reload(crewai_module)
            adapter = crewai_module.CrewAIAdapter(crew=mock_crew)

            async def run() -> str:
                return await adapter.invoke("task input")

            result = asyncio.run(run())
            assert result == "crew output"

    def test_adapter_invoke_kickoff_without_raw(self) -> None:
        import agent_eval.adapters.crewai as crewai_module
        mock_crewai = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "plain string result"
        mock_crewai.Crew = MagicMock

        with patch.dict(
            "sys.modules",
            {"crewai": mock_crewai, "agent_eval.adapters.crewai": crewai_module},
        ):
            importlib.reload(crewai_module)
            adapter = crewai_module.CrewAIAdapter(crew=mock_crew)

            async def run() -> str:
                return await adapter.invoke("task")

            result = asyncio.run(run())
            assert result == "plain string result"

    def test_adapter_invoke_without_kickoff_raises(self) -> None:
        import agent_eval.adapters.crewai as crewai_module
        mock_crewai = MagicMock()
        mock_crew = object()  # Has no kickoff method
        mock_crewai.Crew = MagicMock

        with patch.dict(
            "sys.modules",
            {"crewai": mock_crewai, "agent_eval.adapters.crewai": crewai_module},
        ):
            importlib.reload(crewai_module)
            adapter = crewai_module.CrewAIAdapter(crew=mock_crew)

            async def run() -> str:
                return await adapter.invoke("task")

            with pytest.raises(TypeError, match="kickoff"):
                asyncio.run(run())


# ---------------------------------------------------------------------------
# LangChainAdapter — ImportError path
# ---------------------------------------------------------------------------


class TestLangChainAdapterImportError:
    def test_import_error_raised_when_langchain_not_installed(self) -> None:
        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.runnables": None}):
            from agent_eval.adapters.langchain import LangChainAdapter
            with pytest.raises(ImportError, match="LangChain"):
                LangChainAdapter(agent=MagicMock())

    def test_adapter_invoke_with_ainvoke(self) -> None:
        import agent_eval.adapters.langchain as langchain_module
        mock_lc = MagicMock()
        mock_agent = MagicMock()

        async def async_ainvoke(inp: object) -> dict:
            return {"output": "lc result"}

        mock_agent.ainvoke = async_ainvoke
        del mock_agent.invoke  # Remove sync invoke so ainvoke path is taken

        mock_lc.Runnable = MagicMock
        with patch.dict(
            "sys.modules",
            {
                "langchain_core": mock_lc,
                "langchain_core.runnables": mock_lc,
                "agent_eval.adapters.langchain": langchain_module,
            },
        ):
            importlib.reload(langchain_module)
            adapter = langchain_module.LangChainAdapter(agent=mock_agent, name="lc-agent")

            async def run() -> str:
                return await adapter.invoke("question")

            result = asyncio.run(run())
            assert result == "lc result"

    def test_adapter_invoke_with_sync_invoke(self) -> None:
        import agent_eval.adapters.langchain as langchain_module
        mock_lc = MagicMock()
        mock_agent = MagicMock(spec=["invoke"])  # Only has sync invoke
        mock_agent.invoke.return_value = {"output": "sync lc result"}

        mock_lc.Runnable = MagicMock
        with patch.dict(
            "sys.modules",
            {
                "langchain_core": mock_lc,
                "langchain_core.runnables": mock_lc,
                "agent_eval.adapters.langchain": langchain_module,
            },
        ):
            importlib.reload(langchain_module)
            adapter = langchain_module.LangChainAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("question")

            result = asyncio.run(run())
            assert result == "sync lc result"

    def test_adapter_invoke_no_invoke_method_raises(self) -> None:
        import agent_eval.adapters.langchain as langchain_module
        mock_lc = MagicMock()
        mock_agent = object()  # No invoke or ainvoke

        mock_lc.Runnable = MagicMock
        with patch.dict(
            "sys.modules",
            {
                "langchain_core": mock_lc,
                "langchain_core.runnables": mock_lc,
                "agent_eval.adapters.langchain": langchain_module,
            },
        ):
            importlib.reload(langchain_module)
            adapter = langchain_module.LangChainAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("question")

            with pytest.raises(TypeError, match="invoke"):
                asyncio.run(run())

    def test_adapter_invoke_dict_output_extraction(self) -> None:
        import agent_eval.adapters.langchain as langchain_module
        mock_lc = MagicMock()
        mock_agent = MagicMock(spec=["invoke"])
        mock_agent.invoke.return_value = {"output": "extracted", "other": "ignored"}

        mock_lc.Runnable = MagicMock
        with patch.dict(
            "sys.modules",
            {
                "langchain_core": mock_lc,
                "langchain_core.runnables": mock_lc,
                "agent_eval.adapters.langchain": langchain_module,
            },
        ):
            importlib.reload(langchain_module)
            adapter = langchain_module.LangChainAdapter(agent=mock_agent, output_key="output")

            async def run() -> str:
                return await adapter.invoke("q")

            result = asyncio.run(run())
            assert result == "extracted"


# ---------------------------------------------------------------------------
# OpenAIAgentsAdapter — ImportError path
# ---------------------------------------------------------------------------


class TestOpenAIAgentsAdapterImportError:
    def test_import_error_raised_when_sdk_not_installed(self) -> None:
        with patch.dict("sys.modules", {"agents": None}):
            from agent_eval.adapters.openai_agents import OpenAIAgentsAdapter
            with pytest.raises(ImportError, match="OpenAI Agents"):
                OpenAIAgentsAdapter(agent=MagicMock())
