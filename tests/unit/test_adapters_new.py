"""Unit tests for the new agent-eval adapters.

Covers:
- adapters.anthropic_sdk  — AnthropicAdapter ImportError path and invoke logic
- adapters.microsoft_agents — MicrosoftAgentAdapter ImportError path and invoke logic
"""
from __future__ import annotations

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# AnthropicAdapter
# ===========================================================================


class TestAnthropicAdapterImportError:
    def test_import_error_raised_when_anthropic_not_installed(self) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
            from agent_eval.adapters.anthropic_sdk import AnthropicAdapter
            with pytest.raises(ImportError, match="Anthropic"):
                AnthropicAdapter(client=MagicMock())

    def test_adapter_name_default(self) -> None:
        import agent_eval.adapters.anthropic_sdk as anthropic_module
        mock_anthropic = MagicMock()

        with patch.dict(
            "sys.modules",
            {"anthropic": mock_anthropic, "agent_eval.adapters.anthropic_sdk": anthropic_module},
        ):
            importlib.reload(anthropic_module)
            adapter = anthropic_module.AnthropicAdapter(client=MagicMock())
            assert adapter.name == "anthropic-agent"

    def test_adapter_name_custom(self) -> None:
        import agent_eval.adapters.anthropic_sdk as anthropic_module
        mock_anthropic = MagicMock()

        with patch.dict(
            "sys.modules",
            {"anthropic": mock_anthropic, "agent_eval.adapters.anthropic_sdk": anthropic_module},
        ):
            importlib.reload(anthropic_module)
            adapter = anthropic_module.AnthropicAdapter(
                client=MagicMock(), name="my-claude"
            )
            assert adapter.name == "my-claude"

    def test_adapter_invoke_extracts_text_block(self) -> None:
        import agent_eval.adapters.anthropic_sdk as anthropic_module
        mock_anthropic = MagicMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello from Claude!"

        mock_response = MagicMock()
        mock_response.content = [text_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"anthropic": mock_anthropic, "agent_eval.adapters.anthropic_sdk": anthropic_module},
        ):
            importlib.reload(anthropic_module)
            adapter = anthropic_module.AnthropicAdapter(client=mock_client)

            async def run() -> str:
                return await adapter.invoke("Hello!")

            result = asyncio.run(run())
            assert result == "Hello from Claude!"

    def test_adapter_invoke_raises_when_no_messages_api(self) -> None:
        import agent_eval.adapters.anthropic_sdk as anthropic_module
        mock_anthropic = MagicMock()

        # Client with no messages attribute
        mock_client = object()

        with patch.dict(
            "sys.modules",
            {"anthropic": mock_anthropic, "agent_eval.adapters.anthropic_sdk": anthropic_module},
        ):
            importlib.reload(anthropic_module)
            adapter = anthropic_module.AnthropicAdapter(client=mock_client)

            async def run() -> str:
                return await adapter.invoke("Hello!")

            with pytest.raises(TypeError, match="messages.create"):
                asyncio.run(run())

    def test_adapter_invoke_passes_system_prompt(self) -> None:
        import agent_eval.adapters.anthropic_sdk as anthropic_module
        mock_anthropic = MagicMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [text_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"anthropic": mock_anthropic, "agent_eval.adapters.anthropic_sdk": anthropic_module},
        ):
            importlib.reload(anthropic_module)
            adapter = anthropic_module.AnthropicAdapter(
                client=mock_client, system_prompt="You are a helpful assistant."
            )

            async def run() -> str:
                return await adapter.invoke("Hi")

            asyncio.run(run())

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("system") == "You are a helpful assistant."

    def test_adapter_invoke_uses_specified_model(self) -> None:
        import agent_eval.adapters.anthropic_sdk as anthropic_module
        mock_anthropic = MagicMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "OK"

        mock_response = MagicMock()
        mock_response.content = [text_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"anthropic": mock_anthropic, "agent_eval.adapters.anthropic_sdk": anthropic_module},
        ):
            importlib.reload(anthropic_module)
            adapter = anthropic_module.AnthropicAdapter(
                client=mock_client, model="claude-3-haiku-20240307"
            )

            async def run() -> str:
                return await adapter.invoke("test")

            asyncio.run(run())

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("model") == "claude-3-haiku-20240307"

    def test_adapter_invoke_fallback_to_str_for_non_text_blocks(self) -> None:
        import agent_eval.adapters.anthropic_sdk as anthropic_module
        mock_anthropic = MagicMock()

        # Response with no text-type blocks
        tool_block = MagicMock()
        tool_block.type = "tool_use"

        mock_response = MagicMock()
        mock_response.content = [tool_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"anthropic": mock_anthropic, "agent_eval.adapters.anthropic_sdk": anthropic_module},
        ):
            importlib.reload(anthropic_module)
            adapter = anthropic_module.AnthropicAdapter(client=mock_client)

            async def run() -> str:
                return await adapter.invoke("test")

            result = asyncio.run(run())
            # Falls back to str(response)
            assert isinstance(result, str)


# ===========================================================================
# MicrosoftAgentAdapter
# ===========================================================================


class TestMicrosoftAgentAdapterImportError:
    def test_import_error_raised_when_sdk_not_installed(self) -> None:
        with patch.dict("sys.modules", {"microsoft.agents": None, "microsoft": None}):
            from agent_eval.adapters.microsoft_agents import MicrosoftAgentAdapter
            with pytest.raises(ImportError, match="Microsoft"):
                MicrosoftAgentAdapter(agent=MagicMock())

    def test_adapter_name_default(self) -> None:
        import agent_eval.adapters.microsoft_agents as ms_module

        mock_ms = MagicMock()
        mock_ms_agents = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "microsoft": mock_ms,
                "microsoft.agents": mock_ms_agents,
                "agent_eval.adapters.microsoft_agents": ms_module,
            },
        ):
            importlib.reload(ms_module)
            adapter = ms_module.MicrosoftAgentAdapter(agent=MagicMock())
            assert adapter.name == "microsoft-agent"

    def test_adapter_name_custom(self) -> None:
        import agent_eval.adapters.microsoft_agents as ms_module

        mock_ms = MagicMock()
        mock_ms_agents = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "microsoft": mock_ms,
                "microsoft.agents": mock_ms_agents,
                "agent_eval.adapters.microsoft_agents": ms_module,
            },
        ):
            importlib.reload(ms_module)
            adapter = ms_module.MicrosoftAgentAdapter(agent=MagicMock(), name="teams-bot")
            assert adapter.name == "teams-bot"

    def test_adapter_invoke_with_on_message_activity(self) -> None:
        import agent_eval.adapters.microsoft_agents as ms_module

        mock_ms = MagicMock()
        mock_ms_agents = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "microsoft": mock_ms,
                "microsoft.agents": mock_ms_agents,
                "agent_eval.adapters.microsoft_agents": ms_module,
            },
        ):
            importlib.reload(ms_module)

            # Simulated bot that writes a response via send_activity
            async def mock_on_message(context: object) -> None:
                await context.send_activity("Bot says hello!")

            mock_agent = MagicMock()
            mock_agent.on_message_activity = mock_on_message
            del mock_agent.on_turn

            adapter = ms_module.MicrosoftAgentAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("Hello!")

            result = asyncio.run(run())
            assert result == "Bot says hello!"

    def test_adapter_invoke_with_on_turn_fallback(self) -> None:
        import agent_eval.adapters.microsoft_agents as ms_module

        mock_ms = MagicMock()
        mock_ms_agents = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "microsoft": mock_ms,
                "microsoft.agents": mock_ms_agents,
                "agent_eval.adapters.microsoft_agents": ms_module,
            },
        ):
            importlib.reload(ms_module)

            async def mock_on_turn(context: object) -> None:
                await context.send_activity("Handled by on_turn")

            mock_agent = MagicMock()
            del mock_agent.on_message_activity
            mock_agent.on_turn = mock_on_turn

            adapter = ms_module.MicrosoftAgentAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("Hello!")

            result = asyncio.run(run())
            assert result == "Handled by on_turn"

    def test_adapter_invoke_raises_when_no_handlers(self) -> None:
        import agent_eval.adapters.microsoft_agents as ms_module

        mock_ms = MagicMock()
        mock_ms_agents = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "microsoft": mock_ms,
                "microsoft.agents": mock_ms_agents,
                "agent_eval.adapters.microsoft_agents": ms_module,
            },
        ):
            importlib.reload(ms_module)

            # Plain object with no handlers
            mock_agent = object()
            adapter = ms_module.MicrosoftAgentAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("Hello!")

            with pytest.raises(TypeError, match="on_message_activity"):
                asyncio.run(run())

    def test_adapter_invoke_returns_empty_string_when_no_response_sent(self) -> None:
        import agent_eval.adapters.microsoft_agents as ms_module

        mock_ms = MagicMock()
        mock_ms_agents = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "microsoft": mock_ms,
                "microsoft.agents": mock_ms_agents,
                "agent_eval.adapters.microsoft_agents": ms_module,
            },
        ):
            importlib.reload(ms_module)

            async def silent_handler(context: object) -> None:
                pass  # Never calls send_activity

            mock_agent = MagicMock()
            mock_agent.on_message_activity = silent_handler
            del mock_agent.on_turn

            adapter = ms_module.MicrosoftAgentAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("Hello!")

            result = asyncio.run(run())
            assert result == ""

    def test_minimal_turn_context_captures_multiple_activities(self) -> None:
        import agent_eval.adapters.microsoft_agents as ms_module

        mock_ms = MagicMock()
        mock_ms_agents = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "microsoft": mock_ms,
                "microsoft.agents": mock_ms_agents,
                "agent_eval.adapters.microsoft_agents": ms_module,
            },
        ):
            importlib.reload(ms_module)

            async def multi_response(context: object) -> None:
                await context.send_activity("First part.")
                await context.send_activity("Second part.")

            mock_agent = MagicMock()
            mock_agent.on_message_activity = multi_response
            del mock_agent.on_turn

            adapter = ms_module.MicrosoftAgentAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("go")

            result = asyncio.run(run())
            assert "First part." in result
            assert "Second part." in result
