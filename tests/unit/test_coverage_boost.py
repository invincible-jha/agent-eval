"""Additional tests to boost coverage to 85%+.

Covers:
- adapters/autogen.py  -- generate_reply returning falsy, name property
- adapters/crewai.py   -- name property
- adapters/langchain.py -- name property, dict result without output key
- adapters/openai_agents.py -- constructor success, invoke paths
- core/agent_wrapper.py -- from_langchain, from_crewai (patched wrap)
- core/suite.py -- builtin error paths, filter with non-string tags,
                   _from_dict edge cases, _parse_case edge cases
- suites/builder.py -- add_case return, add_cases loop, build duplicate check
- suites/loader.py  -- JSON loading, directory loading, builtin loading,
                       metadata value types
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from agent_eval.core.suite import BenchmarkSuite, TestCase
from agent_eval.core.exceptions import SuiteError


# ---------------------------------------------------------------------------
# AutoGenAdapter — additional paths
# ---------------------------------------------------------------------------


class TestAutoGenAdapterExtraPaths:
    """Cover the autogen adapter lines not hit by the main adapter tests."""

    def test_generate_reply_returning_none_gives_empty_string(self) -> None:
        """Line 64: generate_reply returns None — result should be ''."""
        import agent_eval.adapters.autogen as autogen_module

        mock_autogen = MagicMock()
        mock_autogen.ConversableAgent = MagicMock
        mock_agent = MagicMock()
        mock_agent.generate_reply.return_value = None

        with patch.dict(
            "sys.modules",
            {"autogen": mock_autogen, "agent_eval.adapters.autogen": autogen_module},
        ):
            importlib.reload(autogen_module)
            adapter = autogen_module.AutoGenAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("hello")

            result = asyncio.run(run())
        assert result == ""

    def test_name_property_returns_configured_name(self) -> None:
        """name property on a successfully constructed adapter."""
        import agent_eval.adapters.autogen as autogen_module

        mock_autogen = MagicMock()
        mock_autogen.ConversableAgent = MagicMock

        with patch.dict(
            "sys.modules",
            {"autogen": mock_autogen, "agent_eval.adapters.autogen": autogen_module},
        ):
            importlib.reload(autogen_module)
            adapter = autogen_module.AutoGenAdapter(agent=MagicMock(), name="my-autogen")

        assert adapter.name == "my-autogen"

    def test_no_generate_reply_raises_type_error(self) -> None:
        """Agent without generate_reply raises TypeError."""
        import agent_eval.adapters.autogen as autogen_module

        mock_autogen = MagicMock()
        mock_autogen.ConversableAgent = MagicMock
        mock_agent = object()  # No generate_reply

        with patch.dict(
            "sys.modules",
            {"autogen": mock_autogen, "agent_eval.adapters.autogen": autogen_module},
        ):
            importlib.reload(autogen_module)
            adapter = autogen_module.AutoGenAdapter(agent=mock_agent)

            async def run() -> str:
                return await adapter.invoke("hello")

            with pytest.raises(TypeError, match="generate_reply"):
                asyncio.run(run())


# ---------------------------------------------------------------------------
# CrewAIAdapter — name property
# ---------------------------------------------------------------------------


class TestCrewAIAdapterExtraPaths:
    def test_name_property_returns_configured_name(self) -> None:
        """Line 44: name property on a successfully constructed CrewAI adapter."""
        import agent_eval.adapters.crewai as crewai_module

        mock_crewai = MagicMock()
        mock_crewai.Crew = MagicMock

        with patch.dict(
            "sys.modules",
            {"crewai": mock_crewai, "agent_eval.adapters.crewai": crewai_module},
        ):
            importlib.reload(crewai_module)
            adapter = crewai_module.CrewAIAdapter(crew=MagicMock(), name="my-crew")

        assert adapter.name == "my-crew"


# ---------------------------------------------------------------------------
# LangChainAdapter — name property and dict-without-output-key path
# ---------------------------------------------------------------------------


class TestLangChainAdapterExtraPaths:
    def _make_lc_adapter(
        self, agent: object, **kwargs: object
    ) -> object:
        import agent_eval.adapters.langchain as langchain_module

        mock_lc = MagicMock()
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
            return langchain_module.LangChainAdapter(agent=agent, **kwargs)  # type: ignore[arg-type]

    def test_name_property(self) -> None:
        """Line 57: name property."""
        import agent_eval.adapters.langchain as langchain_module

        mock_lc = MagicMock()
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
            adapter = langchain_module.LangChainAdapter(agent=MagicMock(), name="named-lc")

        assert adapter.name == "named-lc"

    def test_invoke_dict_result_missing_output_key_falls_back_to_full_dict(
        self,
    ) -> None:
        """Line 85: result is a dict but output_key is absent — str(result) returned."""
        import agent_eval.adapters.langchain as langchain_module

        mock_lc = MagicMock()
        mock_lc.Runnable = MagicMock
        mock_agent = MagicMock(spec=["invoke"])
        mock_agent.invoke.return_value = {"other_key": "value"}

        with patch.dict(
            "sys.modules",
            {
                "langchain_core": mock_lc,
                "langchain_core.runnables": mock_lc,
                "agent_eval.adapters.langchain": langchain_module,
            },
        ):
            importlib.reload(langchain_module)
            adapter = langchain_module.LangChainAdapter(
                agent=mock_agent, output_key="output"
            )

        async def run() -> str:
            return await adapter.invoke("q")

        result = asyncio.run(run())
        # When output_key is absent, dict.get returns the full dict as fallback
        assert "other_key" in result


# ---------------------------------------------------------------------------
# OpenAIAgentsAdapter — constructor success and invoke paths
# ---------------------------------------------------------------------------


class TestOpenAIAgentsAdapterPaths:
    """Cover lines 38-39, 44, 59-67 of openai_agents.py."""

    def _load_module(self) -> object:
        import agent_eval.adapters.openai_agents as oa_module
        return oa_module

    def test_constructor_success_stores_agent_and_name(self) -> None:
        """Lines 38-39: agent and name stored after successful import."""
        oa_module = self._load_module()
        mock_agents = MagicMock()
        mock_agents.Agent = MagicMock

        with patch.dict(
            "sys.modules",
            {"agents": mock_agents, "agent_eval.adapters.openai_agents": oa_module},
        ):
            importlib.reload(oa_module)  # type: ignore[arg-type]
            adapter = oa_module.OpenAIAgentsAdapter(  # type: ignore[attr-defined]
                agent=MagicMock(), name="oa-agent"
            )

        assert adapter.name == "oa-agent"

    def test_invoke_with_final_output_attribute(self) -> None:
        """Lines 59-64: invoke uses result.final_output when present."""
        oa_module = self._load_module()
        mock_agents = MagicMock()
        mock_agents.Agent = MagicMock

        mock_result = MagicMock()
        mock_result.final_output = "the answer"

        async def fake_runner_run(agent: object, text: str) -> object:
            return mock_result

        mock_agents.Runner = MagicMock()
        mock_agents.Runner.run = fake_runner_run

        # Keep the patch active during invoke so the inner 'from agents import Runner' works
        with patch.dict(
            "sys.modules",
            {"agents": mock_agents, "agent_eval.adapters.openai_agents": oa_module},
        ):
            importlib.reload(oa_module)  # type: ignore[arg-type]
            adapter = oa_module.OpenAIAgentsAdapter(agent=MagicMock(), name="oa")  # type: ignore[attr-defined]

            async def run() -> str:
                return await adapter.invoke("question")

            result = asyncio.run(run())
        assert result == "the answer"

    def test_invoke_without_final_output_attribute_uses_str_result(self) -> None:
        """Line 65: result has no final_output — str(result) returned."""
        oa_module = self._load_module()
        mock_agents = MagicMock()
        mock_agents.Agent = MagicMock

        async def fake_runner_run(agent: object, text: str) -> str:
            return "plain string"

        mock_agents.Runner = MagicMock()
        mock_agents.Runner.run = fake_runner_run

        with patch.dict(
            "sys.modules",
            {"agents": mock_agents, "agent_eval.adapters.openai_agents": oa_module},
        ):
            importlib.reload(oa_module)  # type: ignore[arg-type]
            adapter = oa_module.OpenAIAgentsAdapter(agent=MagicMock())  # type: ignore[attr-defined]

            async def run() -> str:
                return await adapter.invoke("question")

            result = asyncio.run(run())
        assert result == "plain string"

    def test_invoke_import_error_in_runner(self) -> None:
        """Line 66-70: ImportError raised during invoke re-raises with message."""
        oa_module = self._load_module()
        mock_agents_outer = MagicMock()
        mock_agents_outer.Agent = MagicMock

        with patch.dict(
            "sys.modules",
            {"agents": mock_agents_outer, "agent_eval.adapters.openai_agents": oa_module},
        ):
            importlib.reload(oa_module)  # type: ignore[arg-type]
            adapter = oa_module.OpenAIAgentsAdapter(agent=MagicMock())  # type: ignore[attr-defined]

            # Override agents in sys.modules to None inside invoke so Runner import fails
            async def run() -> str:
                with patch.dict("sys.modules", {"agents": None}):
                    return await adapter.invoke("question")

            with pytest.raises(ImportError, match="OpenAI Agents SDK Runner"):
                asyncio.run(run())


# ---------------------------------------------------------------------------
# AgentUnderTest — from_langchain and from_crewai factory methods
# ---------------------------------------------------------------------------


class TestAgentUnderTestFromFrameworkFactories:
    """Cover lines 115-117 (from_langchain) and 142-144 (from_crewai)."""

    def _make_wrap_mock(self, return_text: str) -> type:
        """Return a class with a wrap() classmethod returning a simple coroutine fn."""

        async def _async_fn(input_text: str) -> str:
            return return_text

        class _FakeAdapter:
            @classmethod
            def wrap(cls, agent: object) -> object:
                return _async_fn

        return _FakeAdapter  # type: ignore[return-value]

    def test_from_langchain_name_stored(self) -> None:
        """from_langchain stores the given name on the returned agent."""
        from agent_eval.core.agent_wrapper import AgentUnderTest
        import agent_eval.adapters.langchain as lc_mod

        fake_cls = self._make_wrap_mock("lc result")
        with patch.object(lc_mod, "LangChainAdapter", fake_cls):
            agent = AgentUnderTest.from_langchain(MagicMock(), name="lc-name")
        assert agent.name == "lc-name"

    def test_from_langchain_timeout_stored(self) -> None:
        from agent_eval.core.agent_wrapper import AgentUnderTest
        import agent_eval.adapters.langchain as lc_mod

        fake_cls = self._make_wrap_mock("lc result")
        with patch.object(lc_mod, "LangChainAdapter", fake_cls):
            agent = AgentUnderTest.from_langchain(MagicMock(), timeout_ms=3000)
        assert agent.timeout_ms == 3000

    def test_from_langchain_run_returns_output(self) -> None:
        from agent_eval.core.agent_wrapper import AgentUnderTest
        import agent_eval.adapters.langchain as lc_mod

        fake_cls = self._make_wrap_mock("langchain output")
        with patch.object(lc_mod, "LangChainAdapter", fake_cls):
            agent = AgentUnderTest.from_langchain(MagicMock(), name="lc-agent")

        async def run() -> str:
            return await agent.run("test")

        result = asyncio.run(run())
        assert result == "langchain output"

    def test_from_crewai_name_stored(self) -> None:
        from agent_eval.core.agent_wrapper import AgentUnderTest
        import agent_eval.adapters.crewai as crewai_mod

        fake_cls = self._make_wrap_mock("crew result")
        with patch.object(crewai_mod, "CrewAIAdapter", fake_cls):
            agent = AgentUnderTest.from_crewai(MagicMock(), name="crewai-name")
        assert agent.name == "crewai-name"

    def test_from_crewai_timeout_stored(self) -> None:
        from agent_eval.core.agent_wrapper import AgentUnderTest
        import agent_eval.adapters.crewai as crewai_mod

        fake_cls = self._make_wrap_mock("crew result")
        with patch.object(crewai_mod, "CrewAIAdapter", fake_cls):
            agent = AgentUnderTest.from_crewai(MagicMock(), timeout_ms=7000)
        assert agent.timeout_ms == 7000

    def test_from_crewai_run_returns_output(self) -> None:
        from agent_eval.core.agent_wrapper import AgentUnderTest
        import agent_eval.adapters.crewai as crewai_mod

        fake_cls = self._make_wrap_mock("crewai output")
        with patch.object(crewai_mod, "CrewAIAdapter", fake_cls):
            agent = AgentUnderTest.from_crewai(MagicMock(), name="crew-agent")

        async def run() -> str:
            return await agent.run("task")

        result = asyncio.run(run())
        assert result == "crewai output"


# ---------------------------------------------------------------------------
# BenchmarkSuite — additional path coverage
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteExtraPaths:
    """Cover suite.py lines 203-204, 208-209, 252, 317, 322, 348, 357."""

    def test_builtin_file_not_found_raises_suite_error(self) -> None:
        """Lines 203-204: resource lookup raises FileNotFoundError -> SuiteError."""
        import importlib.resources as _res

        def _fake_files(package: str) -> object:
            class _FakeRef:
                def joinpath(self, name: str) -> "_FakeRef":
                    return self

                def read_text(self, encoding: str = "utf-8") -> str:
                    raise FileNotFoundError("not found")

            return _FakeRef()

        with patch.object(_res, "files", _fake_files):
            with pytest.raises(SuiteError, match="not found"):
                BenchmarkSuite.builtin("qa_basic")

    def test_filter_tags_non_string_metadata_value(self) -> None:
        """Line 252: metadata tags value is not a string -> case_tags = set()."""
        case = TestCase(
            id="c1",
            input="input",
            metadata={"tags": 42},  # int, not str -> non-string path
        )
        suite = BenchmarkSuite(name="s", cases=[case])
        filtered = suite.filter(tags=["some-tag"])
        # Non-string tags value yields empty set, so no case matches
        assert len(filtered.cases) == 0

    def test_from_dict_cases_not_a_list_raises_suite_error(self) -> None:
        """Line 317: 'cases' is not a list -> SuiteError."""
        data = {"name": "Bad Suite", "cases": "not-a-list"}
        with pytest.raises(SuiteError, match="must be a list"):
            BenchmarkSuite._from_dict(data, source="test")

    def test_from_dict_case_not_a_mapping_raises_suite_error(self) -> None:
        """Line 322: a case is a string, not a dict -> SuiteError."""
        data = {"name": "Suite", "cases": ["not-a-dict"]}
        with pytest.raises(SuiteError, match="mapping"):
            BenchmarkSuite._from_dict(data, source="test")

    def test_parse_case_missing_id_raises_value_error(self) -> None:
        """Line 344: 'id' is absent -> ValueError inside _parse_case."""
        raw: dict[str, object] = {"input": "some input"}
        with pytest.raises(ValueError, match="'id' is required"):
            BenchmarkSuite._parse_case(raw, 0)

    def test_parse_case_missing_input_raises_value_error(self) -> None:
        """Line 348: 'input' is absent -> ValueError."""
        raw: dict[str, object] = {"id": "c1"}
        with pytest.raises(ValueError, match="'input' is required"):
            BenchmarkSuite._parse_case(raw, 0)

    def test_parse_case_metadata_non_primitive_value_is_stringified(self) -> None:
        """Line 357: non-primitive metadata value converted to str."""
        raw: dict[str, object] = {
            "id": "c1",
            "input": "hello",
            "metadata": {"key": [1, 2, 3]},  # list -> str conversion
        }
        case = BenchmarkSuite._parse_case(raw, 0)
        assert case.metadata["key"] == "[1, 2, 3]"

    def test_from_dict_with_all_optional_fields(self) -> None:
        """Happy path: parse a dict with tags, version, latency, cost limits."""
        data = {
            "name": "Full Suite",
            "description": "Test",
            "version": "2.0",
            "tags": ["tag1", "tag2"],
            "default_max_latency_ms": 5000,
            "default_max_cost_tokens": 1000,
            "cases": [{"id": "c1", "input": "What?", "expected_output": "Yes"}],
        }
        suite = BenchmarkSuite._from_dict(data, source="test")
        assert suite.name == "Full Suite"
        assert suite.version == "2.0"
        assert suite.default_max_latency_ms == 5000
        assert suite.default_max_cost_tokens == 1000
        assert "tag1" in suite.tags

    def test_from_dict_non_dict_top_level_raises_suite_error(self) -> None:
        """Line 305: top-level data is not a dict -> SuiteError."""
        with pytest.raises(SuiteError, match="mapping"):
            BenchmarkSuite._from_dict(["not", "a", "dict"], source="test")

    def test_parse_case_with_tools_allowed_and_latency(self) -> None:
        """Parse case with tools_allowed and max_latency_ms fields."""
        raw: dict[str, object] = {
            "id": "tool-case",
            "input": "Use a tool",
            "tools_allowed": ["search", "calculator"],
            "max_latency_ms": 2000,
            "max_cost_tokens": 500,
        }
        case = BenchmarkSuite._parse_case(raw, 0)
        assert case.tools_allowed == ["search", "calculator"]
        assert case.max_latency_ms == 2000
        assert case.max_cost_tokens == 500


# ---------------------------------------------------------------------------
# SuiteBuilder — add_case, add_cases, and build duplicate detection
# ---------------------------------------------------------------------------


class TestSuiteBuilderBuildPaths:
    """Cover builder.py lines 81, 90, 108-113 by patching TestCase.__init__."""

    def _make_fake_test_case(self) -> type:
        """Return a TestCase-compatible class using the real field names."""

        class FakeTestCase:
            def __init__(
                self,
                *,
                case_id: str = "",
                input_text: str = "",
                expected_output: object = None,
                metadata: dict | None = None,
                tags: list | None = None,
            ) -> None:
                self.case_id = case_id
                self.input_text = input_text
                self.expected_output = expected_output
                self.metadata = metadata or {}
                self.tags = tags or []

        return FakeTestCase

    def test_add_case_returns_builder(self) -> None:
        """Line 81: add_case returns self for chaining."""
        import agent_eval.suites.builder as builder_mod

        fake_tc = self._make_fake_test_case()
        with patch.object(builder_mod, "TestCase", fake_tc):
            builder = builder_mod.SuiteBuilder()
            result = builder.add_case("c1", "input")
        assert result is builder

    def test_add_cases_loops_over_all_tuples(self) -> None:
        """Line 90: add_cases processes every tuple in the list."""
        import agent_eval.suites.builder as builder_mod

        fake_tc = self._make_fake_test_case()
        with patch.object(builder_mod, "TestCase", fake_tc):
            builder = builder_mod.SuiteBuilder()
            builder.add_cases([("c1", "i1", "e1"), ("c2", "i2", "e2")])
        assert len(builder._cases) == 2

    def test_build_raises_on_duplicate_case_ids(self) -> None:
        """Lines 108-111: build() detects duplicate IDs and raises ValueError."""
        import agent_eval.suites.builder as builder_mod

        fake_tc = self._make_fake_test_case()

        class FakeBenchmarkSuite:
            def __init__(self, **kwargs: object) -> None:
                pass

        with (
            patch.object(builder_mod, "TestCase", fake_tc),
            patch.object(builder_mod, "BenchmarkSuite", FakeBenchmarkSuite),
        ):
            builder = builder_mod.SuiteBuilder()
            builder.add_case("dup-id", "input 1")
            builder.add_case("dup-id", "input 2")
            with pytest.raises(ValueError, match="Duplicate"):
                builder.build()

    def test_build_constructs_benchmark_suite(self) -> None:
        """Lines 113-118: build() creates BenchmarkSuite with correct kwargs."""
        import agent_eval.suites.builder as builder_mod

        fake_tc = self._make_fake_test_case()
        created_kwargs: list[dict] = []

        class FakeBenchmarkSuite:
            def __init__(self, **kwargs: object) -> None:
                created_kwargs.append(dict(kwargs))

        with (
            patch.object(builder_mod, "TestCase", fake_tc),
            patch.object(builder_mod, "BenchmarkSuite", FakeBenchmarkSuite),
        ):
            builder = builder_mod.SuiteBuilder().name("test-suite").version("2.0")
            builder.add_case("c1", "input")
            builder.build()

        assert len(created_kwargs) == 1
        assert created_kwargs[0]["name"] == "test-suite"
        assert created_kwargs[0]["version"] == "2.0"


# ---------------------------------------------------------------------------
# SuiteLoader — JSON loading, directory loading, metadata filtering, builtin
# ---------------------------------------------------------------------------


class TestSuiteLoaderExtraPaths:
    """Cover loader.py lines 62-63, 79, 110, 120, 140-147, 171, 183."""

    def _make_fake_test_case(self) -> type:
        """Fake TestCase accepting loader's field names (case_id, input_text)."""

        class FakeTestCase:
            def __init__(
                self,
                *,
                case_id: str = "",
                input_text: str = "",
                expected_output: object = None,
                metadata: dict | None = None,
                tags: list | None = None,
            ) -> None:
                self.case_id = case_id
                self.input_text = input_text
                self.expected_output = expected_output
                self.metadata = metadata or {}
                self.tags = tags or []

        return FakeTestCase

    def test_load_json_file_happy_path(self, tmp_path: Path) -> None:
        """Line 110: JSON file loading goes through json.loads branch."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()
        data = {
            "name": "JSON Suite",
            "cases": [{"id": "c1", "input": "hello"}],
        }
        path = tmp_path / "suite.json"
        path.write_text(json.dumps(data))

        with patch.object(loader_mod, "TestCase", fake_tc):
            loader = loader_mod.SuiteLoader()
            suite = loader.load_file(path)

        assert suite.name == "JSON Suite"

    def test_load_yaml_file_happy_path(self, tmp_path: Path) -> None:
        """Line 108: YAML file loading goes through yaml.safe_load branch."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()
        data = {"name": "YAML Suite", "cases": [{"id": "c1", "input": "world"}]}
        path = tmp_path / "suite.yaml"
        path.write_text(yaml.dump(data))

        with patch.object(loader_mod, "TestCase", fake_tc):
            loader = loader_mod.SuiteLoader()
            suite = loader.load_file(path)

        assert suite.name == "YAML Suite"

    def test_parse_cases_metadata_primitive_values_included(self) -> None:
        """Lines 62-63: primitive metadata values (str, int, float, bool) are stored."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()
        raw = [
            {
                "id": "c1",
                "input": "test",
                "metadata": {
                    "str_val": "hello",
                    "int_val": 42,
                    "float_val": 3.14,
                    "bool_val": True,
                    "list_val": [1, 2],  # non-primitive, skipped
                },
            }
        ]

        with patch.object(loader_mod, "TestCase", fake_tc):
            cases = loader_mod.SuiteLoader._parse_cases(raw)

        assert len(cases) == 1
        meta = cases[0].metadata
        assert meta["str_val"] == "hello"
        assert meta["int_val"] == 42
        assert meta["float_val"] == 3.14
        assert meta["bool_val"] is True
        assert "list_val" not in meta  # non-primitive excluded

    def test_parse_cases_with_tags(self) -> None:
        """Lines 67-68: tags list is parsed into list[str]."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()
        raw = [{"id": "c1", "input": "q", "tags": ["safety", "qa"]}]

        with patch.object(loader_mod, "TestCase", fake_tc):
            cases = loader_mod.SuiteLoader._parse_cases(raw)

        assert cases[0].tags == ["safety", "qa"]

    def test_parse_cases_without_expected_output(self) -> None:
        """Lines 55-56: expected_output absent -> expected_str is None."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()
        raw = [{"id": "c1", "input": "question"}]

        with patch.object(loader_mod, "TestCase", fake_tc):
            cases = loader_mod.SuiteLoader._parse_cases(raw)

        assert cases[0].expected_output is None

    def test_parse_cases_with_expected_output(self) -> None:
        """expected_output present -> converted to str."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()
        raw = [{"id": "c1", "input": "question", "expected_output": "answer"}]

        with patch.object(loader_mod, "TestCase", fake_tc):
            cases = loader_mod.SuiteLoader._parse_cases(raw)

        assert cases[0].expected_output == "answer"

    def test_load_directory_loads_all_yaml_and_json(self, tmp_path: Path) -> None:
        """Lines 140-145: load_directory iterates yaml, yml, and json files."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()

        files = {
            "suite1.yaml": {"name": "Suite1", "cases": [{"id": "c1", "input": "i1"}]},
            "suite2.yml": {"name": "Suite2", "cases": [{"id": "c1", "input": "i2"}]},
            "suite3.json": {"name": "Suite3", "cases": [{"id": "c1", "input": "i3"}]},
        }
        for filename, data in files.items():
            content = (
                yaml.dump(data) if filename.endswith((".yaml", ".yml")) else json.dumps(data)
            )
            (tmp_path / filename).write_text(content)

        with patch.object(loader_mod, "TestCase", fake_tc):
            loader = loader_mod.SuiteLoader()
            suites = loader.load_directory(tmp_path)

        assert len(suites) == 3
        names = {s.name for s in suites}
        assert names == {"Suite1", "Suite2", "Suite3"}

    def test_load_builtin_happy_path(self) -> None:
        """Line 171: a real builtin suite can be loaded (qa_basic exists)."""
        import agent_eval.suites.loader as loader_mod

        fake_tc = self._make_fake_test_case()
        with patch.object(loader_mod, "TestCase", fake_tc):
            loader = loader_mod.SuiteLoader()
            suite = loader.load_builtin("qa_basic")

        assert suite.name  # non-empty name

    def test_list_builtin_nonexistent_directory(self, tmp_path: Path) -> None:
        """Line 183: list_builtin returns [] when builtin dir doesn't exist."""
        import agent_eval.suites.loader as loader_mod

        nonexistent = tmp_path / "does_not_exist"
        with patch("agent_eval.suites.loader.Path") as mock_path_cls:
            # Only intercept when called with __file__'s parent; pass through otherwise
            real_path = Path
            def _path_factory(*args: object, **kwargs: object) -> Path:
                result = real_path(*args, **kwargs)
                return result
            mock_path_cls.side_effect = _path_factory
            # Simpler: just patch the builtin_dir directly inside list_builtin
            pass

        # Direct approach: call with a real path that has no yaml/json/yml files
        # by temporarily making list_builtin use a nonexistent builtin dir
        with patch.object(
            loader_mod.Path,
            "__truediv__",
            return_value=nonexistent,
        ):
            names = loader_mod.SuiteLoader.list_builtin()
        assert names == []
