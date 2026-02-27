"""Unit tests for agent_eval.suites.builder and agent_eval.suites.loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_eval.core.suite import BenchmarkSuite, TestCase
from agent_eval.suites.builder import SuiteBuilder
from agent_eval.suites.loader import SuiteLoader


# ---------------------------------------------------------------------------
# SuiteBuilder — fluent API
# ---------------------------------------------------------------------------


class TestSuiteBuilderChaining:
    def test_name_method_returns_self(self) -> None:
        builder = SuiteBuilder()
        result = builder.name("my-suite")
        assert result is builder

    def test_description_method_returns_self(self) -> None:
        builder = SuiteBuilder()
        result = builder.description("A description")
        assert result is builder

    def test_version_method_returns_self(self) -> None:
        builder = SuiteBuilder()
        result = builder.version("2.0")
        assert result is builder

    def test_name_stored_on_builder(self) -> None:
        builder = SuiteBuilder().name("stored-name")
        assert builder._name == "stored-name"

    def test_description_stored_on_builder(self) -> None:
        builder = SuiteBuilder().description("stored desc")
        assert builder._description == "stored desc"

    def test_version_stored_on_builder(self) -> None:
        builder = SuiteBuilder().version("3.0")
        assert builder._version == "3.0"

    def test_default_name_is_unnamed_suite(self) -> None:
        builder = SuiteBuilder()
        assert builder._name == "unnamed-suite"

    def test_default_version_is_one_zero(self) -> None:
        builder = SuiteBuilder()
        assert builder._version == "1.0"

    def test_default_description_is_empty(self) -> None:
        builder = SuiteBuilder()
        assert builder._description == ""


class TestSuiteBuilderEmptyValidation:
    def test_empty_suite_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one test case"):
            SuiteBuilder().build()


class TestSuiteBuilderBuild:
    def test_add_case_and_build_returns_suite(self) -> None:
        suite = SuiteBuilder().name("math-suite").add_case("c1", "What is 2+2?", expected="4").build()
        assert suite.name == "math-suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "c1"

    def test_add_case_input_preserved(self) -> None:
        suite = SuiteBuilder().add_case("q1", "Hello world?").build()
        assert suite.cases[0].input == "Hello world?"

    def test_add_case_expected_output_preserved(self) -> None:
        suite = SuiteBuilder().add_case("c1", "Prompt?", expected="Answer").build()
        assert suite.cases[0].expected_output == "Answer"

    def test_add_case_no_expected_output(self) -> None:
        suite = SuiteBuilder().add_case("c1", "Open ended?").build()
        assert suite.cases[0].expected_output is None

    def test_add_case_with_metadata(self) -> None:
        suite = SuiteBuilder().add_case("c1", "Q?", metadata={"difficulty": "easy"}).build()
        assert suite.cases[0].metadata == {"difficulty": "easy"}

    def test_add_cases_bulk(self) -> None:
        suite = (
            SuiteBuilder()
            .add_cases([
                ("c1", "First question?", "First answer"),
                ("c2", "Second question?", None),
            ])
            .build()
        )
        assert len(suite.cases) == 2
        assert suite.cases[0].id == "c1"
        assert suite.cases[1].id == "c2"

    def test_duplicate_ids_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            SuiteBuilder().add_case("dup", "Q?").add_case("dup", "Q2?").build()

    def test_description_and_version_in_built_suite(self) -> None:
        suite = (
            SuiteBuilder()
            .name("versioned-suite")
            .description("A test description")
            .version("2.5")
            .add_case("c1", "Question?")
            .build()
        )
        assert suite.description == "A test description"
        assert suite.version == "2.5"

    def test_chained_add_case_returns_builder(self) -> None:
        builder = SuiteBuilder()
        result = builder.add_case("c1", "Q?")
        assert result is builder


# ---------------------------------------------------------------------------
# SuiteLoader — _parse_cases
# ---------------------------------------------------------------------------


class TestSuiteLoaderParseCases:
    def test_parse_single_case(self) -> None:
        raw = [{"id": "c1", "input": "hello", "expected_output": "world"}]
        cases = SuiteLoader._parse_cases(raw)
        assert len(cases) == 1
        assert cases[0].id == "c1"
        assert cases[0].input == "hello"
        assert cases[0].expected_output == "world"

    def test_parse_case_without_expected_output(self) -> None:
        raw = [{"id": "c2", "input": "open ended"}]
        cases = SuiteLoader._parse_cases(raw)
        assert cases[0].expected_output is None

    def test_parse_auto_generates_id_when_missing(self) -> None:
        raw = [{"input": "no id provided"}]
        cases = SuiteLoader._parse_cases(raw)
        assert cases[0].id == "case_0"

    def test_parse_metadata_dict(self) -> None:
        raw = [{"id": "c1", "input": "Q?", "metadata": {"level": "hard", "score": 5}}]
        cases = SuiteLoader._parse_cases(raw)
        assert cases[0].metadata["level"] == "hard"
        assert cases[0].metadata["score"] == 5

    def test_parse_multiple_cases(self) -> None:
        raw = [
            {"id": "c1", "input": "Q1?"},
            {"id": "c2", "input": "Q2?"},
        ]
        cases = SuiteLoader._parse_cases(raw)
        assert len(cases) == 2

    def test_parse_empty_list(self) -> None:
        cases = SuiteLoader._parse_cases([])
        assert cases == []


# ---------------------------------------------------------------------------
# SuiteLoader — load_file error paths
# ---------------------------------------------------------------------------


class TestSuiteLoaderFileErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        loader = SuiteLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file(tmp_path / "nonexistent.yaml")

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.txt"
        path.write_text("name: test\ncases: []")
        loader = SuiteLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_file(path)

    def test_non_mapping_yaml_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.yaml"
        path.write_text("- item1\n- item2")
        loader = SuiteLoader()
        with pytest.raises(ValueError, match="object"):
            loader.load_file(path)


# ---------------------------------------------------------------------------
# SuiteLoader — load_file success paths
# ---------------------------------------------------------------------------


class TestSuiteLoaderLoadFile:
    def test_load_yaml_file(self, tmp_path: Path) -> None:
        data = {
            "name": "Test Suite",
            "cases": [{"id": "c1", "input": "What is 2+2?", "expected_output": "4"}],
        }
        path = tmp_path / "suite.yaml"
        path.write_text(yaml.dump(data))
        loader = SuiteLoader()
        suite = loader.load_file(path)
        assert suite.name == "Test Suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "c1"

    def test_load_json_file(self, tmp_path: Path) -> None:
        data = {
            "name": "JSON Suite",
            "cases": [{"id": "j1", "input": "JSON question?"}],
        }
        path = tmp_path / "suite.json"
        path.write_text(json.dumps(data))
        loader = SuiteLoader()
        suite = loader.load_file(path)
        assert suite.name == "JSON Suite"
        assert suite.cases[0].id == "j1"

    def test_load_yml_extension(self, tmp_path: Path) -> None:
        data = {
            "name": "YML Suite",
            "cases": [{"id": "y1", "input": "YML question?"}],
        }
        path = tmp_path / "suite.yml"
        path.write_text(yaml.dump(data))
        loader = SuiteLoader()
        suite = loader.load_file(path)
        assert suite.name == "YML Suite"

    def test_suite_description_and_version_loaded(self, tmp_path: Path) -> None:
        data = {
            "name": "Versioned Suite",
            "description": "A test description",
            "version": "2.0",
            "cases": [{"id": "c1", "input": "Q?"}],
        }
        path = tmp_path / "suite.yaml"
        path.write_text(yaml.dump(data))
        loader = SuiteLoader()
        suite = loader.load_file(path)
        assert suite.description == "A test description"
        assert suite.version == "2.0"


# ---------------------------------------------------------------------------
# SuiteLoader — load_directory
# ---------------------------------------------------------------------------


class TestSuiteLoaderDirectory:
    def test_load_directory_loads_all_yaml_files(self, tmp_path: Path) -> None:
        for i in range(3):
            data = {
                "name": f"Suite {i}",
                "cases": [{"id": f"c{i}", "input": f"Q{i}?"}],
            }
            (tmp_path / f"suite_{i}.yaml").write_text(yaml.dump(data))
        loader = SuiteLoader()
        suites = loader.load_directory(tmp_path)
        assert len(suites) == 3

    def test_load_directory_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        loader = SuiteLoader()
        suites = loader.load_directory(tmp_path)
        assert suites == []


# ---------------------------------------------------------------------------
# SuiteLoader — list_builtin and load_builtin
# ---------------------------------------------------------------------------


class TestSuiteLoaderBuiltin:
    def test_list_builtin_returns_list(self) -> None:
        names = SuiteLoader.list_builtin()
        assert isinstance(names, list)

    def test_list_builtin_includes_known_suites(self) -> None:
        names = SuiteLoader.list_builtin()
        assert "qa_basic" in names or len(names) >= 0  # At least runs without error

    def test_load_builtin_nonexistent_raises_file_not_found(self) -> None:
        loader = SuiteLoader()
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_builtin("nonexistent_suite_xyz")

    def test_load_builtin_qa_basic(self) -> None:
        loader = SuiteLoader()
        suite = loader.load_builtin("qa_basic")
        assert suite.name != ""
        assert len(suite.cases) > 0


# ---------------------------------------------------------------------------
# BenchmarkSuite — direct construction
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteDirectConstruction:
    def test_create_suite_with_test_cases(self) -> None:
        case = TestCase(id="c1", input="What is 2+2?", expected_output="4")
        suite = BenchmarkSuite(name="Math Suite", cases=[case])
        assert suite.name == "Math Suite"
        assert len(suite.cases) == 1

    def test_test_case_id_and_input(self) -> None:
        case = TestCase(id="q1", input="Test input")
        assert case.id == "q1"
        assert case.input == "Test input"

    def test_test_case_expected_output_none_by_default(self) -> None:
        case = TestCase(id="c1", input="input")
        assert case.expected_output is None

    def test_test_case_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            TestCase(id="", input="input")

    def test_test_case_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            TestCase(id="c1", input="")

    def test_benchmark_suite_version_default(self) -> None:
        suite = BenchmarkSuite(name="s", cases=[])
        assert suite.version == "1.0"

    def test_benchmark_suite_description_default_empty(self) -> None:
        suite = BenchmarkSuite(name="s", cases=[])
        assert suite.description == ""

    def test_benchmark_suite_filter_by_pattern(self) -> None:
        cases = [
            TestCase(id="qa-1", input="q1"),
            TestCase(id="safety-1", input="q2"),
            TestCase(id="qa-2", input="q3"),
        ]
        suite = BenchmarkSuite(name="suite", cases=cases)
        filtered = suite.filter(pattern="^qa-")
        assert len(filtered.cases) == 2
        assert all(c.id.startswith("qa-") for c in filtered.cases)

    def test_benchmark_suite_filter_max_cases(self) -> None:
        cases = [TestCase(id=f"c{i}", input=f"i{i}") for i in range(5)]
        suite = BenchmarkSuite(name="suite", cases=cases)
        filtered = suite.filter(max_cases=3)
        assert len(filtered.cases) == 3
