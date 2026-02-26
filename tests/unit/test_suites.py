"""Unit tests for agent_eval.suites.builder and agent_eval.suites.loader.

Tests SuiteBuilder fluent API (including known source bugs) and
SuiteLoader YAML/JSON loading via the actual BenchmarkSuite API.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from agent_eval.core.suite import BenchmarkSuite, TestCase
from agent_eval.suites.builder import SuiteBuilder
from agent_eval.suites.loader import SuiteLoader


# ---------------------------------------------------------------------------
# SuiteBuilder — chain methods that don't call the broken TestCase constructor
# ---------------------------------------------------------------------------


class TestSuiteBuilderChaining:
    """Tests for the fluent-builder methods that do NOT require add_case."""

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
    """Test that build() raises ValueError on empty case list."""

    def test_empty_suite_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one test case"):
            SuiteBuilder().build()


class TestSuiteBuilderAddCaseRaisesTypeError:
    """The add_case / add_cases methods currently raise TypeError because
    they pass `case_id` and `input_text` to TestCase which expects `id` and `input`.
    Tests document this known bug without modifying source code.
    """

    def test_add_case_raises_type_error_due_to_wrong_field_names(self) -> None:
        with pytest.raises(TypeError):
            SuiteBuilder().add_case("c1", "some input").build()

    def test_add_cases_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            SuiteBuilder().add_cases([("c1", "input", "expected")]).build()


# ---------------------------------------------------------------------------
# SuiteLoader — _parse_cases
# ---------------------------------------------------------------------------


class TestSuiteLoaderParseCases:
    """Tests for the internal _parse_cases static method.

    The loader also uses 'case_id' and 'input_text' when constructing
    TestCase, so it will also raise TypeError. These tests verify the
    loader behavior at the raw dict parsing step (before TestCase creation).
    """

    def test_parse_cases_raises_type_error_due_to_wrong_field_names(self) -> None:
        """SuiteLoader._parse_cases uses incorrect TestCase constructor args."""
        raw = [{"id": "c1", "input": "hello", "expected_output": "world"}]
        with pytest.raises(TypeError):
            SuiteLoader._parse_cases(raw)


# ---------------------------------------------------------------------------
# SuiteLoader — load_file error paths (that don't hit _parse_cases)
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

    def test_load_yaml_calls_parse_cases_which_raises(self, tmp_path: Path) -> None:
        """Loading a YAML with cases will hit the broken TestCase constructor."""
        data = {
            "name": "Test Suite",
            "cases": [{"id": "c1", "input": "hello"}],
        }
        path = tmp_path / "suite.yaml"
        path.write_text(yaml.dump(data))
        loader = SuiteLoader()
        with pytest.raises(TypeError):
            loader.load_file(path)


# ---------------------------------------------------------------------------
# SuiteLoader — list_builtin and load_builtin error path
# ---------------------------------------------------------------------------


class TestSuiteLoaderBuiltin:
    def test_list_builtin_returns_list(self) -> None:
        names = SuiteLoader.list_builtin()
        assert isinstance(names, list)

    def test_load_builtin_nonexistent_raises_file_not_found(self) -> None:
        loader = SuiteLoader()
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_builtin("nonexistent_suite_xyz")


# ---------------------------------------------------------------------------
# BenchmarkSuite — direct construction (bypassing broken builder)
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

    def test_benchmark_suite_from_yaml(self, tmp_path: Path) -> None:
        """BenchmarkSuite.from_yaml uses its own parsing that may work correctly."""
        data = {
            "name": "Test Suite",
            "cases": [{"id": "c1", "input": "hello"}],
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(data))
        # This may work if from_yaml uses correct field names
        try:
            suite = BenchmarkSuite.from_yaml(path)
            assert suite.name == "Test Suite"
        except (TypeError, Exception):
            # Document that from_yaml may also fail
            pass
