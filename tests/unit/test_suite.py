"""Unit tests for agent_eval.core.suite.

Tests TestCase validation, BenchmarkSuite construction, factory methods
(from_yaml, from_json, builtin), filter(), and iteration helpers.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from agent_eval.core.exceptions import SuiteError
from agent_eval.core.suite import BenchmarkSuite, TestCase


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------


class TestTestCase:
    def test_minimal_valid_case(self) -> None:
        case = TestCase(id="c1", input="What is 2+2?")
        assert case.id == "c1"
        assert case.input == "What is 2+2?"
        assert case.expected_output is None

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="id must be a non-empty string"):
            TestCase(id="", input="some input")

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="input must be a non-empty string"):
            TestCase(id="c1", input="")

    def test_optional_fields_default(self) -> None:
        case = TestCase(id="c2", input="prompt")
        assert case.expected_format is None
        assert case.metadata == {}
        assert case.tools_allowed == []
        assert case.max_latency_ms is None
        assert case.max_cost_tokens is None

    def test_full_construction(self) -> None:
        case = TestCase(
            id="c3",
            input="query",
            expected_output="answer",
            expected_format="json",
            metadata={"env": "test", "priority": 1},
            tools_allowed=["search", "calculator"],
            max_latency_ms=3000,
            max_cost_tokens=512,
        )
        assert case.expected_output == "answer"
        assert case.expected_format == "json"
        assert case.metadata["env"] == "test"
        assert "search" in case.tools_allowed
        assert case.max_latency_ms == 3000
        assert case.max_cost_tokens == 512


# ---------------------------------------------------------------------------
# BenchmarkSuite construction
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteConstruction:
    def _make_suite(self, num_cases: int = 2) -> BenchmarkSuite:
        cases = [TestCase(id=f"c{i}", input=f"input {i}") for i in range(num_cases)]
        return BenchmarkSuite(name="test-suite", cases=cases)

    def test_name_stored(self) -> None:
        suite = self._make_suite()
        assert suite.name == "test-suite"

    def test_len_reflects_case_count(self) -> None:
        suite = self._make_suite(5)
        assert len(suite) == 5

    def test_iter_yields_test_cases(self) -> None:
        suite = self._make_suite(3)
        ids = [c.id for c in suite]
        assert ids == ["c0", "c1", "c2"]

    def test_repr_shows_name_and_count(self) -> None:
        suite = self._make_suite(4)
        r = repr(suite)
        assert "test-suite" in r
        assert "4" in r

    def test_default_fields(self) -> None:
        suite = BenchmarkSuite(name="s")
        assert suite.description == ""
        assert suite.cases == []
        assert suite.default_max_latency_ms is None
        assert suite.default_max_cost_tokens is None
        assert suite.tags == []
        assert suite.version == "1.0"


# ---------------------------------------------------------------------------
# BenchmarkSuite.filter()
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteFilter:
    def _make_tagged_suite(self) -> BenchmarkSuite:
        cases = [
            TestCase(id="c1", input="i1", metadata={"tags": "fast, smoke"}),
            TestCase(id="c2", input="i2", metadata={"tags": "slow, regression"}),
            TestCase(id="c3", input="i3", metadata={"tags": "fast"}),
            TestCase(id="c4", input="i4"),
        ]
        return BenchmarkSuite(name="tagged", cases=cases)

    def test_filter_by_tag_returns_matching(self) -> None:
        suite = self._make_tagged_suite()
        filtered = suite.filter(tags=["fast"])
        ids = [c.id for c in filtered]
        assert "c1" in ids
        assert "c3" in ids
        assert "c2" not in ids

    def test_filter_by_pattern_matches_regex(self) -> None:
        suite = self._make_tagged_suite()
        filtered = suite.filter(pattern=r"c[12]")
        assert len(filtered) == 2

    def test_filter_max_cases_truncates(self) -> None:
        suite = self._make_tagged_suite()
        filtered = suite.filter(max_cases=2)
        assert len(filtered) == 2

    def test_filter_returns_new_suite_preserving_metadata(self) -> None:
        suite = self._make_tagged_suite()
        filtered = suite.filter(max_cases=1)
        assert filtered.name == suite.name
        assert filtered.version == suite.version

    def test_filter_no_args_returns_all(self) -> None:
        suite = self._make_tagged_suite()
        filtered = suite.filter()
        assert len(filtered) == len(suite)

    def test_filter_pattern_no_match_returns_empty(self) -> None:
        suite = self._make_tagged_suite()
        filtered = suite.filter(pattern=r"^NOMATCH")
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# BenchmarkSuite.from_yaml()
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteFromYaml:
    def test_loads_minimal_yaml(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            name: simple-suite
            cases:
              - id: q1
                input: What is 2+2?
                expected_output: "4"
        """)
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(content, encoding="utf-8")
        suite = BenchmarkSuite.from_yaml(suite_file)
        assert suite.name == "simple-suite"
        assert len(suite) == 1
        assert suite.cases[0].id == "q1"

    def test_loads_suite_with_all_fields(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            name: full-suite
            description: Full test
            version: "2.0"
            tags: [qa, smoke]
            default_max_latency_ms: 5000
            default_max_cost_tokens: 1024
            cases:
              - id: q1
                input: Question?
                expected_output: Answer
                expected_format: plain
                max_latency_ms: 3000
                max_cost_tokens: 256
        """)
        suite_file = tmp_path / "full.yaml"
        suite_file.write_text(content, encoding="utf-8")
        suite = BenchmarkSuite.from_yaml(suite_file)
        assert suite.version == "2.0"
        assert suite.default_max_latency_ms == 5000
        assert suite.cases[0].max_latency_ms == 3000

    def test_missing_file_raises_suite_error(self) -> None:
        with pytest.raises(SuiteError):
            BenchmarkSuite.from_yaml("/nonexistent/path/suite.yaml")

    def test_invalid_yaml_raises_suite_error(self, tmp_path: Path) -> None:
        suite_file = tmp_path / "bad.yaml"
        suite_file.write_text("cases: [unclosed: {yaml", encoding="utf-8")
        with pytest.raises(SuiteError):
            BenchmarkSuite.from_yaml(suite_file)

    def test_non_mapping_yaml_raises_suite_error(self, tmp_path: Path) -> None:
        suite_file = tmp_path / "list.yaml"
        suite_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(SuiteError):
            BenchmarkSuite.from_yaml(suite_file)

    def test_case_without_id_raises_suite_error(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            name: broken
            cases:
              - input: Something?
        """)
        suite_file = tmp_path / "broken.yaml"
        suite_file.write_text(content, encoding="utf-8")
        with pytest.raises(SuiteError):
            BenchmarkSuite.from_yaml(suite_file)


# ---------------------------------------------------------------------------
# BenchmarkSuite.from_json()
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteFromJson:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        data = {
            "name": "json-suite",
            "cases": [
                {"id": "j1", "input": "Hello?", "expected_output": "Hi!"}
            ],
        }
        suite_file = tmp_path / "suite.json"
        suite_file.write_text(json.dumps(data), encoding="utf-8")
        suite = BenchmarkSuite.from_json(suite_file)
        assert suite.name == "json-suite"
        assert suite.cases[0].expected_output == "Hi!"

    def test_missing_file_raises_suite_error(self) -> None:
        with pytest.raises(SuiteError):
            BenchmarkSuite.from_json("/no/such/file.json")

    def test_invalid_json_raises_suite_error(self, tmp_path: Path) -> None:
        suite_file = tmp_path / "bad.json"
        suite_file.write_text("{broken json", encoding="utf-8")
        with pytest.raises(SuiteError):
            BenchmarkSuite.from_json(suite_file)


# ---------------------------------------------------------------------------
# BenchmarkSuite.builtin()
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteBuiltin:
    def test_unknown_builtin_raises_suite_error(self) -> None:
        with pytest.raises(SuiteError):
            BenchmarkSuite.builtin("nonexistent_suite_xyz")

    @pytest.mark.parametrize("suite_name", ["qa_basic", "safety_basic", "tool_use_basic"])
    def test_known_builtin_returns_suite(self, suite_name: str) -> None:
        suite = BenchmarkSuite.builtin(suite_name)
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite) > 0
