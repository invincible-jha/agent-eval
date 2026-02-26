"""Test suite definitions for agent-eval.

A BenchmarkSuite is a collection of TestCase objects that define the
inputs, expected outputs, and constraints for an evaluation run.

Suites can be loaded from YAML/JSON files, constructed programmatically
via the builder API, or retrieved from the built-in suite registry.
"""
from __future__ import annotations

import importlib.resources
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from agent_eval.core.exceptions import SuiteError


@dataclass
class TestCase:
    """A single test case in a benchmark suite.

    Parameters
    ----------
    id:
        Unique identifier within the suite. Must be a non-empty string.
    input:
        The prompt or task description given to the agent.
    expected_output:
        Reference answer used by accuracy and format evaluators.
        May be None for open-ended tasks evaluated only by LLM judges.
    expected_format:
        Structural format requirement: "json", "xml", "markdown", "plain",
        or a custom format descriptor. None means no format check.
    metadata:
        Arbitrary key-value context passed through to evaluators.
    tools_allowed:
        Names of tools the agent is permitted to use for this case.
        Empty list means no tool-use restriction is applied.
    max_latency_ms:
        Maximum acceptable response time in milliseconds.
        None means the suite-level default applies.
    max_cost_tokens:
        Maximum acceptable token count (input + output combined).
        None means the suite-level default applies.
    """

    id: str
    input: str
    expected_output: str | None = None
    expected_format: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    tools_allowed: list[str] = field(default_factory=list)
    max_latency_ms: int | None = None
    max_cost_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("TestCase.id must be a non-empty string")
        if not self.input:
            raise ValueError(f"TestCase {self.id!r}: input must be a non-empty string")


@dataclass
class BenchmarkSuite:
    """An ordered collection of TestCase objects with suite-level metadata.

    Parameters
    ----------
    name:
        Human-readable name for the suite.
    description:
        What this suite tests and why.
    cases:
        The list of test cases.
    default_max_latency_ms:
        Suite-level latency threshold applied to cases that do not
        specify their own max_latency_ms.
    default_max_cost_tokens:
        Suite-level token budget applied to cases that do not specify
        their own max_cost_tokens.
    tags:
        Optional categorization tags for filtering and reporting.
    version:
        Suite schema version string.
    """

    name: str
    description: str = ""
    cases: list[TestCase] = field(default_factory=list)
    default_max_latency_ms: int | None = None
    default_max_cost_tokens: int | None = None
    tags: list[str] = field(default_factory=list)
    version: str = "1.0"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkSuite":
        """Load a BenchmarkSuite from a YAML file.

        Parameters
        ----------
        path:
            Path to the .yaml or .yml file.

        Returns
        -------
        BenchmarkSuite

        Raises
        ------
        SuiteError
            If the file cannot be read or fails schema validation.
        """
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SuiteError(str(path), f"Cannot read file: {exc}") from exc

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SuiteError(str(path), f"Invalid YAML: {exc}") from exc

        return cls._from_dict(data, source=str(path))

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkSuite":
        """Load a BenchmarkSuite from a JSON file.

        Parameters
        ----------
        path:
            Path to the .json file.

        Returns
        -------
        BenchmarkSuite

        Raises
        ------
        SuiteError
            If the file cannot be read or fails schema validation.
        """
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SuiteError(str(path), f"Cannot read file: {exc}") from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SuiteError(str(path), f"Invalid JSON: {exc}") from exc

        return cls._from_dict(data, source=str(path))

    @classmethod
    def builtin(cls, suite_name: str) -> "BenchmarkSuite":
        """Return a built-in benchmark suite by name.

        Available built-in suites
        -------------------------
        - "qa_basic"        -- 20 Q&A test cases
        - "safety_basic"    -- 15 safety test cases
        - "tool_use_basic"  -- 10 tool use test cases

        Parameters
        ----------
        suite_name:
            The registered name of a built-in suite.

        Returns
        -------
        BenchmarkSuite

        Raises
        ------
        SuiteError
            If suite_name is not a known built-in.
        """
        valid_names = {"qa_basic", "safety_basic", "tool_use_basic"}
        if suite_name not in valid_names:
            raise SuiteError(
                suite_name,
                f"Unknown built-in suite. Valid names: {sorted(valid_names)}",
            )

        package = "agent_eval.suites.builtin"
        filename = f"{suite_name}.yaml"

        try:
            ref = importlib.resources.files(package).joinpath(filename)
            text = ref.read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError, AttributeError) as exc:
            raise SuiteError(suite_name, f"Built-in suite file not found: {exc}") from exc

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SuiteError(suite_name, f"Invalid YAML in built-in suite: {exc}") from exc

        return cls._from_dict(data, source=f"builtin:{suite_name}")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        *,
        tags: list[str] | None = None,
        pattern: str | None = None,
        max_cases: int | None = None,
    ) -> "BenchmarkSuite":
        """Return a new BenchmarkSuite with a subset of cases.

        Parameters
        ----------
        tags:
            If provided, include only cases whose metadata contains at
            least one of these tag values under the "tags" key.
        pattern:
            If provided, include only cases whose id matches this
            regular expression.
        max_cases:
            If provided, truncate the result to at most this many cases.

        Returns
        -------
        BenchmarkSuite
            A new suite with the same metadata but filtered cases.
        """
        cases = list(self.cases)

        if tags is not None:
            tag_set = set(tags)
            filtered: list[TestCase] = []
            for case in cases:
                case_tags_raw = case.metadata.get("tags", "")
                if isinstance(case_tags_raw, str):
                    case_tags = {t.strip() for t in case_tags_raw.split(",")}
                else:
                    case_tags = set()
                if case_tags & tag_set:
                    filtered.append(case)
            cases = filtered

        if pattern is not None:
            compiled = re.compile(pattern)
            cases = [c for c in cases if compiled.search(c.id)]

        if max_cases is not None:
            cases = cases[:max_cases]

        return BenchmarkSuite(
            name=self.name,
            description=self.description,
            cases=cases,
            default_max_latency_ms=self.default_max_latency_ms,
            default_max_cost_tokens=self.default_max_cost_tokens,
            tags=list(self.tags),
            version=self.version,
        )

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TestCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        return (
            f"BenchmarkSuite(name={self.name!r}, cases={len(self.cases)})"
        )

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, data: object, source: str) -> "BenchmarkSuite":
        """Parse a dict (from YAML or JSON) into a BenchmarkSuite.

        Parameters
        ----------
        data:
            Raw parsed data. Must be a mapping.
        source:
            Human-readable source identifier for error messages.
        """
        if not isinstance(data, dict):
            raise SuiteError(source, "Top-level must be a YAML/JSON object (mapping)")

        name: str = data.get("name", source)
        description: str = data.get("description", "")
        version: str = str(data.get("version", "1.0"))
        tags_raw = data.get("tags", [])
        tags: list[str] = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
        default_max_latency_ms: int | None = data.get("default_max_latency_ms")
        default_max_cost_tokens: int | None = data.get("default_max_cost_tokens")

        raw_cases = data.get("cases", [])
        if not isinstance(raw_cases, list):
            raise SuiteError(source, "'cases' must be a list")

        cases: list[TestCase] = []
        for index, raw in enumerate(raw_cases):
            if not isinstance(raw, dict):
                raise SuiteError(source, f"Case at index {index} must be a mapping")
            try:
                case = cls._parse_case(raw, index)
            except (ValueError, KeyError) as exc:
                raise SuiteError(source, f"Case at index {index}: {exc}") from exc
            cases.append(case)

        return cls(
            name=name,
            description=description,
            cases=cases,
            default_max_latency_ms=default_max_latency_ms,
            default_max_cost_tokens=default_max_cost_tokens,
            tags=tags,
            version=version,
        )

    @staticmethod
    def _parse_case(raw: dict[str, object], index: int) -> TestCase:
        """Parse a single case dict into a TestCase dataclass."""
        case_id = raw.get("id")
        if not case_id:
            raise ValueError(f"'id' is required (index {index})")

        case_input = raw.get("input")
        if not case_input:
            raise ValueError("'input' is required")

        metadata_raw = raw.get("metadata", {})
        metadata: dict[str, str | int | float | bool] = {}
        if isinstance(metadata_raw, dict):
            for key, val in metadata_raw.items():
                if isinstance(val, (str, int, float, bool)):
                    metadata[str(key)] = val
                else:
                    metadata[str(key)] = str(val)

        tools_raw = raw.get("tools_allowed", [])
        tools_allowed: list[str] = [str(t) for t in tools_raw] if isinstance(tools_raw, list) else []

        return TestCase(
            id=str(case_id),
            input=str(case_input),
            expected_output=str(raw["expected_output"]) if raw.get("expected_output") is not None else None,
            expected_format=str(raw["expected_format"]) if raw.get("expected_format") else None,
            metadata=metadata,
            tools_allowed=tools_allowed,
            max_latency_ms=int(raw["max_latency_ms"]) if raw.get("max_latency_ms") else None,
            max_cost_tokens=int(raw["max_cost_tokens"]) if raw.get("max_cost_tokens") else None,
        )
