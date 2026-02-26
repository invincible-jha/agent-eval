"""Suite loader for YAML and JSON test suite files.

Loads BenchmarkSuite definitions from structured YAML or JSON files,
validates them, and provides discovery of built-in suites.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agent_eval.core.suite import BenchmarkSuite, TestCase


class SuiteFileSchema(BaseModel):
    """Pydantic model for validating suite files."""

    name: str
    description: str = ""
    version: str = "1.0"
    tags: list[str] = Field(default_factory=list)
    cases: list[dict[str, str | dict[str, str | int | float | bool] | list[str]]]

    model_config = {"arbitrary_types_allowed": False}


class SuiteLoader:
    """Loads BenchmarkSuite instances from YAML/JSON files.

    Examples
    --------
    ::

        loader = SuiteLoader()
        suite = loader.load_file("tests/my_suite.yaml")

        # Load all suites in a directory
        suites = loader.load_directory("tests/suites/")

        # Load a built-in suite
        suite = loader.load_builtin("qa_basic")
    """

    @staticmethod
    def _parse_cases(
        raw_cases: list[dict[str, str | dict[str, str | int | float | bool] | list[str]]],
    ) -> list[TestCase]:
        """Convert raw case dicts to TestCase objects."""
        cases: list[TestCase] = []
        for raw in raw_cases:
            case_id = str(raw.get("id", f"case_{len(cases)}"))
            input_text = str(raw.get("input", ""))
            expected = raw.get("expected_output")
            expected_str = str(expected) if expected is not None else None

            raw_metadata = raw.get("metadata", {})
            metadata: dict[str, str | int | float | bool] = {}
            if isinstance(raw_metadata, dict):
                for k, v in raw_metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[str(k)] = v

            raw_tags = raw.get("tags", [])
            tags: list[str] = []
            if isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags]

            cases.append(
                TestCase(
                    case_id=case_id,
                    input_text=input_text,
                    expected_output=expected_str,
                    metadata=metadata,
                    tags=tags,
                )
            )
        return cases

    def load_file(self, path: str | Path) -> BenchmarkSuite:
        """Load a suite from a YAML or JSON file.

        Parameters
        ----------
        path:
            Path to the suite file (.yaml, .yml, or .json).

        Returns
        -------
        BenchmarkSuite
            The loaded and validated suite.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If the file format is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Suite file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")

        if file_path.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(content)
        elif file_path.suffix == ".json":
            raw = json.loads(content)
        else:
            raise ValueError(f"Unsupported suite file format: {file_path.suffix}")

        if not isinstance(raw, dict):
            raise ValueError(f"Suite file must contain a YAML/JSON object, got {type(raw).__name__}")

        schema = SuiteFileSchema(**raw)
        cases = self._parse_cases(schema.cases)

        return BenchmarkSuite(
            name=schema.name,
            description=schema.description,
            version=schema.version,
            cases=cases,
        )

    def load_directory(self, path: str | Path) -> list[BenchmarkSuite]:
        """Load all suite files in a directory.

        Parameters
        ----------
        path:
            Directory containing .yaml/.yml/.json suite files.

        Returns
        -------
        list[BenchmarkSuite]
            All successfully loaded suites.
        """
        dir_path = Path(path)
        suites: list[BenchmarkSuite] = []

        for ext in ("*.yaml", "*.yml", "*.json"):
            for file_path in sorted(dir_path.glob(ext)):
                suites.append(self.load_file(file_path))

        return suites

    def load_builtin(self, name: str) -> BenchmarkSuite:
        """Load a built-in suite by name.

        Parameters
        ----------
        name:
            Built-in suite name (without extension). Available:
            ``qa_basic``, ``safety_basic``, ``tool_use_basic``.

        Returns
        -------
        BenchmarkSuite
            The built-in suite.
        """
        builtin_dir = Path(__file__).parent / "builtin"
        candidates = [
            builtin_dir / f"{name}.yaml",
            builtin_dir / f"{name}.yml",
            builtin_dir / f"{name}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return self.load_file(candidate)

        available = self.list_builtin()
        raise FileNotFoundError(
            f"Built-in suite '{name}' not found. Available: {available}"
        )

    @staticmethod
    def list_builtin() -> list[str]:
        """List names of all available built-in suites."""
        builtin_dir = Path(__file__).parent / "builtin"
        if not builtin_dir.exists():
            return []
        names: list[str] = []
        for f in sorted(builtin_dir.iterdir()):
            if f.suffix in (".yaml", ".yml", ".json") and not f.name.startswith("_"):
                names.append(f.stem)
        return names
