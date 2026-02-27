"""Fluent API for programmatic test suite construction.

SuiteBuilder provides a builder pattern for constructing BenchmarkSuite
objects without requiring YAML files.
"""
from __future__ import annotations

from agent_eval.core.suite import BenchmarkSuite, TestCase


class SuiteBuilder:
    """Build a BenchmarkSuite programmatically.

    Examples
    --------
    ::

        suite = (
            SuiteBuilder()
            .name("my-suite")
            .description("Custom evaluation suite")
            .add_case("q1", "What is 2+2?", expected="4")
            .add_case("q2", "Capital of France?", expected="Paris")
            .build()
        )
    """

    def __init__(self) -> None:
        self._name: str = "unnamed-suite"
        self._description: str = ""
        self._version: str = "1.0"
        self._cases: list[TestCase] = []

    def name(self, name: str) -> SuiteBuilder:
        """Set the suite name."""
        self._name = name
        return self

    def description(self, description: str) -> SuiteBuilder:
        """Set the suite description."""
        self._description = description
        return self

    def version(self, version: str) -> SuiteBuilder:
        """Set the suite version."""
        self._version = version
        return self

    def add_case(
        self,
        case_id: str,
        input_text: str,
        expected: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        tags: list[str] | None = None,
    ) -> SuiteBuilder:
        """Add a single test case.

        Parameters
        ----------
        case_id:
            Unique identifier for the test case.
        input_text:
            Input prompt for the agent.
        expected:
            Expected output (if applicable).
        metadata:
            Arbitrary metadata for evaluators.
        tags:
            Tags for filtering.
        """
        self._cases.append(
            TestCase(
                id=case_id,
                input=input_text,
                expected_output=expected,
                metadata=metadata or {},
            )
        )
        return self

    def add_cases(
        self,
        cases: list[tuple[str, str, str | None]],
    ) -> SuiteBuilder:
        """Add multiple test cases as (id, input, expected) tuples."""
        for case_id, input_text, expected in cases:
            self.add_case(case_id, input_text, expected)
        return self

    def build(self) -> BenchmarkSuite:
        """Build and validate the suite.

        Returns
        -------
        BenchmarkSuite
            The constructed suite.

        Raises
        ------
        ValueError
            If the suite has no cases or has duplicate IDs.
        """
        if not self._cases:
            raise ValueError("Suite must have at least one test case.")

        ids = [c.id for c in self._cases]
        duplicates = [cid for cid in ids if ids.count(cid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate case IDs: {sorted(set(duplicates))}")

        return BenchmarkSuite(
            name=self._name,
            description=self._description,
            version=self._version,
            cases=list(self._cases),
        )
