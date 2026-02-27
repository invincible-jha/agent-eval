"""Tests for agent_eval.integrations.deepeval_adapter.

DeepEval is mocked — it is not required in CI.
"""
from __future__ import annotations

import sys
import types
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixture: inject a fake deepeval module
# ---------------------------------------------------------------------------


class FakeLLMTestCase:
    """Minimal stand-in for deepeval.test_case.LLMTestCase."""

    def __init__(
        self,
        input: str = "",
        expected_output: Optional[str] = None,
        context: Optional[list[str]] = None,
        id: Optional[str] = None,
    ) -> None:
        self.input = input
        self.expected_output = expected_output
        self.context = context or []
        self.id = id


class FakeDataset:
    """Minimal stand-in for deepeval.dataset.EvaluationDataset."""

    def __init__(self, test_cases: list[FakeLLMTestCase]) -> None:
        self.test_cases = test_cases


def _make_mock_deepeval_module() -> types.ModuleType:
    mod = types.ModuleType("deepeval")
    test_case_mod = types.ModuleType("deepeval.test_case")
    test_case_mod.LLMTestCase = FakeLLMTestCase  # type: ignore[attr-defined]

    mod.test_case = test_case_mod  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def inject_deepeval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake deepeval before each test."""
    fake_deepeval = _make_mock_deepeval_module()
    test_case_mod = fake_deepeval.test_case
    monkeypatch.setitem(sys.modules, "deepeval", fake_deepeval)
    monkeypatch.setitem(sys.modules, "deepeval.test_case", test_case_mod)
    monkeypatch.delitem(
        sys.modules,
        "agent_eval.integrations.deepeval_adapter",
        raising=False,
    )


def _import_adapter() -> Any:
    from agent_eval.integrations import deepeval_adapter  # noqa: PLC0415

    return deepeval_adapter


def _make_importer(**kwargs: Any) -> Any:
    adapter = _import_adapter()
    return adapter.DeepEvalImporter(**kwargs)


def _make_case(
    input_text: str = "What is 2+2?",
    expected_output: str = "4",
    context: Optional[list[str]] = None,
    case_id: Optional[str] = None,
) -> FakeLLMTestCase:
    return FakeLLMTestCase(
        input=input_text,
        expected_output=expected_output,
        context=context or [],
        id=case_id,
    )


def _simple_agent(text: str) -> str:
    """Deterministic agent that always returns '4'."""
    return "4"


def _failing_agent(text: str) -> str:
    return "completely wrong answer with no overlap"


# ---------------------------------------------------------------------------
# DeepEvalImporter — construction
# ---------------------------------------------------------------------------


class TestDeepEvalImporterConstruction:
    def test_default_construction(self) -> None:
        importer = _make_importer()
        assert importer is not None

    def test_pass_threshold_stored(self) -> None:
        importer = _make_importer(pass_threshold=0.5)
        assert importer._pass_threshold == 0.5

    def test_invalid_pass_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_importer(pass_threshold=1.5)

    def test_custom_pass_checker_accepted(self) -> None:
        checker = lambda expected, actual: expected == actual  # noqa: E731
        importer = _make_importer(pass_checker=checker)
        assert importer._pass_checker is checker

    def test_repr_contains_class_name(self) -> None:
        importer = _make_importer()
        assert "DeepEvalImporter" in repr(importer)


# ---------------------------------------------------------------------------
# import_test_cases — list input
# ---------------------------------------------------------------------------


class TestImportTestCases:
    def test_imports_list_of_llm_test_cases(self) -> None:
        importer = _make_importer()
        cases_raw = [_make_case("Q1", "A1"), _make_case("Q2", "A2")]
        cases = importer.import_test_cases(cases_raw)
        assert len(cases) == 2

    def test_input_text_mapped(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Hello")])
        assert cases[0].input_text == "Hello"

    def test_expected_output_mapped(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case(expected_output="42")])
        assert cases[0].expected_output == "42"

    def test_none_expected_output_preserved(self) -> None:
        importer = _make_importer()
        raw = FakeLLMTestCase(input="Q", expected_output=None)
        cases = importer.import_test_cases([raw])
        assert cases[0].expected_output is None

    def test_context_mapped(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases(
            [_make_case(context=["doc1", "doc2"])]
        )
        assert cases[0].context == ["doc1", "doc2"]

    def test_case_id_from_raw_id(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case(case_id="my-case-001")])
        assert cases[0].case_id == "my-case-001"

    def test_case_id_auto_generated_when_no_id(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case()])
        assert cases[0].case_id.startswith("case-")

    def test_dataset_with_test_cases_attr(self) -> None:
        importer = _make_importer()
        dataset = FakeDataset([_make_case("Q1"), _make_case("Q2")])
        cases = importer.import_test_cases(dataset)
        assert len(cases) == 2

    def test_invalid_type_raises_type_error(self) -> None:
        importer = _make_importer()
        with pytest.raises(TypeError):
            importer.import_test_cases(["not_a_test_case"])  # type: ignore[arg-type]

    def test_empty_list_returns_empty(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([])
        assert cases == []


# ---------------------------------------------------------------------------
# import_single_case
# ---------------------------------------------------------------------------


class TestImportSingleCase:
    def test_converts_single_case(self) -> None:
        importer = _make_importer()
        raw = _make_case("Hello", "World")
        case = importer.import_single_case(raw)
        assert case.input_text == "Hello"
        assert case.expected_output == "World"

    def test_custom_case_id_override(self) -> None:
        importer = _make_importer()
        raw = _make_case()
        case = importer.import_single_case(raw, case_id="override-id")
        assert case.case_id == "override-id"

    def test_invalid_type_raises_type_error(self) -> None:
        importer = _make_importer()
        with pytest.raises(TypeError):
            importer.import_single_case("not_a_case")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_with_statistics — positive path
# ---------------------------------------------------------------------------


class TestRunWithStatistics:
    def test_returns_statistical_result(self) -> None:
        adapter = _import_adapter()
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=5, k=3)
        assert isinstance(result, adapter.StatisticalResult)

    def test_total_runs_equals_cases_times_n(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases(
            [_make_case("Q1"), _make_case("Q2")]
        )
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=4, k=2)
        assert result.total_runs == 8

    def test_pass_at_k_in_range(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=5, k=3)
        assert 0.0 <= result.pass_at_k_value <= 1.0

    def test_successful_runs_counted(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=5, k=3)
        assert result.successful_runs == 5  # agent always returns "4"

    def test_failing_agent_zero_successes(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _failing_agent, n_runs=5, k=3)
        assert result.successful_runs == 0

    def test_run_results_length(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=3, k=1)
        assert len(result.run_results) == 3

    def test_mean_latency_non_negative(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=2, k=1)
        assert result.mean_latency_ms >= 0.0

    def test_confidence_interval_bounds(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=5, k=3)
        ci = result.confidence_interval_result
        assert 0.0 <= ci.lower <= ci.upper <= 1.0

    def test_variance_non_negative(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=5, k=1)
        assert result.variance >= 0.0

    def test_agent_exception_recorded_as_error(self) -> None:
        def raising_agent(text: str) -> str:
            raise RuntimeError("agent exploded")

        importer = _make_importer()
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, raising_agent, n_runs=2, k=1)
        errors = [r for r in result.run_results if r.error is not None]
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# run_with_statistics — validation errors
# ---------------------------------------------------------------------------


class TestRunWithStatisticsValidation:
    def test_empty_cases_raises(self) -> None:
        importer = _make_importer()
        with pytest.raises(ValueError, match="cases must not be empty"):
            importer.run_with_statistics([], _simple_agent, n_runs=5, k=3)

    def test_zero_n_runs_raises(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case()])
        with pytest.raises(ValueError):
            importer.run_with_statistics(cases, _simple_agent, n_runs=0, k=1)

    def test_zero_k_raises(self) -> None:
        importer = _make_importer()
        cases = importer.import_test_cases([_make_case()])
        with pytest.raises(ValueError):
            importer.run_with_statistics(cases, _simple_agent, n_runs=5, k=0)


# ---------------------------------------------------------------------------
# Custom pass_checker
# ---------------------------------------------------------------------------


class TestCustomPassChecker:
    def test_custom_checker_used(self) -> None:
        # Checker that always returns True
        importer = _make_importer(pass_checker=lambda e, a: True)
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _failing_agent, n_runs=3, k=1)
        assert result.successful_runs == 3

    def test_custom_checker_always_false(self) -> None:
        importer = _make_importer(pass_checker=lambda e, a: False)
        cases = importer.import_test_cases([_make_case("Q", "4")])
        result = importer.run_with_statistics(cases, _simple_agent, n_runs=3, k=1)
        assert result.successful_runs == 0


# ---------------------------------------------------------------------------
# EvalCase dataclass
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_eval_case_fields(self) -> None:
        adapter = _import_adapter()
        case = adapter.EvalCase(
            case_id="c1",
            input_text="Hello",
            expected_output="World",
        )
        assert case.case_id == "c1"
        assert case.input_text == "Hello"
        assert case.expected_output == "World"
        assert case.context == []
        assert case.metadata == {}
