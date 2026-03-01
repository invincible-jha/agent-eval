"""Tests for agent_eval.pytest_plugin.scaffold — EvalScaffoldGenerator.

Coverage targets (15+ tests)
-----------------------------
- EvalScaffoldGenerator.generate_from_dict: valid spec, missing behaviors key,
  empty behaviors list, single behavior, multiple behaviors, custom dimensions,
  custom threshold, no expected_intent, generated content structure
- EvalScaffoldGenerator.generate_from_bsl: file-not-found error, BSL file with
  behavior blocks, BSL file with no behavior blocks raises ValueError
- EvalScaffoldGenerator.write_to_file: writes content to path, creates parent
  dirs, returns resolved Path
- _to_safe_identifier: various name formats
- _render_dimensions_list: list input, tuple input, non-list input
- _render_positive_test / _render_negative_test: non-list dimensions branch,
  non-numeric threshold branch
- _parse_bsl_with_library: mock bsl_lang available path and exception fallback
- Package __init__ re-export of EvalScaffoldGenerator
"""
from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_eval.pytest_plugin.scaffold import (
    EvalScaffoldGenerator,
    _extract_behaviors_from_bsl_text,
    _render_dimensions_list,
    _render_negative_test,
    _render_positive_test,
    _to_safe_identifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_SPEC: dict[str, object] = {
    "agent_name": "test_agent",
    "behaviors": [
        {
            "name": "capital_city_query",
            "given": "A user asks for the capital of France",
            "when": "The agent processes the query",
            "then": "The response contains 'Paris'",
            "expected_intent": "Paris",
            "dimensions": ["accuracy", "safety"],
            "threshold": 0.85,
        }
    ],
}

_MULTI_BEHAVIOR_SPEC: dict[str, object] = {
    "agent_name": "multi_agent",
    "behaviors": [
        {
            "name": "greeting",
            "given": "A user says hello",
            "when": "The agent receives the greeting",
            "then": "The response is a friendly greeting",
            "expected_intent": "hello",
            "dimensions": ["accuracy", "safety"],
            "threshold": 0.8,
        },
        {
            "name": "farewell",
            "given": "A user says goodbye",
            "when": "The agent receives the farewell",
            "then": "The response is a polite goodbye",
            "expected_intent": "goodbye",
            "dimensions": ["accuracy"],
            "threshold": 0.75,
        },
    ],
}

_BSL_TEXT_VALID = textwrap.dedent(
    """\
    behavior capital_query {
        given: "User asks for the capital city"
        when: "Agent processes the request"
        then: "Response contains 'Paris'"
    }

    behavior safety_check {
        given: "User sends an unsafe prompt"
        when: "Agent evaluates the content"
        then: "Response does not contain harmful content"
    }
    """
)

_BSL_TEXT_NO_BEHAVIORS = "// This BSL file has no behavior blocks.\nagent_name: test_agent\n"


def _make_generator() -> EvalScaffoldGenerator:
    return EvalScaffoldGenerator()


# ---------------------------------------------------------------------------
# _to_safe_identifier
# ---------------------------------------------------------------------------


class TestToSafeIdentifier:
    def test_simple_name_unchanged(self) -> None:
        assert _to_safe_identifier("capital") == "capital"

    def test_spaces_replaced_with_underscore(self) -> None:
        assert _to_safe_identifier("capital city query") == "capital_city_query"

    def test_special_chars_replaced(self) -> None:
        result = _to_safe_identifier("my-behavior!v2")
        assert "my" in result
        assert "behavior" in result
        assert "-" not in result
        assert "!" not in result

    def test_mixed_case_lowercased(self) -> None:
        assert _to_safe_identifier("CapitalCity") == "capitalcity"

    def test_empty_string_returns_behavior(self) -> None:
        assert _to_safe_identifier("") == "behavior"

    def test_consecutive_special_chars_collapsed(self) -> None:
        result = _to_safe_identifier("a--b")
        assert "__" not in result
        assert result == "a_b"


# ---------------------------------------------------------------------------
# _render_dimensions_list
# ---------------------------------------------------------------------------


class TestRenderDimensionsList:
    def test_list_input_produces_quoted_items(self) -> None:
        result = _render_dimensions_list(["accuracy", "safety"])
        assert '"accuracy"' in result
        assert '"safety"' in result

    def test_tuple_input_accepted(self) -> None:
        result = _render_dimensions_list(("accuracy",))
        assert '"accuracy"' in result

    def test_non_list_falls_back_to_defaults(self) -> None:
        result = _render_dimensions_list(None)  # type: ignore[arg-type]
        assert '"accuracy"' in result
        assert '"safety"' in result
        assert '"cost"' in result
        assert '"latency"' in result

    def test_output_is_valid_list_literal(self) -> None:
        result = _render_dimensions_list(["accuracy"])
        assert result.startswith("[")
        assert result.endswith("]")


# ---------------------------------------------------------------------------
# _extract_behaviors_from_bsl_text (fallback regex parser)
# ---------------------------------------------------------------------------


class TestExtractBehaviorsFromBslText:
    def test_extracts_single_behavior(self) -> None:
        behaviors = _extract_behaviors_from_bsl_text(_BSL_TEXT_VALID)
        assert len(behaviors) >= 1

    def test_extracts_correct_name(self) -> None:
        behaviors = _extract_behaviors_from_bsl_text(_BSL_TEXT_VALID)
        names = [str(b["name"]) for b in behaviors]
        assert "capital_query" in names

    def test_extracts_given_clause(self) -> None:
        behaviors = _extract_behaviors_from_bsl_text(_BSL_TEXT_VALID)
        # Collect all given values across all extracted behaviors
        all_given = " ".join(str(b["given"]).lower() for b in behaviors)
        assert "capital" in all_given or "user" in all_given

    def test_returns_empty_list_for_no_behaviors(self) -> None:
        behaviors = _extract_behaviors_from_bsl_text(_BSL_TEXT_NO_BEHAVIORS)
        assert behaviors == []

    def test_extracts_multiple_behaviors(self) -> None:
        behaviors = _extract_behaviors_from_bsl_text(_BSL_TEXT_VALID)
        assert len(behaviors) == 2

    def test_behavior_has_expected_intent(self) -> None:
        behaviors = _extract_behaviors_from_bsl_text(_BSL_TEXT_VALID)
        first = behaviors[0]
        # expected_intent derived from quoted word in "then" clause
        assert "expected_intent" in first
        assert isinstance(first["expected_intent"], str)


# ---------------------------------------------------------------------------
# EvalScaffoldGenerator.generate_from_dict
# ---------------------------------------------------------------------------


class TestGenerateFromDict:
    def test_returns_string(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert isinstance(result, str)

    def test_output_contains_module_docstring(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert '"""' in result

    def test_output_contains_agent_name(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "test_agent" in result

    def test_output_contains_imports(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "import pytest" in result
        assert "from agent_eval.pytest_plugin import EvalContext" in result

    def test_positive_test_function_generated(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "def test_capital_city_query_positive" in result

    def test_negative_test_function_generated(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "def test_capital_city_query_negative" in result

    def test_agent_eval_marker_present(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "@pytest.mark.agent_eval" in result

    def test_expected_intent_in_positive_test(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "Paris" in result

    def test_threshold_in_marker(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "0.85" in result

    def test_multiple_behaviors_produce_multiple_functions(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MULTI_BEHAVIOR_SPEC)
        assert "def test_greeting_positive" in result
        assert "def test_farewell_positive" in result
        assert "def test_greeting_negative" in result
        assert "def test_farewell_negative" in result

    def test_eval_context_fixture_in_signature(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "eval_context: EvalContext" in result

    def test_assert_accuracy_call_in_positive_test(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "eval_context.assert_accuracy" in result

    def test_assert_safety_call_in_positive_test_when_in_dimensions(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        # "safety" is in dimensions for the minimal spec
        assert "eval_context.assert_safety" in result

    def test_negative_test_uses_empty_response(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        # The negative test checks that an empty response fails
        assert 'response: str = ""' in result

    def test_missing_behaviors_key_raises_value_error(self) -> None:
        gen = _make_generator()
        with pytest.raises(ValueError, match="behaviors"):
            gen.generate_from_dict({"agent_name": "x"})

    def test_empty_behaviors_list_raises_value_error(self) -> None:
        gen = _make_generator()
        with pytest.raises(ValueError, match="behaviors"):
            gen.generate_from_dict({"behaviors": []})

    def test_output_ends_with_newline(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert result.endswith("\n")

    def test_given_when_then_in_docstring(self) -> None:
        gen = _make_generator()
        result = gen.generate_from_dict(_MINIMAL_SPEC)
        assert "Given:" in result
        assert "When:" in result
        assert "Then:" in result

    def test_dimensions_only_accuracy_skips_safety_assertion(self) -> None:
        gen = _make_generator()
        spec: dict[str, object] = {
            "behaviors": [
                {
                    "name": "accuracy_only",
                    "given": "test",
                    "when": "test",
                    "then": "test",
                    "expected_intent": "test",
                    "dimensions": ["accuracy"],
                    "threshold": 0.8,
                }
            ]
        }
        result = gen.generate_from_dict(spec)
        # Only accuracy dimension — safety assertion should NOT be present in positive test
        # Count occurrences: "assert_safety" should not appear in positive body
        # We rely on the fact the spec has no "safety" dimension
        positive_start = result.find("def test_accuracy_only_positive")
        negative_start = result.find("def test_accuracy_only_negative")
        positive_body = result[positive_start:negative_start]
        assert "assert_safety" not in positive_body


# ---------------------------------------------------------------------------
# EvalScaffoldGenerator.generate_from_bsl
# ---------------------------------------------------------------------------


class TestGenerateFromBsl:
    def test_raises_file_not_found_for_missing_file(self, tmp_path: Path) -> None:
        gen = _make_generator()
        with pytest.raises(FileNotFoundError, match="BSL spec not found"):
            gen.generate_from_bsl(tmp_path / "nonexistent.bsl")

    def test_raises_value_error_for_no_behaviors(self, tmp_path: Path) -> None:
        bsl_file = tmp_path / "empty.bsl"
        bsl_file.write_text(_BSL_TEXT_NO_BEHAVIORS, encoding="utf-8")
        gen = _make_generator()
        with pytest.raises(ValueError, match="No behaviour blocks found"):
            gen.generate_from_bsl(bsl_file)

    def test_valid_bsl_file_generates_test_module(self, tmp_path: Path) -> None:
        bsl_file = tmp_path / "spec.bsl"
        bsl_file.write_text(_BSL_TEXT_VALID, encoding="utf-8")
        gen = _make_generator()
        result = gen.generate_from_bsl(bsl_file)
        assert isinstance(result, str)
        assert "import pytest" in result

    def test_bsl_file_stem_used_as_agent_name(self, tmp_path: Path) -> None:
        bsl_file = tmp_path / "my_service.bsl"
        bsl_file.write_text(_BSL_TEXT_VALID, encoding="utf-8")
        gen = _make_generator()
        result = gen.generate_from_bsl(bsl_file)
        assert "my_service" in result

    def test_bsl_generates_positive_tests_per_behavior(self, tmp_path: Path) -> None:
        bsl_file = tmp_path / "spec.bsl"
        bsl_file.write_text(_BSL_TEXT_VALID, encoding="utf-8")
        gen = _make_generator()
        result = gen.generate_from_bsl(bsl_file)
        assert "def test_capital_query_positive" in result
        assert "def test_safety_check_positive" in result

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        bsl_file = tmp_path / "spec.bsl"
        bsl_file.write_text(_BSL_TEXT_VALID, encoding="utf-8")
        gen = _make_generator()
        # Pass as str rather than Path
        result = gen.generate_from_bsl(str(bsl_file))
        assert "import pytest" in result


# ---------------------------------------------------------------------------
# EvalScaffoldGenerator.write_to_file
# ---------------------------------------------------------------------------


class TestWriteToFile:
    def test_writes_content_to_path(self, tmp_path: Path) -> None:
        gen = _make_generator()
        output_path = tmp_path / "test_output.py"
        content = "# generated\n"
        gen.write_to_file(content, output_path)
        assert output_path.read_text(encoding="utf-8") == content

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        gen = _make_generator()
        output_path = tmp_path / "test_output.py"
        returned = gen.write_to_file("# x\n", output_path)
        assert returned == output_path

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        gen = _make_generator()
        nested_path = tmp_path / "deeply" / "nested" / "test_file.py"
        gen.write_to_file("# content\n", nested_path)
        assert nested_path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        gen = _make_generator()
        output_path = tmp_path / "test_output.py"
        output_path.write_text("old content", encoding="utf-8")
        gen.write_to_file("new content\n", output_path)
        assert output_path.read_text(encoding="utf-8") == "new content\n"

    def test_full_round_trip_from_dict(self, tmp_path: Path) -> None:
        gen = _make_generator()
        content = gen.generate_from_dict(_MINIMAL_SPEC)
        output_path = tmp_path / "tests" / "test_generated_eval.py"
        returned = gen.write_to_file(content, output_path)
        assert returned.exists()
        assert "def test_capital_city_query_positive" in returned.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Package __init__ re-export
# ---------------------------------------------------------------------------


class TestPackageExport:
    def test_eval_scaffold_generator_importable_from_package(self) -> None:
        from agent_eval.pytest_plugin import EvalScaffoldGenerator as ExportedGenerator

        assert ExportedGenerator is EvalScaffoldGenerator

    def test_eval_scaffold_generator_in_all(self) -> None:
        import agent_eval.pytest_plugin as plugin_pkg

        assert "EvalScaffoldGenerator" in plugin_pkg.__all__


# ---------------------------------------------------------------------------
# Branch coverage: _render_positive_test / _render_negative_test edge cases
# ---------------------------------------------------------------------------


class TestRenderPositiveTestBranches:
    def test_non_list_dimensions_falls_back_to_defaults(self) -> None:
        # Pass dimensions as a non-list to exercise the else branch on line 248.
        behavior: dict[str, object] = {
            "name": "fallback_dims",
            "given": "test given",
            "when": "test when",
            "then": "test then",
            "expected_intent": "test",
            "dimensions": "not-a-list",  # triggers the else branch
            "threshold": 0.8,
        }
        result = _render_positive_test(behavior)
        # Defaults include all four dimensions — all assertions should appear.
        assert "assert_accuracy" in result
        assert "assert_safety" in result
        assert "assert_cost" in result
        assert "assert_latency" in result

    def test_non_numeric_threshold_falls_back_to_default(self) -> None:
        behavior: dict[str, object] = {
            "name": "bad_threshold",
            "given": "g",
            "when": "w",
            "then": "t",
            "expected_intent": "intent",
            "dimensions": ["accuracy"],
            "threshold": "not-a-number",  # triggers the isinstance guard
        }
        result = _render_positive_test(behavior)
        # Default threshold 0.8 should appear in the marker.
        assert "0.8" in result

    def test_empty_dimensions_generates_pass_body(self) -> None:
        # When no known dimension is requested, assertions_block falls back to "pass".
        behavior: dict[str, object] = {
            "name": "no_dims",
            "given": "g",
            "when": "w",
            "then": "t",
            "expected_intent": "intent",
            "dimensions": [],  # no recognized dimensions
            "threshold": 0.8,
        }
        result = _render_positive_test(behavior)
        assert "pass" in result


class TestRenderNegativeTestBranches:
    def test_non_numeric_threshold_falls_back_to_default(self) -> None:
        behavior: dict[str, object] = {
            "name": "bad_threshold_neg",
            "given": "g",
            "when": "w",
            "then": "t",
            "expected_intent": "intent",
            "dimensions": ["accuracy"],
            "threshold": None,  # triggers the isinstance guard in negative test
        }
        result = _render_negative_test(behavior)
        assert "0.8" in result

    def test_negative_test_contains_assert_not_all_passed(self) -> None:
        behavior: dict[str, object] = {
            "name": "neg_check",
            "given": "g",
            "when": "w",
            "then": "t",
            "expected_intent": "something",
            "dimensions": ["accuracy"],
            "threshold": 0.8,
        }
        result = _render_negative_test(behavior)
        assert "assert not eval_context.all_passed" in result


# ---------------------------------------------------------------------------
# Branch coverage: _parse_bsl_with_library (mocked bsl_lang)
# ---------------------------------------------------------------------------


class TestParseBslWithLibrary:
    def test_library_path_returns_behaviors(self) -> None:
        """Test _parse_bsl_with_library when bsl_lang is synthetically available."""
        import agent_eval.pytest_plugin.scaffold as scaffold_module

        # Create a minimal synthetic bsl_lang module.
        fake_bsl_lang = types.ModuleType("bsl_lang")

        def _fake_parse(text: str) -> object:
            ast = MagicMock()
            node = MagicMock()
            node.name = "my_behavior"
            node.given = "given text"
            node.when = "when text"
            node.then = "then text"
            node.expected_intent = "result"
            ast.behaviors = [node]
            return ast

        fake_bsl_lang.parse = _fake_parse  # type: ignore[attr-defined]

        original_available = scaffold_module._BSL_AVAILABLE
        try:
            # Patch the module-level flag and inject the fake module.
            scaffold_module._BSL_AVAILABLE = True
            sys.modules["bsl_lang"] = fake_bsl_lang

            gen = EvalScaffoldGenerator()
            behaviors = gen._parse_bsl_with_library("behavior my_behavior { }", Path("x.bsl"))
            assert len(behaviors) == 1
            assert behaviors[0]["name"] == "my_behavior"
        finally:
            scaffold_module._BSL_AVAILABLE = original_available
            sys.modules.pop("bsl_lang", None)

    def test_library_exception_falls_back_to_regex(self) -> None:
        """Test _parse_bsl_with_library falls back when bsl_lang.parse raises."""
        import agent_eval.pytest_plugin.scaffold as scaffold_module

        fake_bsl_lang = types.ModuleType("bsl_lang")

        def _raising_parse(text: str) -> object:
            raise RuntimeError("parse error")

        fake_bsl_lang.parse = _raising_parse  # type: ignore[attr-defined]

        original_available = scaffold_module._BSL_AVAILABLE
        try:
            scaffold_module._BSL_AVAILABLE = True
            sys.modules["bsl_lang"] = fake_bsl_lang

            gen = EvalScaffoldGenerator()
            # The BSL text has valid behavior blocks — regex fallback should find them.
            behaviors = gen._parse_bsl_with_library(_BSL_TEXT_VALID, Path("x.bsl"))
            assert len(behaviors) == 2
        finally:
            scaffold_module._BSL_AVAILABLE = original_available
            sys.modules.pop("bsl_lang", None)

    def test_generate_from_bsl_routes_through_parse_bsl_with_library(
        self, tmp_path: Path
    ) -> None:
        """Integration: generate_from_bsl calls _parse_bsl_with_library when available."""
        import agent_eval.pytest_plugin.scaffold as scaffold_module

        bsl_file = tmp_path / "spec.bsl"
        bsl_file.write_text(_BSL_TEXT_VALID, encoding="utf-8")

        original_available = scaffold_module._BSL_AVAILABLE
        captured_calls: list[str] = []

        def _mock_parse_bsl(self_inner: object, bsl_text: str, spec_path: Path) -> list[dict[str, object]]:
            captured_calls.append("called")
            # Return behaviors via regex so the rest of the pipeline works.
            return _extract_behaviors_from_bsl_text(bsl_text)

        try:
            scaffold_module._BSL_AVAILABLE = True
            gen = EvalScaffoldGenerator()
            # Patch the instance method directly.
            gen._parse_bsl_with_library = lambda bsl_text, spec_path: (  # type: ignore[method-assign]
                captured_calls.append("called") or _extract_behaviors_from_bsl_text(bsl_text)
            )
            result = gen.generate_from_bsl(bsl_file)
            assert captured_calls, "_parse_bsl_with_library was not called"
            assert "import pytest" in result
        finally:
            scaffold_module._BSL_AVAILABLE = original_available


# ---------------------------------------------------------------------------
# Branch coverage: _extract_behaviors_from_bsl_text — no-quoted-intent path
# ---------------------------------------------------------------------------


class TestExtractBehaviorsNoquotedIntent:
    def test_then_without_quoted_intent_uses_last_word(self) -> None:
        # "then" clause with no quoted string — fallback to last word.
        bsl_text = textwrap.dedent(
            """\
            behavior plain_then {
                given: "Some context"
                when: "Some action"
                then: The response is correct
            }
            """
        )
        behaviors = _extract_behaviors_from_bsl_text(bsl_text)
        assert len(behaviors) == 1
        intent = str(behaviors[0]["expected_intent"])
        # Last word of "The response is correct" should be "correct"
        assert intent == "correct"

    def test_then_with_empty_body_gives_empty_expected_intent(self) -> None:
        # A behavior block with empty then clause.
        bsl_text = textwrap.dedent(
            """\
            behavior empty_then {
                given: "g"
                when: "w"
            }
            """
        )
        behaviors = _extract_behaviors_from_bsl_text(bsl_text)
        assert len(behaviors) == 1
        # No "then" match — expected_intent should be empty string.
        assert behaviors[0]["expected_intent"] == ""
