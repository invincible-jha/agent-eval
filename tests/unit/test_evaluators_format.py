"""Unit tests for agent_eval.evaluators.format.

Tests BasicFormatEvaluator and the private validation helpers for
JSON, XML, Markdown, and plain formats.
"""
from __future__ import annotations

import pytest

from agent_eval.core.evaluator import Dimension
from agent_eval.evaluators.format import (
    BasicFormatEvaluator,
    _validate_json,
    _validate_markdown,
    _validate_xml,
)


# ---------------------------------------------------------------------------
# _validate_json
# ---------------------------------------------------------------------------


class TestValidateJson:
    def test_valid_json_object(self) -> None:
        valid, reason = _validate_json('{"key": "value"}', [])
        assert valid is True

    def test_valid_json_array(self) -> None:
        valid, reason = _validate_json('[1, 2, 3]', [])
        assert valid is True

    def test_invalid_json_returns_false(self) -> None:
        valid, reason = _validate_json('not json at all', [])
        assert valid is False
        assert "Invalid JSON" in reason

    def test_required_fields_present_passes(self) -> None:
        valid, reason = _validate_json('{"name": "Alice", "age": 30}', ["name", "age"])
        assert valid is True

    def test_missing_required_field_fails(self) -> None:
        valid, reason = _validate_json('{"name": "Alice"}', ["name", "missing_field"])
        assert valid is False
        assert "missing_field" in reason

    def test_json_in_markdown_code_block(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        valid, reason = _validate_json(text, [])
        assert valid is True

    def test_json_embedded_with_prefix_text(self) -> None:
        text = 'Here is the output: {"result": 42}'
        valid, reason = _validate_json(text, [])
        assert valid is True

    def test_array_with_required_fields_skips_field_check(self) -> None:
        # Required fields check only applies to dict, not array
        valid, reason = _validate_json('[1, 2, 3]', ["name"])
        assert valid is True


# ---------------------------------------------------------------------------
# _validate_xml
# ---------------------------------------------------------------------------


class TestValidateXml:
    def test_valid_xml_passes(self) -> None:
        valid, reason = _validate_xml("<root><child>text</child></root>")
        assert valid is True

    def test_invalid_xml_fails(self) -> None:
        valid, reason = _validate_xml("<unclosed>tag")
        assert valid is False
        assert "Invalid XML" in reason

    def test_xml_in_markdown_code_block(self) -> None:
        text = "```xml\n<root>value</root>\n```"
        valid, reason = _validate_xml(text)
        assert valid is True

    def test_empty_element_valid(self) -> None:
        valid, reason = _validate_xml("<empty/>")
        assert valid is True


# ---------------------------------------------------------------------------
# _validate_markdown
# ---------------------------------------------------------------------------


class TestValidateMarkdown:
    def test_header_detected(self) -> None:
        valid, reason = _validate_markdown("# My Header\nSome text")
        assert valid is True

    def test_unordered_list_detected(self) -> None:
        valid, reason = _validate_markdown("- item one\n- item two")
        assert valid is True

    def test_ordered_list_detected(self) -> None:
        valid, reason = _validate_markdown("1. first\n2. second")
        assert valid is True

    def test_bold_detected(self) -> None:
        valid, reason = _validate_markdown("This is **bold** text")
        assert valid is True

    def test_code_block_detected(self) -> None:
        valid, reason = _validate_markdown("```python\nprint('hi')\n```")
        assert valid is True

    def test_link_detected(self) -> None:
        valid, reason = _validate_markdown("[link text](https://example.com)")
        assert valid is True

    def test_plain_text_not_markdown(self) -> None:
        valid, reason = _validate_markdown("just plain text with no structure")
        assert valid is False

    def test_asterisk_list_detected(self) -> None:
        valid, reason = _validate_markdown("* item one\n* item two")
        assert valid is True


# ---------------------------------------------------------------------------
# BasicFormatEvaluator construction
# ---------------------------------------------------------------------------


class TestBasicFormatEvaluatorConstruction:
    def test_default_format_is_auto(self) -> None:
        ev = BasicFormatEvaluator()
        assert ev.expected_format == "auto"

    def test_required_fields_default_empty(self) -> None:
        ev = BasicFormatEvaluator()
        assert ev.required_fields == []

    def test_dimension_property(self) -> None:
        ev = BasicFormatEvaluator()
        assert ev.dimension == Dimension.FORMAT

    def test_name_property(self) -> None:
        ev = BasicFormatEvaluator()
        assert ev.name == "BasicFormatEvaluator"


# ---------------------------------------------------------------------------
# evaluate — empty output
# ---------------------------------------------------------------------------


class TestFormatEvaluatorEmptyOutput:
    def test_empty_output_fails(self) -> None:
        ev = BasicFormatEvaluator(expected_format="json")
        result = ev.evaluate("c1", "", None, {})
        assert result.passed is False
        assert result.score == 0.0
        assert "Empty" in result.reason

    def test_whitespace_only_output_fails(self) -> None:
        ev = BasicFormatEvaluator(expected_format="json")
        result = ev.evaluate("c1", "   ", None, {})
        assert result.passed is False


# ---------------------------------------------------------------------------
# evaluate — JSON format
# ---------------------------------------------------------------------------


class TestFormatEvaluatorJSON:
    @pytest.fixture
    def ev(self) -> BasicFormatEvaluator:
        return BasicFormatEvaluator(expected_format="json")

    def test_valid_json_passes(self, ev: BasicFormatEvaluator) -> None:
        result = ev.evaluate("c1", '{"answer": 42}', None, {})
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_json_fails(self, ev: BasicFormatEvaluator) -> None:
        result = ev.evaluate("c1", "not json", None, {})
        assert result.passed is False

    def test_required_fields_enforced(self) -> None:
        ev = BasicFormatEvaluator(expected_format="json", required_fields=["name", "age"])
        result = ev.evaluate("c1", '{"name": "Alice"}', None, {})
        assert result.passed is False

    def test_all_required_fields_present_passes(self) -> None:
        ev = BasicFormatEvaluator(expected_format="json", required_fields=["name"])
        result = ev.evaluate("c1", '{"name": "Alice"}', None, {})
        assert result.passed is True


# ---------------------------------------------------------------------------
# evaluate — XML format
# ---------------------------------------------------------------------------


class TestFormatEvaluatorXML:
    @pytest.fixture
    def ev(self) -> BasicFormatEvaluator:
        return BasicFormatEvaluator(expected_format="xml")

    def test_valid_xml_passes(self, ev: BasicFormatEvaluator) -> None:
        result = ev.evaluate("c1", "<response>text</response>", None, {})
        assert result.passed is True

    def test_invalid_xml_fails(self, ev: BasicFormatEvaluator) -> None:
        result = ev.evaluate("c1", "<broken xml", None, {})
        assert result.passed is False


# ---------------------------------------------------------------------------
# evaluate — Markdown format
# ---------------------------------------------------------------------------


class TestFormatEvaluatorMarkdown:
    @pytest.fixture
    def ev(self) -> BasicFormatEvaluator:
        return BasicFormatEvaluator(expected_format="markdown")

    def test_valid_markdown_passes(self, ev: BasicFormatEvaluator) -> None:
        result = ev.evaluate("c1", "# Heading\n- item", None, {})
        assert result.passed is True

    def test_plain_text_fails_markdown_check(self, ev: BasicFormatEvaluator) -> None:
        result = ev.evaluate("c1", "this is just text", None, {})
        assert result.passed is False


# ---------------------------------------------------------------------------
# evaluate — plain format
# ---------------------------------------------------------------------------


class TestFormatEvaluatorPlain:
    def test_plain_always_passes_non_empty(self) -> None:
        ev = BasicFormatEvaluator(expected_format="plain")
        result = ev.evaluate("c1", "any text works", None, {})
        assert result.passed is True

    def test_strict_plain_fails_empty_output(self) -> None:
        # strict=True means empty output fails — but we already catch empty before
        ev = BasicFormatEvaluator(expected_format="plain", strict=True)
        # Empty output is caught before the strict check
        result = ev.evaluate("c1", "", None, {})
        assert result.passed is False


# ---------------------------------------------------------------------------
# evaluate — auto format
# ---------------------------------------------------------------------------


class TestFormatEvaluatorAuto:
    def test_auto_reads_format_from_metadata(self) -> None:
        ev = BasicFormatEvaluator(expected_format="auto")
        result = ev.evaluate("c1", '{"key": "value"}', None, {"expected_format": "json"})
        assert result.passed is True

    def test_auto_defaults_to_plain_when_no_metadata(self) -> None:
        ev = BasicFormatEvaluator(expected_format="auto")
        result = ev.evaluate("c1", "any text output", None, {})
        assert result.passed is True

    def test_auto_with_markdown_metadata(self) -> None:
        ev = BasicFormatEvaluator(expected_format="auto")
        result = ev.evaluate(
            "c1", "# Header\n- list item", None,
            {"expected_format": "markdown"}
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# evaluate — unknown format
# ---------------------------------------------------------------------------


class TestFormatEvaluatorUnknownFormat:
    def test_unknown_format_passes_with_note(self) -> None:
        ev = BasicFormatEvaluator(expected_format="auto")  # type: ignore[arg-type]
        # Inject unknown format via metadata
        result = ev.evaluate("c1", "some output", None, {"expected_format": "csv"})
        assert result.passed is True
        assert "csv" in result.reason.lower() or "unknown" in result.reason.lower()
