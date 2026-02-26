"""Basic format evaluator for agent-eval.

Validates that agent output conforms to a required structural format:
JSON, XML, Markdown, or plain text. For structured outputs, validates
that required fields are present.

NOTE: This is a commodity format evaluator. It performs syntactic
validation only. It does NOT perform semantic schema validation (JSON Schema,
XML Schema/DTD), does NOT verify field types or value constraints beyond
presence checks, and does NOT validate against OpenAPI response schemas.
Full schema validation is available via the plugin system.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Literal

from agent_eval.core.evaluator import Dimension, DimensionScore, Evaluator

FormatType = Literal["json", "xml", "markdown", "plain", "auto"]


def _validate_json(text: str, required_fields: list[str]) -> tuple[bool, str]:
    """Attempt to parse text as JSON and check for required fields.

    Parameters
    ----------
    text:
        The string to validate.
    required_fields:
        Top-level keys that must be present in the parsed JSON object.

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason)
    """
    stripped = text.strip()
    # Find JSON content (may be embedded in markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", stripped)
    if json_match:
        stripped = json_match.group(1)
    else:
        # Try to find raw JSON
        json_start = stripped.find("{")
        if json_start == -1:
            json_start = stripped.find("[")
        if json_start >= 0:
            stripped = stripped[json_start:]

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    if required_fields and isinstance(parsed, dict):
        missing = [f for f in required_fields if f not in parsed]
        if missing:
            return False, f"Missing required JSON fields: {missing}"

    return True, "Valid JSON"


def _validate_xml(text: str) -> tuple[bool, str]:
    """Attempt to parse text as XML.

    Parameters
    ----------
    text:
        The string to validate.

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason)
    """
    stripped = text.strip()
    # Extract XML from markdown code block if present
    xml_match = re.search(r"```(?:xml)?\s*([\s\S]*?)\s*```", stripped)
    if xml_match:
        stripped = xml_match.group(1).strip()

    try:
        ET.fromstring(stripped)
        return True, "Valid XML"
    except ET.ParseError as exc:
        return False, f"Invalid XML: {exc}"


def _validate_markdown(text: str) -> tuple[bool, str]:
    """Check that text contains at least minimal Markdown structure.

    Passes if the text contains any of: headers (#), lists (- or *),
    bold (**), italic (*), code blocks (```), or links ([text](url)).

    Parameters
    ----------
    text:
        The string to validate.

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason)
    """
    markdown_indicators = [
        re.compile(r"^#{1,6}\s+", re.MULTILINE),  # Headers
        re.compile(r"^[-*+]\s+", re.MULTILINE),     # Lists
        re.compile(r"\*\*[^*]+\*\*"),               # Bold
        re.compile(r"```"),                          # Code blocks
        re.compile(r"\[.+?\]\(.+?\)"),              # Links
        re.compile(r"^\d+\.\s+", re.MULTILINE),     # Ordered lists
    ]
    for pattern in markdown_indicators:
        if pattern.search(text):
            return True, "Valid Markdown structure detected"
    return False, "No Markdown structure detected (headers, lists, bold, code blocks)"


class BasicFormatEvaluator(Evaluator):
    """Evaluates whether agent output conforms to a required format.

    Supported formats:
    - "json":     Output must be parseable as JSON. Optional required fields.
    - "xml":      Output must be parseable as XML.
    - "markdown": Output must contain Markdown structural elements.
    - "plain":    Always passes (accepts any text output).
    - "auto":     Uses expected_format from test case metadata, or "plain".

    NOTE: This is NOT a schema validator. JSON validation checks parseability
    and field presence only, NOT field types, value constraints, or JSON Schema
    compliance. XML validation checks well-formedness only, NOT schema/DTD
    conformance. Full schema validation is available via the plugin system.

    Parameters
    ----------
    expected_format:
        The format to validate against. Use "auto" to read from test case
        metadata (expected_format field).
    required_fields:
        For JSON format: list of top-level keys that must be present.
    strict:
        If True, "plain" format still checks that output is non-empty.
    """

    def __init__(
        self,
        expected_format: FormatType = "auto",
        required_fields: list[str] | None = None,
        strict: bool = False,
    ) -> None:
        self.expected_format = expected_format
        self.required_fields = required_fields or []
        self.strict = strict

    @property
    def dimension(self) -> Dimension:
        return Dimension.FORMAT

    @property
    def name(self) -> str:
        return "BasicFormatEvaluator"

    def evaluate(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        """Validate agent output format.

        Parameters
        ----------
        case_id:
            Test case identifier.
        agent_output:
            The agent's output text to validate.
        expected_output:
            Not used by this evaluator.
        metadata:
            May contain ``expected_format`` key when self.expected_format
            is "auto".

        Returns
        -------
        DimensionScore
        """
        if not agent_output.strip():
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                reason="Empty agent output",
            )

        # Determine effective format
        effective_format: str = self.expected_format
        if effective_format == "auto":
            fmt_raw = metadata.get("expected_format", "plain")
            effective_format = str(fmt_raw) if fmt_raw else "plain"

        if effective_format == "json":
            is_valid, reason = _validate_json(agent_output, self.required_fields)
        elif effective_format == "xml":
            is_valid, reason = _validate_xml(agent_output)
        elif effective_format == "markdown":
            is_valid, reason = _validate_markdown(agent_output)
        elif effective_format == "plain":
            if self.strict and not agent_output.strip():
                is_valid, reason = False, "Empty output in strict mode"
            else:
                is_valid, reason = True, "Plain text output accepted"
        else:
            # Unknown format: pass with a note
            is_valid = True
            reason = f"Unknown format {effective_format!r}; skipping validation"

        return DimensionScore(
            dimension=self.dimension,
            score=1.0 if is_valid else 0.0,
            passed=is_valid,
            reason=reason,
        )
