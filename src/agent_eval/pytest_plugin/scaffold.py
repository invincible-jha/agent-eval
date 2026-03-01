"""BSL-to-pytest scaffold generator for pytest-agent-eval.

Generates pytest evaluation test files from BSL (Behaviour Specification
Language) specifications or plain Python dicts. Each behaviour in the
specification becomes one or two pytest test functions: a positive case that
asserts the expected outcome and a negative (edge-case) variant that checks
handling of boundary or invalid inputs.

Usage — from a dict spec
------------------------
::

    from agent_eval.pytest_plugin.scaffold import EvalScaffoldGenerator
    from pathlib import Path

    generator = EvalScaffoldGenerator()
    spec = {
        "agent_name": "my_agent",
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
    content = generator.generate_from_dict(spec)
    output_path = generator.write_to_file(content, Path("tests/test_my_agent_eval.py"))

Usage — from a BSL spec file
-----------------------------
::

    content = generator.generate_from_bsl(Path("specs/my_agent.bsl"))
    generator.write_to_file(content, Path("tests/test_my_agent_eval.py"))

Generated file structure
------------------------
- Module-level docstring describing the source spec
- Standard imports (``pytest``, ``agent_eval.pytest_plugin``)
- One ``test_<behavior_name>_positive`` function per behaviour
- One ``test_<behavior_name>_negative`` function per behaviour
- Each test function carries the ``@pytest.mark.agent_eval`` marker
- Docstrings derived from the Given/When/Then fields in the spec
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Optional bsl-lang dependency
# ---------------------------------------------------------------------------
# bsl-lang is an optional dependency.  Guard the import so that scaffold.py
# works in environments that do not have bsl-lang installed.  When it is
# available, ``_BSL_AVAILABLE`` is True and :meth:`EvalScaffoldGenerator.generate_from_bsl`
# can parse real BSL files.  When it is absent the method falls back to
# reading the file as plain text and extracting behaviours via a simple
# regex heuristic.

try:
    import bsl_lang  # type: ignore[import-not-found]

    _BSL_AVAILABLE: bool = True
except ModuleNotFoundError:
    _BSL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_INDENT: str = "    "
_SAFE_NAME_PATTERN: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9_]")
_BEHAVIOR_BLOCK_PATTERN: re.Pattern[str] = re.compile(
    r"behavior\s+(\w+)\s*\{([^}]*)\}", re.DOTALL | re.IGNORECASE
)
_GIVEN_PATTERN: re.Pattern[str] = re.compile(
    r'given\s*:\s*"?([^\n"]+)"?', re.IGNORECASE
)
_WHEN_PATTERN: re.Pattern[str] = re.compile(
    r'when\s*:\s*"?([^\n"]+)"?', re.IGNORECASE
)
_THEN_PATTERN: re.Pattern[str] = re.compile(
    r'then\s*:\s*"?([^\n"]+)"?', re.IGNORECASE
)

_DEFAULT_DIMENSIONS: tuple[str, ...] = ("accuracy", "safety", "cost", "latency")
_DEFAULT_THRESHOLD: float = 0.8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_safe_identifier(text: str) -> str:
    """Convert *text* into a valid Python identifier fragment.

    Replaces any character that is not alphanumeric or underscore with an
    underscore and strips leading/trailing underscores.

    Parameters
    ----------
    text:
        Arbitrary string to convert.

    Returns
    -------
    str
        A lowercased identifier-safe string, e.g.
        ``"capital city query"`` -> ``"capital_city_query"``.
    """
    safe = _SAFE_NAME_PATTERN.sub("_", text.lower()).strip("_")
    # Collapse multiple consecutive underscores.
    safe = re.sub(r"_+", "_", safe)
    return safe or "behavior"


def _extract_behaviors_from_bsl_text(bsl_text: str) -> list[dict[str, object]]:
    """Extract behaviour dicts from raw BSL source text using regex.

    This is a best-effort fallback used when the ``bsl-lang`` package is
    not installed.  It recognises the canonical BSL block structure::

        behavior <name> {
            given: "..."
            when:  "..."
            then:  "..."
        }

    Parameters
    ----------
    bsl_text:
        Raw content of a ``.bsl`` specification file.

    Returns
    -------
    list[dict[str, object]]
        One dict per matched behaviour block.  Missing fields default to
        empty strings.
    """
    behaviors: list[dict[str, object]] = []
    for match in _BEHAVIOR_BLOCK_PATTERN.finditer(bsl_text):
        name = match.group(1).strip()
        body = match.group(2)

        given_match = _GIVEN_PATTERN.search(body)
        when_match = _WHEN_PATTERN.search(body)
        then_match = _THEN_PATTERN.search(body)

        given = given_match.group(1).strip() if given_match else ""
        when = when_match.group(1).strip() if when_match else ""
        then = then_match.group(1).strip() if then_match else ""

        # Derive a basic expected_intent from the "then" clause — the first
        # quoted string or the last word.
        intent_match = re.search(r"'([^']+)'|\"([^\"]+)\"", then)
        if intent_match:
            expected_intent: str = intent_match.group(1) or intent_match.group(2) or then
        else:
            # Take the last meaningful word as a rough intent keyword.
            words = then.split()
            expected_intent = words[-1].strip(".'\"") if words else ""

        behaviors.append(
            {
                "name": name,
                "given": given,
                "when": when,
                "then": then,
                "expected_intent": expected_intent,
                "dimensions": list(_DEFAULT_DIMENSIONS),
                "threshold": _DEFAULT_THRESHOLD,
            }
        )
    return behaviors


def _render_dimensions_list(dimensions: object) -> str:
    """Render a dimensions value as a Python list literal string.

    Parameters
    ----------
    dimensions:
        Either a list/tuple of strings or any other value.  Non-sequence
        values cause the default dimensions to be used.

    Returns
    -------
    str
        Python source for a list literal, e.g.
        ``'["accuracy", "safety"]'``.
    """
    if isinstance(dimensions, (list, tuple)):
        parts = [f'"{str(d)}"' for d in dimensions]
        return "[" + ", ".join(parts) + "]"
    parts = [f'"{d}"' for d in _DEFAULT_DIMENSIONS]
    return "[" + ", ".join(parts) + "]"


def _render_positive_test(behavior: dict[str, object]) -> str:
    """Render the positive pytest test function for *behavior*.

    The positive test exercises the happy path: the agent produces the
    expected output and should pass accuracy, safety, cost, and latency
    assertions.

    Parameters
    ----------
    behavior:
        A behaviour dict with keys ``name``, ``given``, ``when``, ``then``,
        ``expected_intent``, ``dimensions``, and ``threshold``.

    Returns
    -------
    str
        Source code for the test function, including decorator and body.
    """
    name = _to_safe_identifier(str(behavior.get("name", "behavior")))
    given = str(behavior.get("given", ""))
    when = str(behavior.get("when", ""))
    then = str(behavior.get("then", ""))
    expected_intent = str(behavior.get("expected_intent", ""))
    dimensions = behavior.get("dimensions", list(_DEFAULT_DIMENSIONS))
    threshold = behavior.get("threshold", _DEFAULT_THRESHOLD)
    if not isinstance(threshold, (int, float)):
        threshold = _DEFAULT_THRESHOLD
    threshold_float = float(threshold)

    dim_list = _render_dimensions_list(dimensions)

    # Build the assertion block — one eval_context call per requested dimension.
    assertion_lines: list[str] = []
    dim_strings: list[str] = []
    if isinstance(dimensions, (list, tuple)):
        dim_strings = [str(d) for d in dimensions]
    else:
        dim_strings = list(_DEFAULT_DIMENSIONS)

    if "accuracy" in dim_strings:
        assertion_lines.append(
            f'eval_context.assert_accuracy(response, expected_intent="{expected_intent}",'
            f" threshold={threshold_float})"
        )
    if "safety" in dim_strings:
        assertion_lines.append("eval_context.assert_safety(response)")
    if "cost" in dim_strings:
        assertion_lines.append("eval_context.assert_cost(response)")
    if "latency" in dim_strings:
        assertion_lines.append("eval_context.assert_latency(elapsed_seconds)")

    assertions_block = ("\n" + _INDENT * 2).join(assertion_lines) if assertion_lines else "pass"

    lines: list[str] = [
        f"@pytest.mark.agent_eval(dimensions={dim_list}, threshold={threshold_float})",
        f"def test_{name}_positive(eval_context: EvalContext) -> None:",
        f'{_INDENT}"""Positive evaluation: {name}.',
        "",
        f"{_INDENT}Given:  {given}",
        f"{_INDENT}When:   {when}",
        f"{_INDENT}Then:   {then}",
        "",
        f"{_INDENT}This test verifies the happy-path behaviour.  Replace the",
        f"{_INDENT}placeholder ``agent_response`` call with your actual agent",
        f"{_INDENT}invocation before running the suite.",
        f'{_INDENT}"""',
        f"{_INDENT}import time",
        "",
        f"{_INDENT}# TODO: replace with your agent invocation.",
        f"{_INDENT}start = time.monotonic()",
        f'{_INDENT}response: str = agent_response("{expected_intent}")',
        f"{_INDENT}elapsed_seconds: float = time.monotonic() - start",
        "",
        f"{_INDENT}{assertions_block}",
        f"{_INDENT}assert eval_context.all_passed",
    ]

    return "\n".join(lines)


def _render_negative_test(behavior: dict[str, object]) -> str:
    """Render the negative / edge-case pytest test for *behavior*.

    The negative test exercises boundary conditions: an empty agent response
    should be detected as a failing accuracy evaluation.

    Parameters
    ----------
    behavior:
        A behaviour dict (same shape as for :func:`_render_positive_test`).

    Returns
    -------
    str
        Source code for the negative test function.
    """
    name = _to_safe_identifier(str(behavior.get("name", "behavior")))
    given = str(behavior.get("given", ""))
    then = str(behavior.get("then", ""))
    expected_intent = str(behavior.get("expected_intent", ""))
    dimensions = behavior.get("dimensions", list(_DEFAULT_DIMENSIONS))
    threshold = behavior.get("threshold", _DEFAULT_THRESHOLD)
    if not isinstance(threshold, (int, float)):
        threshold = _DEFAULT_THRESHOLD
    threshold_float = float(threshold)

    dim_list = _render_dimensions_list(dimensions)

    lines: list[str] = [
        f"@pytest.mark.agent_eval(dimensions={dim_list}, threshold={threshold_float})",
        f"def test_{name}_negative(eval_context: EvalContext) -> None:",
        f'{_INDENT}"""Negative / edge-case evaluation: {name}.',
        "",
        f"{_INDENT}Given:  {given} (empty / invalid response variant)",
        f"{_INDENT}Then:   {then} — should NOT be satisfied by an empty response.",
        "",
        f"{_INDENT}This test verifies that an empty agent response correctly fails",
        f"{_INDENT}the accuracy gate.  Extend it to cover other boundary conditions",
        f"{_INDENT}specific to your agent implementation.",
        f'{_INDENT}"""',
        f"{_INDENT}# Simulate an empty / degenerate agent response.",
        f'{_INDENT}response: str = ""',
        "",
        f'{_INDENT}eval_context.assert_accuracy(response, expected_intent="{expected_intent}")',
        f"{_INDENT}# An empty response must fail the accuracy check.",
        f"{_INDENT}assert not eval_context.all_passed",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvalScaffoldGenerator
# ---------------------------------------------------------------------------


class EvalScaffoldGenerator:
    """Generates pytest evaluation test files from BSL specifications or dicts.

    Instances are stateless — all configuration is passed per-call.  The
    generator is intentionally simple: it produces idiomatic, human-readable
    pytest files that developers are expected to flesh out with real agent
    calls before running the suite.

    Methods
    -------
    generate_from_bsl(bsl_spec)
        Parse a BSL file and return a complete ``.py`` file as a string.
    generate_from_dict(spec)
        Accept a plain Python dict and return a complete ``.py`` file string.
    write_to_file(test_content, output_path)
        Write *test_content* to *output_path*, creating parent dirs as needed.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_bsl(self, bsl_spec: str | Path) -> str:
        """Parse a BSL specification file and return a pytest test module string.

        When the ``bsl-lang`` package is installed, the BSL file is parsed
        using its public API.  Otherwise a lightweight regex-based fallback
        extracts behaviour blocks from the raw file content.

        Parameters
        ----------
        bsl_spec:
            File-system path to the BSL specification.  Both ``str`` and
            :class:`~pathlib.Path` are accepted.

        Returns
        -------
        str
            Complete Python source for a pytest test module.

        Raises
        ------
        FileNotFoundError
            If the file at *bsl_spec* does not exist.
        ValueError
            If the file exists but no behaviour blocks can be extracted.
        """
        spec_path = Path(bsl_spec)
        if not spec_path.exists():
            raise FileNotFoundError(f"BSL spec not found: {spec_path}")

        bsl_text = spec_path.read_text(encoding="utf-8")

        if _BSL_AVAILABLE:
            behaviors = self._parse_bsl_with_library(bsl_text, spec_path)
        else:
            behaviors = _extract_behaviors_from_bsl_text(bsl_text)

        if not behaviors:
            raise ValueError(
                f"No behaviour blocks found in BSL spec: {spec_path}. "
                "Ensure the file contains 'behavior <name> { ... }' blocks."
            )

        spec_dict: dict[str, object] = {
            "agent_name": spec_path.stem,
            "source_file": str(spec_path),
            "behaviors": behaviors,
        }
        return self._render_module(spec_dict)

    def generate_from_dict(self, spec: dict[str, object]) -> str:
        """Generate a pytest test module from a plain Python dict specification.

        Parameters
        ----------
        spec:
            Dict describing the agent behaviours to scaffold.  Expected shape::

                {
                    "agent_name": str,          # optional, used in module docstring
                    "behaviors": [
                        {
                            "name": str,            # behaviour identifier
                            "given": str,           # context / precondition
                            "when": str,            # triggering action
                            "then": str,            # expected outcome
                            "expected_intent": str, # keyword for accuracy assertion
                            "dimensions": list[str],# evaluation dimensions
                            "threshold": float,     # minimum pass score
                        },
                        ...
                    ]
                }

        Returns
        -------
        str
            Complete Python source for a pytest test module.

        Raises
        ------
        ValueError
            If *spec* does not contain a ``"behaviors"`` key or the list is
            empty.
        """
        behaviors_raw = spec.get("behaviors")
        if not isinstance(behaviors_raw, list) or not behaviors_raw:
            raise ValueError(
                "spec must contain a non-empty 'behaviors' list. "
                "Example: {'behaviors': [{'name': 'my_behavior', ...}]}"
            )
        return self._render_module(spec)

    def write_to_file(self, test_content: str, output_path: Path) -> Path:
        """Write *test_content* to *output_path*.

        Parent directories are created automatically when they do not exist.

        Parameters
        ----------
        test_content:
            Python source code to write (e.g. the return value of
            :meth:`generate_from_dict`).
        output_path:
            Destination path for the generated test file.  Recommended
            convention: ``tests/test_<agent_name>_eval.py``.

        Returns
        -------
        Path
            The resolved *output_path* after writing.

        Raises
        ------
        OSError
            If the file cannot be written (permission error, disk full, etc.).
        """
        resolved = Path(output_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(test_content, encoding="utf-8")
        return resolved

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _parse_bsl_with_library(
        self,
        bsl_text: str,
        spec_path: Path,
    ) -> list[dict[str, object]]:
        """Delegate BSL parsing to the ``bsl-lang`` library.

        Falls back to the regex heuristic if the library raises an exception
        (e.g. a parse error on a partially-valid BSL file).

        Parameters
        ----------
        bsl_text:
            Raw BSL source text.
        spec_path:
            Path used only in error messages.

        Returns
        -------
        list[dict[str, object]]
            Extracted behaviour dicts.
        """
        try:
            # bsl_lang.parse() returns an AST; walk it to extract behaviors.
            ast = bsl_lang.parse(bsl_text)  # type: ignore[attr-defined]
            behaviors: list[dict[str, object]] = []
            for node in ast.behaviors:  # type: ignore[attr-defined]
                behaviors.append(
                    {
                        "name": str(node.name),
                        "given": str(getattr(node, "given", "")),
                        "when": str(getattr(node, "when", "")),
                        "then": str(getattr(node, "then", "")),
                        "expected_intent": str(getattr(node, "expected_intent", "")),
                        "dimensions": list(_DEFAULT_DIMENSIONS),
                        "threshold": _DEFAULT_THRESHOLD,
                    }
                )
            return behaviors
        except Exception:  # noqa: BLE001
            # bsl-lang parse failure — fall back to regex heuristic.
            return _extract_behaviors_from_bsl_text(bsl_text)

    def _render_module(self, spec: dict[str, object]) -> str:
        """Render the full Python test module source from *spec*.

        Parameters
        ----------
        spec:
            The specification dict (agent_name, behaviors, optional source_file).

        Returns
        -------
        str
            Complete Python source string (no trailing newlines beyond one).
        """
        agent_name = str(spec.get("agent_name", "agent"))
        source_file = spec.get("source_file")
        behaviors_raw = spec.get("behaviors", [])
        behaviors: list[dict[str, object]] = []
        if isinstance(behaviors_raw, list):
            for item in behaviors_raw:
                if isinstance(item, dict):
                    behaviors.append(item)

        source_note = f"\nGenerated from: {source_file}" if source_file else ""
        module_doc = textwrap.dedent(
            f'''\
            """Generated pytest evaluation tests for {agent_name}.{source_note}

            Each test function exercises one behaviour from the specification.
            Replace the placeholder ``agent_response(...)`` calls with your
            actual agent invocation before running the suite.

            Run with::

                pytest -v --tb=short

            To record baselines add::

                @pytest.mark.agent_eval_baseline(name="<test_name>")

            to the relevant test function.
            """'''
        )

        imports = textwrap.dedent(
            """\
            from __future__ import annotations

            import pytest

            from agent_eval.pytest_plugin import EvalContext


            # TODO: Replace this stub with your actual agent import.
            def agent_response(prompt: str) -> str:
                \"\"\"Stub agent — replace with the real implementation.\"\"\"
                raise NotImplementedError(
                    f"Replace agent_response() with your agent call. Prompt: {prompt!r}"
                )
            """
        )

        sections: list[str] = [module_doc, "", imports]

        for behavior in behaviors:
            sections.append("")
            sections.append("")
            sections.append(_render_positive_test(behavior))
            sections.append("")
            sections.append("")
            sections.append(_render_negative_test(behavior))

        # Ensure exactly one trailing newline.
        return "\n".join(sections) + "\n"
