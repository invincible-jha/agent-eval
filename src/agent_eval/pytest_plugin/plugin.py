"""pytest plugin hooks for pytest-agent-eval.

This module is the entry-point loaded by pytest via the ``pytest11``
entry-point group declared in ``pyproject.toml``::

    [project.entry-points."pytest11"]
    agent_eval = "agent_eval.pytest_plugin.plugin"

Hooks implemented
-----------------
``pytest_configure``
    Registers the ``agent_eval`` and ``agent_eval_baseline`` markers so
    pytest recognises them and includes them in ``--markers`` output.

``pytest_collection_modifyitems``
    Iterates collected items. Any test decorated with
    ``@pytest.mark.agent_eval`` receives an automatic
    ``pytest.mark.timeout(120)`` marker to guard against runaway
    agent calls in CI.

``pytest_sessionfinish``
    After the full test session completes, writes an :class:`EvalReport`
    to ``agent_eval_report.json`` and ``agent_eval_report.md`` in the
    current working directory when any ``agent_eval``-marked tests ran.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.pytest_plugin.markers import ALL_MARKERS
from agent_eval.pytest_plugin.report import EvalReport

# Session-level report accumulator — populated by the eval_context fixture
# via the plugin's shared state key ``"agent_eval_report"``.
_SESSION_REPORT_KEY: str = "agent_eval_report"


# ---------------------------------------------------------------------------
# Plugin hook: configure
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers with pytest.

    Called once during collection before any tests are executed.
    Registering markers here suppresses the ``PytestUnknownMarkWarning``
    that pytest emits when it encounters an unknown marker.

    Parameters
    ----------
    config:
        The pytest ``Config`` object for the current session.
    """
    for marker_spec in ALL_MARKERS:
        config.addinivalue_line("markers", marker_spec.description)
    # Initialise the session-level report accumulator.
    config._agent_eval_report = EvalReport()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Plugin hook: collection
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify collected test items that carry the ``agent_eval`` marker.

    For every test decorated with ``@pytest.mark.agent_eval``:

    1. Injects ``pytest.mark.timeout(120)`` so that slow agent calls
       cannot block CI indefinitely.

    The ``config`` parameter is required by pytest's hook signature even
    though it is not used here; omitting it would cause pytest to skip
    the hook.

    Parameters
    ----------
    config:
        The pytest ``Config`` object for the current session (unused).
    items:
        The list of collected :class:`pytest.Item` objects. Modified
        in-place.
    """
    for item in items:
        if item.get_closest_marker("agent_eval"):
            item.add_marker(pytest.mark.timeout(120))


# ---------------------------------------------------------------------------
# Plugin hook: session finish
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the aggregated evaluation report after the session ends.

    When at least one ``agent_eval``-marked test ran during the session,
    this hook writes:

    - ``agent_eval_report.json`` — full structured report.
    - ``agent_eval_report.md``  — Markdown summary table.

    Both files are written to the current working directory. The hook is
    a no-op when no evaluation tests were executed.

    Parameters
    ----------
    session:
        The pytest :class:`~pytest.Session` that just finished.
    exitstatus:
        Integer exit code of the test run (unused but required by the hook
        signature).
    """
    report: EvalReport | None = getattr(session.config, "_agent_eval_report", None)
    if report is None or report.total_tests == 0:
        return

    output_dir = Path.cwd()
    try:
        (output_dir / "agent_eval_report.json").write_text(
            report.to_json(), encoding="utf-8"
        )
        (output_dir / "agent_eval_report.md").write_text(
            report.to_markdown(), encoding="utf-8"
        )
    except OSError:
        # Non-fatal — report writing failure should not affect test results.
        pass


# ---------------------------------------------------------------------------
# Fixture registration via plugin module
# ---------------------------------------------------------------------------
# pytest discovers fixtures declared in a plugin module automatically when
# the module is listed as a conftest plugin or loaded via an entry-point.
# We re-export the fixture here so pytest's collection mechanism picks it up
# without requiring users to import from fixtures.py directly.

from agent_eval.pytest_plugin.fixtures import eval_context as eval_context  # noqa: E402, F401
