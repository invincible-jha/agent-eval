"""pytest plugin hooks for pytest-agent-eval.

This module is the entry-point loaded by pytest via the ``pytest11``
entry-point group declared in ``pyproject.toml``::

    [project.entry-points."pytest11"]
    agent_eval = "agent_eval.pytest_plugin.plugin"

Hooks implemented
-----------------
``pytest_configure``
    Registers the ``agent_eval`` marker so pytest recognises it and
    includes it in ``--markers`` output.

``pytest_collection_modifyitems``
    Iterates collected items. Any test decorated with
    ``@pytest.mark.agent_eval`` receives an automatic
    ``pytest.mark.timeout(120)`` marker to guard against runaway
    agent calls in CI.
"""
from __future__ import annotations

import pytest

from agent_eval.pytest_plugin.markers import ALL_MARKERS


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
# Fixture registration via plugin module
# ---------------------------------------------------------------------------
# pytest discovers fixtures declared in a plugin module automatically when
# the module is listed as a conftest plugin or loaded via an entry-point.
# We re-export the fixture here so pytest's collection mechanism picks it up
# without requiring users to import from fixtures.py directly.

from agent_eval.pytest_plugin.fixtures import eval_context as eval_context  # noqa: E402, F401
