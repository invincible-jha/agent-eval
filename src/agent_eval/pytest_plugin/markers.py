"""Custom pytest marker definitions for pytest-agent-eval.

Centralises marker metadata so that both the plugin hooks and user-facing
documentation have a single source of truth.

Marker reference
----------------
``@pytest.mark.agent_eval(dimensions=..., runs=..., threshold=...)``

    Mark a test as an agent evaluation test. The plugin hooks use this
    marker to apply specialised behaviour (e.g. timeout injection).

    Parameters (all optional keyword arguments on the marker):

    dimensions : list[str]
        Which quality dimensions to evaluate. Default is all: accuracy,
        safety, cost, latency.
    runs : int
        Number of times to execute the test to measure consistency.
        Default 1. Values > 1 trigger multi-run collection (not yet
        implemented in the open-source version).
    threshold : float
        Minimum aggregate score required to pass. Default 0.8.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarkerSpec:
    """Static specification for a single custom pytest marker.

    Parameters
    ----------
    name:
        The marker name as it appears in ``@pytest.mark.<name>``.
    description:
        Full description shown in ``pytest --markers``.
    """

    name: str
    description: str


# ---------------------------------------------------------------------------
# Registered markers
# ---------------------------------------------------------------------------

AGENT_EVAL_MARKER: MarkerSpec = MarkerSpec(
    name="agent_eval",
    description=(
        "agent_eval(dimensions, runs, threshold): "
        "Mark a test as an agent evaluation test. "
        "dimensions — list of quality dimensions to evaluate "
        "(accuracy, safety, cost, latency); "
        "runs — number of evaluation runs for consistency checks (default 1); "
        "threshold — minimum aggregate score to pass (default 0.8)."
    ),
)

# Ordered list of all markers registered by this plugin.
ALL_MARKERS: tuple[MarkerSpec, ...] = (AGENT_EVAL_MARKER,)


@dataclass(frozen=True)
class AgentEvalMarkerArgs:
    """Parsed arguments from an ``@pytest.mark.agent_eval(...)`` marker.

    Used by hooks that need to inspect marker parameters without
    re-parsing the raw ``marker.kwargs`` dict each time.

    Parameters
    ----------
    dimensions:
        Quality dimensions requested. Defaults to all four built-ins.
    runs:
        Number of evaluation runs requested. Default 1.
    threshold:
        Minimum aggregate pass score. Default 0.8.
    """

    dimensions: tuple[str, ...] = field(
        default_factory=lambda: ("accuracy", "safety", "cost", "latency")  # type: ignore[return-value]
    )
    runs: int = 1
    threshold: float = 0.8

    @classmethod
    def from_marker(cls, marker: object) -> AgentEvalMarkerArgs:
        """Parse an ``AgentEvalMarkerArgs`` from a pytest ``Mark`` object.

        Falls back to defaults for any parameter that is absent or None.

        Parameters
        ----------
        marker:
            A ``pytest.Mark`` instance obtained via
            ``item.get_closest_marker("agent_eval")``.

        Returns
        -------
        AgentEvalMarkerArgs
            Parsed args, with defaults applied for missing fields.
        """
        # pytest.Mark stores keyword args in .kwargs
        kwargs: dict[str, object] = getattr(marker, "kwargs", {}) or {}

        raw_dimensions = kwargs.get("dimensions")
        if isinstance(raw_dimensions, (list, tuple)):
            dimensions: tuple[str, ...] = tuple(str(d) for d in raw_dimensions)
        else:
            dimensions = ("accuracy", "safety", "cost", "latency")

        raw_runs = kwargs.get("runs")
        runs: int = int(raw_runs) if isinstance(raw_runs, int) else 1

        raw_threshold = kwargs.get("threshold")
        threshold: float = float(raw_threshold) if isinstance(raw_threshold, (int, float)) else 0.8

        return cls(dimensions=dimensions, runs=runs, threshold=threshold)
