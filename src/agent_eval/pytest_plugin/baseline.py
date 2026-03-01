"""Baseline store for pytest-agent-eval score comparisons.

Saves evaluation scores from a reference run and compares subsequent runs
against that baseline to detect regressions. Baselines are persisted as
a JSON file on disk so they survive across test sessions.

Usage example
-------------
::

    from agent_eval.pytest_plugin.baseline import BaselineStore

    store = BaselineStore(".agent_eval_baselines.json")
    store.save_baseline("test_my_agent", {"accuracy": 0.9, "safety": 1.0})

    deltas = store.compare("test_my_agent", {"accuracy": 0.85, "safety": 1.0})
    # deltas == {"accuracy": {"baseline": 0.9, "current": 0.85, "delta": -0.05},
    #            "safety": {"baseline": 1.0, "current": 1.0, "delta": 0.0}}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BaselineEntry:
    """A single baseline record for one named test.

    Parameters
    ----------
    test_name:
        The unique name of the test this baseline belongs to.
    scores:
        Mapping of dimension name to float score in [0.0, 1.0].
    recorded_at:
        ISO-8601 UTC timestamp when this baseline was captured.
    """

    test_name: str
    scores: dict[str, float]
    recorded_at: str


class BaselineStore:
    """Save, load, and compare evaluation scores against baselines.

    Baselines are persisted as a JSON file. If the file does not yet exist
    it is created lazily on the first call to :meth:`save_baseline`.

    Parameters
    ----------
    path:
        Path to the JSON file used for persistence.
        Defaults to ``.agent_eval_baselines.json`` in the current directory.
    """

    def __init__(self, path: Path | str = ".agent_eval_baselines.json") -> None:
        self._path: Path = Path(path)
        self._baselines: dict[str, BaselineEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load baselines from the JSON file, if it exists."""
        if not self._path.exists():
            return
        try:
            raw_data: object = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(raw_data, dict):
            return
        for name, entry in raw_data.items():
            if not isinstance(entry, dict):
                continue
            test_name = entry.get("test_name", name)
            scores = entry.get("scores", {})
            recorded_at = entry.get("recorded_at", "")
            if isinstance(test_name, str) and isinstance(scores, dict) and isinstance(recorded_at, str):
                self._baselines[name] = BaselineEntry(
                    test_name=test_name,
                    scores={k: float(v) for k, v in scores.items() if isinstance(v, (int, float))},
                    recorded_at=recorded_at,
                )

    def save_baseline(self, test_name: str, scores: dict[str, float]) -> None:
        """Record a new baseline for *test_name*, overwriting any prior entry.

        Parameters
        ----------
        test_name:
            Unique identifier for the test (e.g. ``"test_my_agent"``).
        scores:
            Dimension scores to persist as the reference values.
        """
        self._baselines[test_name] = BaselineEntry(
            test_name=test_name,
            scores=dict(scores),
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._persist()

    def compare(
        self,
        test_name: str,
        current_scores: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Compare *current_scores* against the stored baseline for *test_name*.

        Returns an empty dict when no baseline exists for *test_name*.

        Parameters
        ----------
        test_name:
            The test whose baseline to compare against.
        current_scores:
            The scores from the current evaluation run.

        Returns
        -------
        dict[str, dict[str, float]]
            For each metric in *current_scores*:
            ``{"baseline": <prior>, "current": <now>, "delta": <now - prior>}``.
            Metrics not present in the baseline are reported with a baseline of 0.0.
        """
        baseline = self._baselines.get(test_name)
        if baseline is None:
            return {}
        result: dict[str, dict[str, float]] = {}
        for metric, current in current_scores.items():
            prior = baseline.scores.get(metric, 0.0)
            result[metric] = {
                "baseline": prior,
                "current": current,
                "delta": current - prior,
            }
        return result

    def get_baseline(self, test_name: str) -> BaselineEntry | None:
        """Return the stored baseline for *test_name*, or ``None``."""
        return self._baselines.get(test_name)

    def _persist(self) -> None:
        """Write all baselines to disk as JSON."""
        data = {
            name: {
                "test_name": entry.test_name,
                "scores": entry.scores,
                "recorded_at": entry.recorded_at,
            }
            for name, entry in self._baselines.items()
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
