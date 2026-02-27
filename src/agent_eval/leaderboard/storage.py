"""JSON file-backed storage for leaderboard submissions.

The :class:`LeaderboardStorage` class provides a simple CRUD interface backed
by a single JSON file.  It is intentionally lightweight — for production use
you would swap this for a database-backed implementation behind the same
interface.

All public methods are thread-safe with respect to the in-process lock, but
concurrent writes from *different processes* are not protected.

File format
-----------
The file is a JSON array of serialised
:class:`~agent_eval.leaderboard.submission.LeaderboardSubmission` objects::

    [
      { "agent_name": "...", "agent_version": "...", ... },
      ...
    ]

Example
-------
>>> from pathlib import Path
>>> from agent_eval.leaderboard.storage import LeaderboardStorage
>>> store = LeaderboardStorage(storage_path=Path("/tmp/lb.json"))
>>> store.save(submission)
>>> all_subs = store.load_all()
>>> store.export_json(Path("/tmp/export.json"))
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Final

from agent_eval.leaderboard.submission import LeaderboardSubmission

# Sentinel used to represent an absent storage path (in-memory mode).
_NO_FILE: Final[str] = ":memory:"


class LeaderboardStorage:
    """Persistent leaderboard storage backed by a JSON file.

    Parameters
    ----------
    storage_path:
        Path to the JSON file used for persistence.  If the file does not
        exist it is created on the first :meth:`save` call.  Pass
        ``None`` to operate in *in-memory* mode (data is never written to
        disk, useful for testing).

    Attributes
    ----------
    storage_path:
        The resolved path (or ``None`` for in-memory mode).
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path: Path | None = storage_path
        self._lock: threading.Lock = threading.Lock()
        self._memory: list[LeaderboardSubmission] = []

        if storage_path is not None and storage_path.exists():
            self._memory = self._read_file(storage_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, submission: LeaderboardSubmission) -> None:
        """Persist a submission, replacing any existing entry with the same
        agent_name + agent_version combination.

        If a submission with the same ``agent_name`` *and* ``agent_version``
        already exists, it is overwritten in-place.  Otherwise the new
        submission is appended.

        Parameters
        ----------
        submission:
            The :class:`~agent_eval.leaderboard.submission.LeaderboardSubmission`
            to persist.
        """
        with self._lock:
            updated = False
            for index, existing in enumerate(self._memory):
                if (
                    existing.agent_name == submission.agent_name
                    and existing.agent_version == submission.agent_version
                ):
                    self._memory[index] = submission
                    updated = True
                    break
            if not updated:
                self._memory.append(submission)
            self._flush()

    def load_all(self) -> list[LeaderboardSubmission]:
        """Return a shallow copy of all stored submissions.

        Returns
        -------
        list[LeaderboardSubmission]
            All submissions currently in storage, in insertion order.
        """
        with self._lock:
            return list(self._memory)

    def load_by_agent(self, agent_name: str) -> list[LeaderboardSubmission]:
        """Return all submissions for a given agent name.

        Parameters
        ----------
        agent_name:
            The ``agent_name`` value to filter by. Comparison is
            case-sensitive.

        Returns
        -------
        list[LeaderboardSubmission]
            Submissions whose ``agent_name`` exactly matches *agent_name*.
        """
        with self._lock:
            return [s for s in self._memory if s.agent_name == agent_name]

    def delete(self, agent_name: str, agent_version: str) -> bool:
        """Delete the submission matching *agent_name* + *agent_version*.

        Parameters
        ----------
        agent_name:
            Name of the agent to delete.
        agent_version:
            Version of the agent to delete.

        Returns
        -------
        bool
            ``True`` if a submission was found and removed, ``False`` if no
            matching entry existed.
        """
        with self._lock:
            before = len(self._memory)
            self._memory = [
                s
                for s in self._memory
                if not (
                    s.agent_name == agent_name and s.agent_version == agent_version
                )
            ]
            removed = len(self._memory) < before
            if removed:
                self._flush()
            return removed

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_json(self, path: Path) -> None:
        """Write all submissions to *path* as a JSON array.

        Parameters
        ----------
        path:
            Destination file path. Parent directories are created if they
            do not exist.  Any existing file at *path* is overwritten.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [s.model_dump(mode="json") for s in self._memory]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def import_json(self, path: Path) -> int:
        """Import submissions from a JSON file, merging into existing data.

        Each imported submission is passed through :meth:`save`, which means
        duplicates (same agent_name + agent_version) are overwritten rather
        than duplicated.

        Parameters
        ----------
        path:
            Source JSON file. Must be a JSON array of submission objects
            compatible with :class:`~agent_eval.leaderboard.submission.LeaderboardSubmission`.

        Returns
        -------
        int
            Number of submissions imported.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file does not contain a valid JSON array.
        """
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON array, got {type(raw).__name__}")
        imported = 0
        for item in raw:
            submission = LeaderboardSubmission.model_validate(item)
            self.save(submission)
            imported += 1
        return imported

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Write the in-memory list to the backing file (if one is set).

        Must be called while ``self._lock`` is held.
        """
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.model_dump(mode="json") for s in self._memory]
        self.storage_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def _read_file(path: Path) -> list[LeaderboardSubmission]:
        """Parse the JSON file at *path* and return a list of submissions.

        Parameters
        ----------
        path:
            The file to read. Must exist and contain a JSON array.

        Returns
        -------
        list[LeaderboardSubmission]
            Parsed and validated submissions.
        """
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError(
                f"Leaderboard file must contain a JSON array, got {type(raw).__name__}"
            )
        return [LeaderboardSubmission.model_validate(item) for item in raw]
