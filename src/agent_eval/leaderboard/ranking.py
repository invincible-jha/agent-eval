"""Ranking engine for leaderboard submissions.

The RankingEngine provides pure, side-effect-free functions for sorting,
filtering, and slicing lists of :class:`~agent_eval.leaderboard.submission.LeaderboardSubmission`
objects.  All methods are class methods so the engine can be used without
instantiation.

Example
-------
>>> from agent_eval.leaderboard.ranking import RankingEngine
>>> ranked = RankingEngine.rank(submissions, sort_by="composite_score")
>>> top3 = RankingEngine.top_n(submissions, n=3, sort_by="composite_score")
>>> lc_only = RankingEngine.filter_by_framework(submissions, "langchain")
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from agent_eval.leaderboard.submission import LeaderboardSubmission

# Fields on LeaderboardSubmission that can be used as sort keys.
_SORTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "composite_score",
        "accuracy_score",
        "safety_score",
        "cost_efficiency",
        "consistency_score",
        "security_score",
        "latency_p95_ms",
        "total_cost_usd",
        "total_tokens",
        "num_runs",
        "submitted_at",
        "agent_name",
        "framework",
        "model",
    }
)


class RankingEngine:
    """Pure-function ranking, filtering, and slicing over submission lists.

    All public methods are static so no instance is required.

    Methods
    -------
    rank(submissions, sort_by, ascending)
        Return a sorted copy of *submissions*.
    top_n(submissions, n, sort_by)
        Return the top *n* submissions ranked by *sort_by* descending.
    filter_by_framework(submissions, framework)
        Return submissions that match the given framework name (case-insensitive).
    filter_by_benchmark(submissions, benchmark_name)
        Return submissions that match the given benchmark name (case-insensitive).
    """

    @staticmethod
    def rank(
        submissions: list[LeaderboardSubmission],
        sort_by: str = "composite_score",
        ascending: bool = False,
    ) -> list[LeaderboardSubmission]:
        """Return a sorted copy of the submission list.

        Parameters
        ----------
        submissions:
            The list of submissions to sort. Not mutated.
        sort_by:
            Attribute name to sort by. Must be a field on
            :class:`~agent_eval.leaderboard.submission.LeaderboardSubmission`.
        ascending:
            When ``True`` sort in ascending order (lowest first).  Default is
            ``False`` (highest first), which is appropriate for score fields.

        Returns
        -------
        list[LeaderboardSubmission]
            A new list sorted according to *sort_by* and *ascending*.

        Raises
        ------
        ValueError
            If *sort_by* is not a recognised sortable field name.
        """
        if sort_by not in _SORTABLE_FIELDS:
            raise ValueError(
                f"Cannot sort by {sort_by!r}. "
                f"Valid fields: {sorted(_SORTABLE_FIELDS)}"
            )
        return sorted(
            submissions,
            key=lambda sub: getattr(sub, sort_by),
            reverse=not ascending,
        )

    @staticmethod
    def top_n(
        submissions: list[LeaderboardSubmission],
        n: int,
        sort_by: str = "composite_score",
    ) -> list[LeaderboardSubmission]:
        """Return the *n* best submissions sorted descending by *sort_by*.

        Parameters
        ----------
        submissions:
            The pool of submissions to select from.
        n:
            Maximum number of submissions to return. If ``n`` exceeds the
            length of *submissions* the full sorted list is returned.
        sort_by:
            Attribute name used for ranking (higher is better).

        Returns
        -------
        list[LeaderboardSubmission]
            At most *n* submissions with the highest *sort_by* values.

        Raises
        ------
        ValueError
            If *n* is not a positive integer, or *sort_by* is invalid.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        ranked = RankingEngine.rank(submissions, sort_by=sort_by, ascending=False)
        return ranked[:n]

    @staticmethod
    def filter_by_framework(
        submissions: list[LeaderboardSubmission],
        framework: str,
    ) -> list[LeaderboardSubmission]:
        """Return all submissions produced by the given framework.

        Comparison is case-insensitive.

        Parameters
        ----------
        submissions:
            The pool of submissions to filter.
        framework:
            Framework name to match (e.g. ``"langchain"``).

        Returns
        -------
        list[LeaderboardSubmission]
            Submissions whose ``framework`` field matches (case-insensitive).
        """
        needle = framework.lower()
        return [s for s in submissions if s.framework.lower() == needle]

    @staticmethod
    def filter_by_benchmark(
        submissions: list[LeaderboardSubmission],
        benchmark_name: str,
    ) -> list[LeaderboardSubmission]:
        """Return all submissions for the given benchmark.

        Comparison is case-insensitive.

        Parameters
        ----------
        submissions:
            The pool of submissions to filter.
        benchmark_name:
            Benchmark name to match (e.g. ``"qa_basic"``).

        Returns
        -------
        list[LeaderboardSubmission]
            Submissions whose ``benchmark_name`` field matches (case-insensitive).
        """
        needle = benchmark_name.lower()
        return [s for s in submissions if s.benchmark_name.lower() == needle]
