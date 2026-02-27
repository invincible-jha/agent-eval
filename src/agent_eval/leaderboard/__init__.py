"""Agent leaderboard system for tracking and ranking agent evaluation results.

This module provides a complete leaderboard system with:
- Pydantic v2 submission models with composite scoring
- A ranking engine for sorting and filtering submissions
- A JSON file-backed storage layer with CRUD operations

Public API
----------
All stable exports are re-exported from this module.

Example
-------
>>> from agent_eval.leaderboard import LeaderboardSubmission, RankingEngine, LeaderboardStorage
>>> submission = LeaderboardSubmission(
...     agent_name="my-agent",
...     agent_version="1.0.0",
...     framework="custom",
...     model="gpt-4o",
...     submitter="team-alpha",
...     accuracy_score=0.92,
...     safety_score=0.98,
...     cost_efficiency=0.75,
...     latency_p95_ms=1200.0,
...     consistency_score=0.88,
...     security_score=0.95,
...     benchmark_name="qa_basic",
...     benchmark_version="1.0",
...     num_runs=10,
...     total_tokens=45000,
...     total_cost_usd=0.54,
... )
>>> submission.composite_score
0.916...
"""
from __future__ import annotations

from agent_eval.leaderboard.submission import CompositeWeights, LeaderboardSubmission
from agent_eval.leaderboard.ranking import RankingEngine
from agent_eval.leaderboard.storage import LeaderboardStorage

__all__ = [
    "CompositeWeights",
    "LeaderboardSubmission",
    "RankingEngine",
    "LeaderboardStorage",
]
