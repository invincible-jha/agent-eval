"""agent_eval.conversational — Multi-turn conversation quality metrics.

Provides commodity-algorithm metrics that measure coherence, relevance,
and resolution across the turns of a multi-turn agent conversation.

Public surface
--------------
Turn
    A single conversational exchange (role + content + optional timestamp).
ConversationReport
    Aggregate evaluation result across all metrics.
ConversationEvaluator
    Runs all conversation metrics and produces a ConversationReport.

Example
-------
::

    from agent_eval.conversational import ConversationEvaluator, Turn

    turns = [
        Turn(role="user", content="What is the capital of France?"),
        Turn(role="assistant", content="The capital of France is Paris."),
        Turn(role="user", content="What is Paris famous for?"),
        Turn(role="assistant", content="Paris is famous for the Eiffel Tower."),
    ]
    evaluator = ConversationEvaluator()
    report = evaluator.evaluate(turns)
    print(report.coherence_score, report.relevance_score, report.resolution_score)
"""
from __future__ import annotations

from agent_eval.conversational.conversational_metrics import (
    ConversationEvaluator,
    ConversationReport,
    Turn,
)

__all__ = [
    "ConversationEvaluator",
    "ConversationReport",
    "Turn",
]
