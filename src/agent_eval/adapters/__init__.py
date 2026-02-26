"""Agent framework adapters for agent-eval.

Adapters wrap various agent frameworks (LangChain, CrewAI, AutoGen,
OpenAI Agents SDK) and generic interfaces (callable, HTTP) as
AgentUnderTest instances for evaluation.
"""
from __future__ import annotations

from agent_eval.adapters.callable import CallableAdapter
from agent_eval.adapters.http import HTTPAdapter

__all__ = [
    "CallableAdapter",
    "HTTPAdapter",
]

# Framework adapters are optional imports — available only when
# the corresponding framework is installed.
