"""CrewAI agent adapter for agent-eval.

Wraps CrewAI Crew instances as evaluatable agents.
Requires ``crewai`` to be installed.
"""
from __future__ import annotations


class CrewAIAdapter:
    """Wraps a CrewAI Crew for evaluation.

    Parameters
    ----------
    crew:
        A CrewAI Crew instance.
    name:
        Human-readable agent name.

    Raises
    ------
    ImportError
        If CrewAI is not installed.
    """

    def __init__(
        self,
        crew: object,
        name: str = "crewai-agent",
    ) -> None:
        try:
            from crewai import Crew  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CrewAI is required for CrewAIAdapter. "
                "Install with: pip install crewai"
            ) from exc

        self._crew = crew
        self._name = name

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    async def invoke(self, input_text: str) -> str:
        """Invoke the CrewAI crew with the given input.

        Parameters
        ----------
        input_text:
            The input prompt.

        Returns
        -------
        str
            The crew's output.
        """
        if hasattr(self._crew, "kickoff"):
            result = self._crew.kickoff(inputs={"input": input_text})  # type: ignore[union-attr]
        else:
            raise TypeError(
                f"Crew {type(self._crew).__name__} does not have kickoff()"
            )

        if hasattr(result, "raw"):
            return str(result.raw)  # type: ignore[union-attr]
        return str(result)
