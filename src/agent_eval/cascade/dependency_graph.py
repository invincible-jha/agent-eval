"""Directed Acyclic Graph (DAG) for tracking step dependencies.

The DependencyGraph models an evaluation pipeline as a DAG where each node
is a step (identified by a string ID) and edges represent "must complete
before" relationships. It is used by CascadeAnalyzer to walk steps in
causal order and attribute failures correctly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


class CycleError(ValueError):
    """Raised when a dependency cycle is detected in the graph."""


@dataclass
class _Node:
    """Internal representation of a single step in the DAG.

    Parameters
    ----------
    step_id:
        Unique identifier for this step.
    dependencies:
        IDs of steps that must complete before this step runs.
    """

    step_id: str
    dependencies: list[str] = field(default_factory=list)


class DependencyGraph:
    """Directed acyclic graph of evaluation step dependencies.

    Steps are added with ``add_step()``. The graph tracks:
    - Forward edges: step → steps that depend on it (dependents)
    - Backward edges: step → steps it depends on (dependencies)

    Use ``topological_order()`` to get a safe execution order.

    Example
    -------
    ::

        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_2"])
        graph.topological_order()  # ["step_1", "step_2", "step_3"]
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _Node] = {}
        # Maps step_id → list of step IDs that depend on it
        self._dependents: dict[str, list[str]] = {}

    def add_step(
        self,
        step_id: str,
        depends_on: list[str] | None = None,
    ) -> None:
        """Add a step to the graph.

        Parameters
        ----------
        step_id:
            Unique identifier for this step.
        depends_on:
            List of step IDs that must complete before this step.
            All referenced steps must already exist in the graph (or will
            be implicitly created as dependency-only stubs).

        Raises
        ------
        ValueError
            If ``step_id`` is already registered in the graph.
        """
        if step_id in self._nodes:
            raise ValueError(
                f"Step {step_id!r} is already registered in the graph."
            )

        resolved_deps: list[str] = depends_on or []

        # Ensure referenced dependency steps exist (as stubs if needed)
        for dep_id in resolved_deps:
            if dep_id not in self._nodes:
                self._nodes[dep_id] = _Node(step_id=dep_id, dependencies=[])
                self._dependents.setdefault(dep_id, [])

        self._nodes[step_id] = _Node(step_id=step_id, dependencies=list(resolved_deps))
        self._dependents.setdefault(step_id, [])

        # Register this step as a dependent of each dependency
        for dep_id in resolved_deps:
            self._dependents[dep_id].append(step_id)

    def get_dependencies(self, step_id: str) -> list[str]:
        """Return the direct dependencies of a step.

        Parameters
        ----------
        step_id:
            The step whose dependencies to retrieve.

        Returns
        -------
        list[str]
            Step IDs that must complete before this step. Empty list for
            steps with no dependencies or steps not in the graph.
        """
        if step_id not in self._nodes:
            return []
        return list(self._nodes[step_id].dependencies)

    def get_dependents(self, step_id: str) -> list[str]:
        """Return the direct dependents of a step.

        Parameters
        ----------
        step_id:
            The step whose dependents to retrieve.

        Returns
        -------
        list[str]
            Step IDs that depend on this step. Empty list for steps with
            no dependents or steps not in the graph.
        """
        return list(self._dependents.get(step_id, []))

    def get_all_dependents(self, step_id: str) -> set[str]:
        """Return the transitive closure of dependents for a step.

        Parameters
        ----------
        step_id:
            The step whose full downstream dependency set to retrieve.

        Returns
        -------
        set[str]
            All step IDs that transitively depend on ``step_id``.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([step_id])

        while queue:
            current = queue.popleft()
            for dependent in self._dependents.get(current, []):
                if dependent not in visited and dependent != step_id:
                    visited.add(dependent)
                    queue.append(dependent)

        return visited

    def step_ids(self) -> list[str]:
        """Return all step IDs in the graph (insertion order preserved).

        Returns
        -------
        list[str]
        """
        return list(self._nodes.keys())

    def __contains__(self, step_id: str) -> bool:
        """Return True if step_id is registered in the graph."""
        return step_id in self._nodes

    def topological_order(self) -> list[str]:
        """Return all steps in topological order (dependencies before dependents).

        Uses Kahn's algorithm (BFS-based). Detects cycles and raises
        ``CycleError`` if the graph is not a DAG.

        Returns
        -------
        list[str]
            Step IDs ordered so that for every edge (A → B), A appears
            before B.

        Raises
        ------
        CycleError
            If the graph contains a cycle.
        """
        if not self._nodes:
            return []

        # Compute in-degree for each node
        in_degree: dict[str, int] = {step_id: 0 for step_id in self._nodes}
        for node in self._nodes.values():
            for dep_id in node.dependencies:
                # dep_id must precede this node → dep_id has an outgoing edge TO node
                # So node's in_degree increases for each dependency
                pass

        # Rebuild in_degree correctly:
        # in_degree[X] = number of steps X depends on that have an edge to X
        # i.e., number of dependencies X has
        in_degree = {step_id: len(node.dependencies) for step_id, node in self._nodes.items()}

        # Start with all nodes that have no dependencies
        queue: deque[str] = deque(
            step_id for step_id, deg in in_degree.items() if deg == 0
        )
        order: list[str] = []

        while queue:
            step_id = queue.popleft()
            order.append(step_id)
            for dependent in self._dependents.get(step_id, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._nodes):
            cycle_nodes = [s for s in self._nodes if s not in order]
            raise CycleError(
                f"Dependency cycle detected among steps: {cycle_nodes}"
            )

        return order
