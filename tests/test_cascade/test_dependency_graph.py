"""Tests for agent_eval.cascade.dependency_graph.

Tests the DependencyGraph DAG — adding steps, querying dependencies/dependents,
and topological ordering.
"""
from __future__ import annotations

import pytest

from agent_eval.cascade.dependency_graph import CycleError, DependencyGraph


class TestDependencyGraphBasics:
    """Tests for basic step registration and querying."""

    def test_empty_graph_has_no_steps(self) -> None:
        graph = DependencyGraph()
        assert graph.step_ids() == []

    def test_add_single_step(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        assert "step_1" in graph

    def test_add_step_with_dependency(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        assert "step_1" in graph
        assert "step_2" in graph

    def test_get_dependencies_no_deps(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        assert graph.get_dependencies("step_1") == []

    def test_get_dependencies_with_one_dep(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        deps = graph.get_dependencies("step_2")
        assert deps == ["step_1"]

    def test_get_dependencies_with_multiple_deps(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2")
        graph.add_step("step_3", depends_on=["step_1", "step_2"])
        deps = graph.get_dependencies("step_3")
        assert set(deps) == {"step_1", "step_2"}

    def test_get_dependencies_for_nonexistent_step(self) -> None:
        graph = DependencyGraph()
        assert graph.get_dependencies("nonexistent") == []

    def test_get_dependents_no_dependents(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        assert graph.get_dependents("step_1") == []

    def test_get_dependents_with_one_dependent(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        dependents = graph.get_dependents("step_1")
        assert dependents == ["step_2"]

    def test_get_dependents_with_multiple_dependents(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_1"])
        dependents = graph.get_dependents("step_1")
        assert set(dependents) == {"step_2", "step_3"}

    def test_duplicate_step_raises(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        with pytest.raises(ValueError, match="already registered"):
            graph.add_step("step_1")

    def test_contains_returns_false_for_missing(self) -> None:
        graph = DependencyGraph()
        assert "missing" not in graph


class TestTopologicalOrder:
    """Tests for topological ordering of steps."""

    def test_single_step_topological_order(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        order = graph.topological_order()
        assert order == ["step_1"]

    def test_linear_chain_topological_order(self) -> None:
        """step_1 → step_2 → step_3 should yield [step_1, step_2, step_3]."""
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_2"])
        order = graph.topological_order()
        # step_1 must come before step_2, step_2 before step_3
        assert order.index("step_1") < order.index("step_2")
        assert order.index("step_2") < order.index("step_3")

    def test_parallel_steps_topological_order(self) -> None:
        """Independent steps can appear in any relative order."""
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2")
        order = graph.topological_order()
        assert set(order) == {"step_1", "step_2"}

    def test_diamond_dependency_topological_order(self) -> None:
        """Diamond: step_4 depends on step_2 and step_3, both depend on step_1."""
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_1"])
        graph.add_step("step_4", depends_on=["step_2", "step_3"])
        order = graph.topological_order()
        # step_1 must be first, step_4 must be last
        assert order[0] == "step_1"
        assert order[-1] == "step_4"
        # step_2 and step_3 come between step_1 and step_4
        idx_2 = order.index("step_2")
        idx_3 = order.index("step_3")
        idx_4 = order.index("step_4")
        assert idx_2 < idx_4
        assert idx_3 < idx_4

    def test_empty_graph_topological_order(self) -> None:
        graph = DependencyGraph()
        assert graph.topological_order() == []


class TestGetAllDependents:
    """Tests for transitive dependent lookup."""

    def test_no_dependents(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        assert graph.get_all_dependents("step_1") == set()

    def test_direct_dependents_only(self) -> None:
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        all_deps = graph.get_all_dependents("step_1")
        assert all_deps == {"step_2"}

    def test_transitive_dependents(self) -> None:
        """step_1 → step_2 → step_3: all_dependents(step_1) = {step_2, step_3}."""
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_2"])
        all_deps = graph.get_all_dependents("step_1")
        assert all_deps == {"step_2", "step_3"}

    def test_diamond_transitive_dependents(self) -> None:
        """Diamond: all_dependents(step_1) = {step_2, step_3, step_4}."""
        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_1"])
        graph.add_step("step_4", depends_on=["step_2", "step_3"])
        all_deps = graph.get_all_dependents("step_1")
        assert all_deps == {"step_2", "step_3", "step_4"}
