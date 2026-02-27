"""Cascade failure analyzer for multi-step agent evaluations.

The CascadeAnalyzer walks a DependencyGraph in topological order and
classifies each failed step as either a ROOT_CAUSE, a CASCADE failure
(caused by an upstream failure), or INDEPENDENT (no dependency context).

This enables teams to see "3 actual failures caused 5 cascade failures"
rather than the misleading "8/10 steps failed".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval.cascade.dependency_graph import DependencyGraph


class FailureClassification(str, Enum):
    """How a failed step is classified in a cascade analysis.

    ROOT_CAUSE:
        This step failed and none of its dependencies also failed.
        It is the origin of a potential cascade.
    CASCADE:
        This step failed AND at least one of its direct dependencies
        also failed. The failure may be caused by the upstream failure.
    INDEPENDENT:
        This step failed but is not present in the dependency graph,
        so no causal context is available.
    """

    ROOT_CAUSE = "root_cause"
    CASCADE = "cascade"
    INDEPENDENT = "independent"


@dataclass(frozen=True)
class StepResult:
    """The outcome of a single evaluation step.

    Parameters
    ----------
    step_id:
        The unique identifier for this step.
    passed:
        True if the step produced a passing result.
    score:
        Numeric score for this step in [0.0, 1.0]. None when not applicable.
    error:
        Exception message if the step raised an error, else None.
    metadata:
        Arbitrary key-value context from the step executor.
    """

    step_id: str
    passed: bool
    score: float | None = None
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class FailedStepAnalysis:
    """Analysis of a single failed step within a cascade.

    Parameters
    ----------
    step_id:
        The step that failed.
    classification:
        How this failure is classified (ROOT_CAUSE, CASCADE, INDEPENDENT).
    failed_dependencies:
        Direct dependencies that also failed. Empty for ROOT_CAUSE and INDEPENDENT.
    caused_failures:
        Steps that transitively failed as a result of this step failing.
        Populated only for ROOT_CAUSE and CASCADE steps.
    step_result:
        The raw StepResult for this step.
    """

    step_id: str
    classification: FailureClassification
    failed_dependencies: list[str] = field(default_factory=list)
    caused_failures: list[str] = field(default_factory=list)
    step_result: StepResult | None = None


@dataclass
class CascadeAnalysis:
    """Complete cascade failure analysis of a multi-step evaluation run.

    Parameters
    ----------
    total_steps:
        Total number of steps in the evaluation (passed + failed).
    passed_steps:
        List of step IDs that passed.
    failed_steps:
        List of all step IDs that failed (root causes + cascades + independent).
    root_causes:
        Step IDs classified as ROOT_CAUSE (the true failures).
    cascade_failures:
        Step IDs classified as CASCADE (failed due to an upstream failure).
    independent_failures:
        Step IDs that failed but are not in the dependency graph.
    step_analyses:
        Detailed analysis for each failed step.
    """

    total_steps: int
    passed_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    cascade_failures: list[str] = field(default_factory=list)
    independent_failures: list[str] = field(default_factory=list)
    step_analyses: list[FailedStepAnalysis] = field(default_factory=list)

    @property
    def actual_failure_count(self) -> int:
        """Number of independent failures (root causes + independent).

        This is the "real" failure count — the number of failures that
        were not merely caused by another failure in the pipeline.
        """
        return len(self.root_causes) + len(self.independent_failures)

    @property
    def cascade_amplification(self) -> float:
        """Ratio of total failures to actual (root) failures.

        Values > 1.0 indicate that cascade effects amplified the true
        failure count. A value of 1.0 means all failures were independent.

        Returns 1.0 when there are no failures (clean run).
        """
        total_failed = len(self.failed_steps)
        actual = self.actual_failure_count
        if actual == 0:
            return 1.0
        return total_failed / actual

    def get_step_analysis(self, step_id: str) -> FailedStepAnalysis | None:
        """Return the FailedStepAnalysis for a specific step, or None."""
        for analysis in self.step_analyses:
            if analysis.step_id == step_id:
                return analysis
        return None


class CascadeAnalyzer:
    """Classifies step failures as root causes, cascades, or independent.

    The analyzer walks the dependency graph in topological order. For each
    failed step it checks whether any of its direct dependencies also failed:

    - If YES → the failure is classified as CASCADE
    - If NO (and step is in graph) → ROOT_CAUSE
    - If step is NOT in graph → INDEPENDENT

    Example
    -------
    ::

        graph = DependencyGraph()
        graph.add_step("step_1")
        graph.add_step("step_2", depends_on=["step_1"])
        graph.add_step("step_3", depends_on=["step_2"])

        results = {
            "step_1": StepResult("step_1", passed=False),
            "step_2": StepResult("step_2", passed=False),
            "step_3": StepResult("step_3", passed=True),
        }

        analyzer = CascadeAnalyzer()
        analysis = analyzer.analyze(graph, results)
        # analysis.root_causes == ["step_1"]
        # analysis.cascade_failures == ["step_2"]
    """

    def analyze(
        self,
        graph: "DependencyGraph",
        results: dict[str, StepResult],
    ) -> CascadeAnalysis:
        """Analyze step results against the dependency graph.

        Parameters
        ----------
        graph:
            The DependencyGraph describing step dependencies.
        results:
            Mapping of step_id → StepResult for all steps in the run.
            May include steps not in the graph (classified as INDEPENDENT).

        Returns
        -------
        CascadeAnalysis
            Complete classification of all failures.
        """
        # Walk graph in topological order for causal analysis
        try:
            topo_order = graph.topological_order()
        except Exception:
            # If cycle detection fails, fall back to dict key order
            topo_order = list(graph.step_ids())

        failed_step_ids: set[str] = {
            step_id for step_id, result in results.items() if not result.passed
        }
        passed_step_ids: list[str] = [
            step_id for step_id, result in results.items() if result.passed
        ]

        step_analyses: list[FailedStepAnalysis] = []
        root_causes: list[str] = []
        cascade_failures: list[str] = []
        independent_failures: list[str] = []

        # Track which steps we've analyzed (in-graph steps first)
        analyzed: set[str] = set()

        # Analyze steps that ARE in the graph, in topological order
        for step_id in topo_order:
            if step_id not in results:
                # No result for this step — skip (not evaluated)
                continue
            result = results[step_id]
            if result.passed:
                analyzed.add(step_id)
                continue

            # This step failed — classify it
            dependencies = graph.get_dependencies(step_id)
            failed_deps = [dep for dep in dependencies if dep in failed_step_ids]

            if failed_deps:
                classification = FailureClassification.CASCADE
                cascade_failures.append(step_id)
            else:
                classification = FailureClassification.ROOT_CAUSE
                root_causes.append(step_id)

            step_analyses.append(
                FailedStepAnalysis(
                    step_id=step_id,
                    classification=classification,
                    failed_dependencies=failed_deps,
                    caused_failures=[],  # populated below
                    step_result=result,
                )
            )
            analyzed.add(step_id)

        # Analyze steps NOT in the graph (independent failures)
        for step_id, result in results.items():
            if step_id in analyzed:
                continue
            if result.passed:
                passed_step_ids.append(step_id)  # shouldn't need, but guard
                continue
            # Not in graph and not passed → INDEPENDENT
            independent_failures.append(step_id)
            step_analyses.append(
                FailedStepAnalysis(
                    step_id=step_id,
                    classification=FailureClassification.INDEPENDENT,
                    failed_dependencies=[],
                    caused_failures=[],
                    step_result=result,
                )
            )

        # Populate caused_failures for root causes (transitive downstream failures)
        for analysis in step_analyses:
            if analysis.classification == FailureClassification.ROOT_CAUSE:
                all_downstream = graph.get_all_dependents(analysis.step_id)
                analysis.caused_failures = [
                    dep for dep in all_downstream if dep in failed_step_ids
                ]

        return CascadeAnalysis(
            total_steps=len(results),
            passed_steps=passed_step_ids,
            failed_steps=list(failed_step_ids),
            root_causes=root_causes,
            cascade_failures=cascade_failures,
            independent_failures=independent_failures,
            step_analyses=step_analyses,
        )
