"""Comparison visualiser for agent-eval benchmark results."""
from __future__ import annotations

import json
from pathlib import Path


COMPETITOR_NOTES = {
    "correlation": (
        "DeepEval: 0.7-0.9 Spearman vs human (LLM-judge metrics). "
        "Ragas: 0.65-0.85 for faithfulness/relevance. "
        "Sources: docs.deepeval.com (2024), arXiv:2309.15217 (2023). "
        "BasicAccuracyEvaluator (Jaccard) expected to correlate lower than LLM judges."
    ),
    "statistical_power": (
        "DeepEval: recommends N>=30 for reliable p-values. "
        "Ragas: uses bootstrap N>=100. "
        "Sources: docs.deepeval.com (2024), arXiv:2309.15217 (2023)."
    ),
    "cascade": (
        "DeepEval multi-step metric: bottleneck step detection. "
        "Ragas: answer correctness for QA chains (no cascade model). "
        "Sources: docs.deepeval.com (2024), arXiv:2309.15217 (2023)."
    ),
}


def _fmt_table(rows: list[tuple[str, str]], title: str) -> None:
    col1_width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    for key, value in rows:
        print(f"  {key:<{col1_width}} {value}")


def _load(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[return-value]


def display_correlation(data: dict[str, object]) -> None:
    by_mode = data.get("correlation_by_mode", {})
    rows: list[tuple[str, str]] = []
    for mode, stats in by_mode.items():
        rows.append((f"{mode} — Spearman rho", str(stats.get("spearman_rho"))))
    rows.append(("N triplets evaluated", str(data.get("n_triplets"))))
    _fmt_table(rows, "Metric Correlation Results (Spearman vs Human)")
    print(f"\n  Competitor: {COMPETITOR_NOTES['correlation']}")


def display_statistical_power(data: dict[str, object]) -> None:
    ci_results = data.get("ci_by_n_runs", {})
    rows: list[tuple[str, str]] = []
    for n_runs, stats in ci_results.items():
        rows.append(
            (f"N={n_runs} — 95% CI width", str(stats.get("ci_width_95pct")))
        )
    min_n = data.get("min_n_for_80pct_power", {})
    for label, n in min_n.items():
        rows.append((f"Min N for 80% power ({label})", str(n)))
    _fmt_table(rows, "Statistical Power Results")
    print(f"\n  Competitor: {COMPETITOR_NOTES['statistical_power']}")


def display_multi_step(data: dict[str, object]) -> None:
    rows: list[tuple[str, str]] = [
        ("Cascade detection accuracy", f"{data.get('cascade_detection_accuracy'):.2%}"),
        ("False cascade rate", f"{data.get('false_cascade_rate'):.2%}"),
        ("Cascade chains tested", str(data.get("n_cascade_chains"))),
        ("Independent chains tested", str(data.get("n_independent_chains"))),
    ]
    _fmt_table(rows, "Multi-Step Cascade Detection Results")
    print(f"\n  Competitor: {COMPETITOR_NOTES['cascade']}")


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    for fname, display_fn in [
        ("baseline.json", display_correlation),
        ("statistical_power_baseline.json", display_statistical_power),
        ("multi_step_baseline.json", display_multi_step),
    ]:
        data = _load(results_dir / fname)
        if data:
            display_fn(data)  # type: ignore[arg-type]
        else:
            print(f"No {fname} found. Run the corresponding benchmark first.")

    print("\n" + "=" * 65)
    print("  Run all benchmarks:")
    print("    python benchmarks/bench_metric_correlation.py")
    print("    python benchmarks/bench_statistical_power.py")
    print("    python benchmarks/bench_multi_step.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
