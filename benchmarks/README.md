# agent-eval Benchmarks

Reproducible benchmark suite for metric correlation, statistical power, and multi-step cascade detection.

## Quick Start

```bash
cd repos/agent-eval
python benchmarks/bench_metric_correlation.py
python benchmarks/bench_statistical_power.py
python benchmarks/bench_multi_step.py
python benchmarks/compare.py
```

## Benchmarks

### bench_metric_correlation.py

**What it measures:** Spearman rank correlation between automated evaluator scores and
synthetic human judgement scores.

**Method:**
- 10 test cases, each with correct/partial/incorrect output variants (30 total triplets).
- Each variant has a pre-assigned synthetic human score (ground truth).
- BasicAccuracyEvaluator runs in three modes (exact, fuzzy, contains).
- Spearman rho measures rank correlation between automated and human scores.

**Key metrics:**
- `spearman_rho` per evaluation mode — target: >0.7

**Competitor reference:**
- DeepEval (LLM-judge): 0.7-0.9 Spearman. Source: docs.deepeval.com (2024).
- Ragas (faithfulness/relevance): 0.65-0.85. Source: arXiv:2309.15217 (2023).

Note: BasicAccuracyEvaluator (Jaccard similarity) is a simpler metric than LLM judges.
It is expected to correlate lower than LLM-based evaluators.

---

### bench_statistical_power.py

**What it measures:** How 95% confidence interval width narrows as the number of
evaluation runs increases.

**Method:**
- Simulates N independent evaluation runs with synthetic agent noise (std_dev=0.05).
- Computes 95% CI width using stdlib-only normal approximation.
- Calculates minimum N required for 80% power at small/medium/large effect sizes.

**Key metrics:**
- `ci_width_95pct` — as N increases, this should decrease
- `min_n_for_80pct_power` — practical guidance for experiment design

**Competitor reference:**
- DeepEval: recommends N>=30 (docs.deepeval.com, 2024).
- Ragas: bootstrap N>=100 (arXiv:2309.15217, 2023).

---

### bench_multi_step.py

**What it measures:** Cascade failure detection accuracy across synthetic 3-step evaluation chains.

**Method:**
- Builds 3-step chains from the eval dataset.
- Cascade chains: step 2 input depends on step 1 output (failure propagates).
- Independent chains: step 2 input is fixed regardless of step 1.
- Injects step-1 failures and measures whether the evaluator detects cascades.

**Key metrics:**
- `cascade_detection_accuracy` — fraction of cascade failures correctly identified
- `false_cascade_rate` — rate of false cascade detection in independent chains

---

## Interpreting Results

- Results saved to `results/` as JSON files.
- Use `compare.py` to display all results in a formatted table.
- All data is synthetic (no LLM calls, no downloads, fixed seeds).
- Human scores in `datasets/eval_dataset.py` are deterministic, not random.

## Competitor Numbers (public only)

| Competitor | Metric | Value | Source |
|------------|--------|-------|--------|
| DeepEval LLM judge | Spearman rho | 0.7-0.9 | docs.deepeval.com, 2024 |
| Ragas faithfulness | Spearman rho | 0.65-0.85 | arXiv:2309.15217, 2023 |
| DeepEval | Recommended N for p-values | >=30 | docs.deepeval.com, 2024 |
| Ragas | Bootstrap CI N | >=100 | arXiv:2309.15217, 2023 |
