/**
 * TypeScript interfaces for the agent-eval evaluation framework.
 *
 * Mirrors the Pydantic models defined in:
 *   agent_eval.schemas.evaluation
 *   agent_eval.schemas.benchmarks
 *   agent_eval.schemas.metrics
 *
 * All interfaces use readonly fields to match Python's frozen Pydantic models.
 */

// ---------------------------------------------------------------------------
// Evaluation dimension
// ---------------------------------------------------------------------------

/**
 * The dimension along which an agent is evaluated.
 * Maps to EvalDimension enum in Python.
 */
export type EvalDimension = "accuracy" | "safety" | "consistency" | "cost";

// ---------------------------------------------------------------------------
// Individual metric score
// ---------------------------------------------------------------------------

/** A scored value for a single metric within one evaluation dimension. */
export interface MetricScore {
  /** Canonical metric identifier (e.g. "exact_match", "toxicity_rate"). */
  readonly metric_id: string;
  /** Human-readable metric name. */
  readonly metric_name: string;
  /** Evaluation dimension this metric belongs to. */
  readonly dimension: EvalDimension;
  /** Numeric score in the range [0, 1] unless noted otherwise. */
  readonly score: number;
  /** Optional lower bound of the confidence interval. */
  readonly ci_lower?: number;
  /** Optional upper bound of the confidence interval. */
  readonly ci_upper?: number;
  /** Arbitrary extra metadata attached by the metric implementation. */
  readonly metadata: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Evaluation configuration
// ---------------------------------------------------------------------------

/** Configuration required to start an evaluation run. */
export interface EvaluationConfig {
  /** Human-readable name for this evaluation run. */
  readonly eval_name: string;
  /** Identifier of the agent under evaluation. */
  readonly agent_id: string;
  /** Benchmark dataset identifier (e.g. "aumos-safety-v1"). */
  readonly benchmark_id: string;
  /** Subset of EvalDimensions to evaluate (defaults to all when absent). */
  readonly dimensions?: readonly EvalDimension[];
  /** Maximum number of evaluation samples to process. */
  readonly sample_limit?: number;
  /** Seed for reproducible sampling. */
  readonly random_seed?: number;
  /** Arbitrary evaluation parameters forwarded to the runner. */
  readonly parameters: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Evaluation result
// ---------------------------------------------------------------------------

/** Full result of a completed evaluation run. */
export interface EvaluationResult {
  /** Unique identifier for this evaluation result record. */
  readonly result_id: string;
  /** The run that produced this result. */
  readonly run_id: string;
  /** Agent that was evaluated. */
  readonly agent_id: string;
  /** Benchmark dataset used. */
  readonly benchmark_id: string;
  /** ISO-8601 UTC timestamp of when the evaluation completed. */
  readonly completed_at: string;
  /** Total wall-clock duration in milliseconds. */
  readonly duration_ms: number;
  /** Per-metric scores indexed by metric_id. */
  readonly metric_scores: readonly MetricScore[];
  /** Aggregate score per dimension (0–1 range). */
  readonly dimension_scores: Readonly<Record<EvalDimension, number>>;
  /** Overall composite score across all dimensions (0–1 range). */
  readonly composite_score: number;
  /** Total number of evaluation samples processed. */
  readonly sample_count: number;
  /** Number of samples that produced an error. */
  readonly error_count: number;
  /** Arbitrary metadata attached to this result. */
  readonly metadata: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Benchmark run
// ---------------------------------------------------------------------------

/** Status of an in-progress or completed benchmark run. */
export type RunStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

/** Represents one execution of a benchmark against an agent. */
export interface BenchmarkRun {
  /** Unique run identifier. */
  readonly run_id: string;
  /** Configuration used to start this run. */
  readonly config: EvaluationConfig;
  /** Current lifecycle status. */
  readonly status: RunStatus;
  /** ISO-8601 UTC timestamp when the run was queued. */
  readonly created_at: string;
  /** ISO-8601 UTC timestamp when the run started executing (null if pending). */
  readonly started_at: string | null;
  /** ISO-8601 UTC timestamp when the run finished (null if not yet complete). */
  readonly finished_at: string | null;
  /** Percentage of samples processed so far [0, 100]. */
  readonly progress_pct: number;
  /** Human-readable error message if status is "failed". */
  readonly error_message: string | null;
  /** Result record once the run reaches "completed" status. */
  readonly result: EvaluationResult | null;
}

// ---------------------------------------------------------------------------
// Benchmark descriptor
// ---------------------------------------------------------------------------

/** Metadata about a registered benchmark dataset. */
export interface BenchmarkDescriptor {
  /** Unique benchmark identifier. */
  readonly benchmark_id: string;
  /** Human-readable name. */
  readonly name: string;
  /** Short description of what the benchmark measures. */
  readonly description: string;
  /** Dimensions covered by this benchmark. */
  readonly dimensions: readonly EvalDimension[];
  /** Total number of samples in the dataset. */
  readonly sample_count: number;
  /** Schema version of the benchmark format. */
  readonly version: string;
}

// ---------------------------------------------------------------------------
// Run comparison
// ---------------------------------------------------------------------------

/** Head-to-head comparison of two benchmark runs. */
export interface RunComparison {
  /** First run identifier (baseline). */
  readonly baseline_run_id: string;
  /** Second run identifier (candidate). */
  readonly candidate_run_id: string;
  /** Delta of composite scores (candidate − baseline). */
  readonly composite_delta: number;
  /** Per-dimension deltas (candidate − baseline). */
  readonly dimension_deltas: Readonly<Record<EvalDimension, number>>;
  /** Per-metric deltas indexed by metric_id. */
  readonly metric_deltas: Readonly<Record<string, number>>;
  /** Whether the candidate is statistically significantly better overall. */
  readonly significant_improvement: boolean;
}

// ---------------------------------------------------------------------------
// API result wrapper (shared pattern)
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-eval API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
