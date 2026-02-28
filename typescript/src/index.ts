/**
 * @aumos/agent-eval
 *
 * TypeScript client for the AumOS agent-eval evaluation framework.
 * Provides HTTP client, metric calculators, and evaluation type definitions.
 */

// Client and configuration
export type { AgentEvalClient, AgentEvalClientConfig } from "./client.js";
export { createAgentEvalClient } from "./client.js";

// Core types
export type {
  EvalDimension,
  MetricScore,
  EvaluationConfig,
  EvaluationResult,
  BenchmarkRun,
  BenchmarkDescriptor,
  RunComparison,
  RunStatus,
  ApiError,
  ApiResult,
} from "./types.js";

// Metric calculator
export type {
  MetricCalculator,
  PredictionPair,
  SafetyAnnotation,
  ConsistencyGroup,
  CostRecord,
} from "./metrics.js";
export { createMetricCalculator } from "./metrics.js";
