/**
 * @aumos/agent-eval
 *
 * TypeScript client for the AumOS agent-eval evaluation framework.
 * Provides HTTP client, metric calculators, and evaluation type definitions.
 *
 * The client is now backed by @aumos/sdk-core for automatic retry,
 * typed error hierarchy, and request lifecycle events.
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

// Re-export sdk-core error hierarchy for callers that want to instanceof-check
export {
  AumosError,
  NetworkError,
  TimeoutError,
  HttpError,
  RateLimitError,
  ValidationError,
  ServerError,
  AbortError,
} from "@aumos/sdk-core";

// Re-export event emitter type for listeners attached via client.events
export type { SdkEventEmitter, SdkEventMap } from "@aumos/sdk-core";
