/**
 * HTTP client for the agent-eval evaluation API.
 *
 * Backed by @aumos/sdk-core's createHttpClient which provides automatic retry
 * with exponential backoff, typed error hierarchy, request lifecycle events,
 * and abort signal support.
 *
 * The public API surface is unchanged — all methods still return ApiResult<T>
 * so existing callers require no migration work.
 *
 * @example
 * ```ts
 * import { createAgentEvalClient } from "@aumos/agent-eval";
 *
 * const client = createAgentEvalClient({ baseUrl: "http://localhost:8090" });
 *
 * // Observe retry events from sdk-core
 * client.events.on("request:retry", ({ payload }) => {
 *   console.warn(`Eval API retry attempt ${payload.attempt}`);
 * });
 *
 * const run = await client.runEvaluation({
 *   eval_name: "safety-smoke",
 *   agent_id: "my-agent",
 *   benchmark_id: "aumos-safety-v1",
 *   dimensions: ["safety", "accuracy"],
 *   parameters: {},
 * });
 *
 * if (run.ok) {
 *   console.log("Run started:", run.data.run_id);
 * }
 * ```
 */

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  RateLimitError,
  ServerError,
  ValidationError,
  AumosError,
} from "@aumos/sdk-core";

import type { HttpClient, SdkEventEmitter } from "@aumos/sdk-core";

import type {
  ApiError,
  ApiResult,
  BenchmarkDescriptor,
  BenchmarkRun,
  EvaluationConfig,
  EvaluationResult,
  RunComparison,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentEvalClient. */
export interface AgentEvalClientConfig {
  /** Base URL of the agent-eval server (e.g. "http://localhost:8090"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
  /** Optional maximum retry count. Defaults to 3. */
  readonly maxRetries?: number;
}

// ---------------------------------------------------------------------------
// Internal adapter — bridges HttpClient throws into ApiResult<T>
// ---------------------------------------------------------------------------

function extractApiError(body: unknown, fallbackMessage: string): ApiError {
  if (
    body !== null &&
    typeof body === "object" &&
    "error" in body &&
    typeof (body as Record<string, unknown>)["error"] === "string"
  ) {
    const candidate = body as Partial<{ error: string; detail: string }>;
    return {
      error: candidate.error ?? fallbackMessage,
      detail: candidate.detail ?? "",
    };
  }
  return { error: fallbackMessage, detail: "" };
}

async function executeApiCall<T>(
  call: () => Promise<T>,
): Promise<ApiResult<T>> {
  try {
    const data = await call();
    return { ok: true, data };
  } catch (error: unknown) {
    if (error instanceof RateLimitError) {
      return {
        ok: false,
        error: extractApiError(error.body, "Rate limit exceeded"),
        status: 429,
      };
    }
    if (error instanceof ValidationError) {
      return {
        ok: false,
        error: {
          error: "Validation failed",
          detail: Object.entries(error.fields)
            .map(([field, messages]) => `${field}: ${messages.join(", ")}`)
            .join("; "),
        },
        status: 422,
      };
    }
    if (error instanceof ServerError) {
      return {
        ok: false,
        error: extractApiError(error.body, `Server error: HTTP ${error.statusCode}`),
        status: error.statusCode,
      };
    }
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: extractApiError(error.body, `HTTP error: ${error.statusCode}`),
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: {
          error: "Network error",
          detail: error instanceof Error ? error.message : String(error),
        },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unknown error", detail: message },
      status: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-eval server. */
export interface AgentEvalClient {
  /**
   * Typed event emitter exposed from the underlying sdk-core HttpClient.
   * Attach listeners here to observe request lifecycle, retries, and errors.
   *
   * @example
   * ```ts
   * client.events.on("request:retry", ({ payload }) => {
   *   console.warn(`Retry attempt ${payload.attempt}, delay ${payload.delayMs}ms`);
   * });
   * ```
   */
  readonly events: SdkEventEmitter;

  /**
   * Start a new evaluation run against a benchmark.
   *
   * @param config - Evaluation configuration including agent_id and benchmark_id.
   * @returns The created BenchmarkRun record (status will be "pending").
   */
  runEvaluation(config: EvaluationConfig): Promise<ApiResult<BenchmarkRun>>;

  /**
   * List all registered benchmark datasets.
   *
   * @returns Array of BenchmarkDescriptor metadata records.
   */
  getBenchmarks(): Promise<ApiResult<readonly BenchmarkDescriptor[]>>;

  /**
   * Retrieve all results for a specific agent, optionally filtered by benchmark.
   *
   * @param options - Filter parameters.
   * @returns Array of EvaluationResult records ordered by completion time descending.
   */
  getResults(options: {
    agentId: string;
    benchmarkId?: string;
    limit?: number;
  }): Promise<ApiResult<readonly EvaluationResult[]>>;

  /**
   * Compare two benchmark runs head-to-head.
   *
   * @param baselineRunId - The run used as the baseline.
   * @param candidateRunId - The run being evaluated against the baseline.
   * @returns A RunComparison with per-dimension and per-metric deltas.
   */
  compareRuns(
    baselineRunId: string,
    candidateRunId: string,
  ): Promise<ApiResult<RunComparison>>;

  /**
   * Retrieve the current status of a benchmark run.
   *
   * @param runId - The run identifier.
   * @returns The BenchmarkRun record with current status and progress.
   */
  getRunStatus(runId: string): Promise<ApiResult<BenchmarkRun>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-eval server.
 *
 * Internally uses @aumos/sdk-core's createHttpClient for automatic retry,
 * typed errors, and request lifecycle events. The public API remains identical
 * to the previous version — all methods return ApiResult<T>.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentEvalClient instance.
 */
export function createAgentEvalClient(
  config: AgentEvalClientConfig,
): AgentEvalClient {
  const httpClient: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    maxRetries: config.maxRetries ?? 3,
    defaultHeaders: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(config.headers as Record<string, string> | undefined),
    },
  });

  return {
    events: httpClient.events,

    runEvaluation(
      evalConfig: EvaluationConfig,
    ): Promise<ApiResult<BenchmarkRun>> {
      return executeApiCall(() =>
        httpClient
          .post<BenchmarkRun>("/evaluations", evalConfig)
          .then((r) => r.data),
      );
    },

    getBenchmarks(): Promise<ApiResult<readonly BenchmarkDescriptor[]>> {
      return executeApiCall(() =>
        httpClient
          .get<readonly BenchmarkDescriptor[]>("/benchmarks")
          .then((r) => r.data),
      );
    },

    getResults(options: {
      agentId: string;
      benchmarkId?: string;
      limit?: number;
    }): Promise<ApiResult<readonly EvaluationResult[]>> {
      const queryParams: Record<string, string> = {
        agent_id: options.agentId,
      };
      if (options.benchmarkId !== undefined) {
        queryParams["benchmark_id"] = options.benchmarkId;
      }
      if (options.limit !== undefined) {
        queryParams["limit"] = String(options.limit);
      }

      return executeApiCall(() =>
        httpClient
          .get<readonly EvaluationResult[]>("/results", { queryParams })
          .then((r) => r.data),
      );
    },

    compareRuns(
      baselineRunId: string,
      candidateRunId: string,
    ): Promise<ApiResult<RunComparison>> {
      return executeApiCall(() =>
        httpClient
          .get<RunComparison>("/runs/compare", {
            queryParams: {
              baseline: baselineRunId,
              candidate: candidateRunId,
            },
          })
          .then((r) => r.data),
      );
    },

    getRunStatus(runId: string): Promise<ApiResult<BenchmarkRun>> {
      return executeApiCall(() =>
        httpClient
          .get<BenchmarkRun>(`/runs/${encodeURIComponent(runId)}`)
          .then((r) => r.data),
      );
    },
  };
}
