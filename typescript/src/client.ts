/**
 * HTTP client for the agent-eval evaluation API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createAgentEvalClient } from "@aumos/agent-eval";
 *
 * const client = createAgentEvalClient({ baseUrl: "http://localhost:8090" });
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
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-eval server. */
export interface AgentEvalClient {
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
 * @param config - Client configuration including base URL.
 * @returns An AgentEvalClient instance.
 */
export function createAgentEvalClient(
  config: AgentEvalClientConfig,
): AgentEvalClient {
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async runEvaluation(
      evalConfig: EvaluationConfig,
    ): Promise<ApiResult<BenchmarkRun>> {
      return fetchJson<BenchmarkRun>(
        `${baseUrl}/evaluations`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(evalConfig),
        },
        timeoutMs,
      );
    },

    async getBenchmarks(): Promise<ApiResult<readonly BenchmarkDescriptor[]>> {
      return fetchJson<readonly BenchmarkDescriptor[]>(
        `${baseUrl}/benchmarks`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getResults(options: {
      agentId: string;
      benchmarkId?: string;
      limit?: number;
    }): Promise<ApiResult<readonly EvaluationResult[]>> {
      const params = new URLSearchParams();
      params.set("agent_id", options.agentId);
      if (options.benchmarkId !== undefined) {
        params.set("benchmark_id", options.benchmarkId);
      }
      if (options.limit !== undefined) {
        params.set("limit", String(options.limit));
      }
      return fetchJson<readonly EvaluationResult[]>(
        `${baseUrl}/results?${params.toString()}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async compareRuns(
      baselineRunId: string,
      candidateRunId: string,
    ): Promise<ApiResult<RunComparison>> {
      const params = new URLSearchParams({
        baseline: baselineRunId,
        candidate: candidateRunId,
      });
      return fetchJson<RunComparison>(
        `${baseUrl}/runs/compare?${params.toString()}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getRunStatus(runId: string): Promise<ApiResult<BenchmarkRun>> {
      return fetchJson<BenchmarkRun>(
        `${baseUrl}/runs/${encodeURIComponent(runId)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },
  };
}

/** Re-export config type for convenience. */
export type { EvaluationConfig, EvaluationResult, BenchmarkRun, BenchmarkDescriptor, RunComparison };
