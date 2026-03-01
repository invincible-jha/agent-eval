/**
 * Tests for @aumos/agent-eval client.
 *
 * Covers:
 * - runEvaluation, getBenchmarks, getResults, compareRuns, getRunStatus
 * - sdk-core error hierarchy: HttpError, NetworkError, RateLimitError, ServerError, ValidationError
 * - Request lifecycle events and retry behavior
 * - Query parameter construction for all filter options
 * - URL encoding of path parameters
 * - Backward compatibility of ApiResult<T> shape
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { createAgentEvalClient } from "../src/client.js";
import type {
  BenchmarkRun,
  BenchmarkDescriptor,
  EvaluationResult,
  RunComparison,
  EvaluationConfig,
} from "../src/types.js";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function makeSuccessResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    headers: {
      get: (name: string) =>
        name.toLowerCase() === "content-type" ? "application/json" : null,
      forEach: (cb: (v: string, k: string) => void) => {
        cb("application/json", "content-type");
      },
    },
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

function makeErrorResponse(
  status: number,
  body: unknown,
  extraHeaders: Record<string, string> = {},
) {
  return {
    ok: false,
    status,
    statusText: `Error ${status}`,
    headers: {
      get: (name: string) => {
        if (name.toLowerCase() === "content-type") return "application/json";
        return extraHeaders[name.toLowerCase()] ?? null;
      },
      forEach: (cb: (v: string, k: string) => void) => {
        cb("application/json", "content-type");
        for (const [k, v] of Object.entries(extraHeaders)) {
          cb(v, k);
        }
      },
    },
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

const BASE_URL = "http://localhost:18090";

const SAMPLE_EVAL_CONFIG: EvaluationConfig = {
  eval_name: "safety-smoke",
  agent_id: "agent-001",
  benchmark_id: "aumos-safety-v1",
  dimensions: ["safety", "accuracy"],
  parameters: {},
};

const SAMPLE_BENCHMARK_RUN: BenchmarkRun = {
  run_id: "run-001",
  config: SAMPLE_EVAL_CONFIG,
  status: "pending",
  created_at: "2024-01-01T00:00:00Z",
  started_at: null,
  finished_at: null,
  progress_pct: 0,
  error_message: null,
  result: null,
};

const SAMPLE_BENCHMARK_DESCRIPTOR: BenchmarkDescriptor = {
  benchmark_id: "aumos-safety-v1",
  name: "AumOS Safety Benchmark v1",
  description: "Evaluates agent safety across adversarial prompts.",
  dimensions: ["safety"],
  sample_count: 500,
  version: "1.0.0",
};

const SAMPLE_EVAL_RESULT: EvaluationResult = {
  result_id: "result-001",
  run_id: "run-001",
  agent_id: "agent-001",
  benchmark_id: "aumos-safety-v1",
  completed_at: "2024-01-01T01:00:00Z",
  duration_ms: 3600000,
  metric_scores: [],
  dimension_scores: { accuracy: 0.9, safety: 0.95, consistency: 0.85, cost: 0.7 },
  composite_score: 0.85,
  sample_count: 500,
  error_count: 0,
  metadata: {},
};

const SAMPLE_RUN_COMPARISON: RunComparison = {
  baseline_run_id: "run-001",
  candidate_run_id: "run-002",
  composite_delta: 0.05,
  dimension_deltas: { accuracy: 0.02, safety: 0.08, consistency: 0.03, cost: -0.01 },
  metric_deltas: { exact_match: 0.02, toxicity_rate: -0.03 },
  significant_improvement: true,
};

// ---------------------------------------------------------------------------
// runEvaluation()
// ---------------------------------------------------------------------------

describe("createAgentEvalClient — runEvaluation()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with BenchmarkRun on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_BENCHMARK_RUN)));

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.runEvaluation(SAMPLE_EVAL_CONFIG);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.run_id).toBe("run-001");
      expect(result.data.status).toBe("pending");
      expect(result.data.progress_pct).toBe(0);
    }
  });

  it("returns ok:false with status 422 on validation error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(422, {
          fields: {
            agent_id: ["Agent ID cannot be empty"],
            benchmark_id: ["Unknown benchmark"],
          },
        }),
      ),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.runEvaluation({
      ...SAMPLE_EVAL_CONFIG,
      agent_id: "",
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(422);
      expect(result.error.error).toBe("Validation failed");
      expect(result.error.detail).toContain("agent_id");
    }
  });

  it("POSTs to /evaluations endpoint", async () => {
    let capturedUrl = "";
    let capturedMethod = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string, init?: RequestInit) => {
        capturedUrl = url;
        capturedMethod = init?.method ?? "GET";
        return Promise.resolve(makeSuccessResponse(SAMPLE_BENCHMARK_RUN));
      }),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.runEvaluation(SAMPLE_EVAL_CONFIG);

    expect(capturedUrl).toContain("/evaluations");
    expect(capturedMethod).toBe("POST");
  });
});

// ---------------------------------------------------------------------------
// getBenchmarks()
// ---------------------------------------------------------------------------

describe("createAgentEvalClient — getBenchmarks()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with BenchmarkDescriptor array on success", async () => {
    const benchmarks = [SAMPLE_BENCHMARK_DESCRIPTOR];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(benchmarks)));

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getBenchmarks();

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data).toHaveLength(1);
      expect(result.data[0]?.benchmark_id).toBe("aumos-safety-v1");
      expect(result.data[0]?.sample_count).toBe(500);
    }
  });

  it("returns ok:true with empty array when no benchmarks exist", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse([])));

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getBenchmarks();

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data).toHaveLength(0);
    }
  });

  it("GETs from /benchmarks endpoint", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse([]));
      }),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getBenchmarks();

    expect(capturedUrl).toContain("/benchmarks");
    expect(capturedUrl).not.toContain("?");
  });
});

// ---------------------------------------------------------------------------
// getResults()
// ---------------------------------------------------------------------------

describe("createAgentEvalClient — getResults()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with EvaluationResult array on success", async () => {
    const results = [SAMPLE_EVAL_RESULT];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(results)));

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getResults({ agentId: "agent-001" });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data).toHaveLength(1);
      expect(result.data[0]?.composite_score).toBe(0.85);
    }
  });

  it("passes agentId as required agent_id query param", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse([]));
      }),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getResults({ agentId: "agent-001" });

    expect(capturedUrl).toContain("agent_id=agent-001");
  });

  it("passes optional benchmarkId and limit as query params", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse([]));
      }),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getResults({
      agentId: "agent-001",
      benchmarkId: "aumos-safety-v1",
      limit: 10,
    });

    expect(capturedUrl).toContain("agent_id=agent-001");
    expect(capturedUrl).toContain("benchmark_id=aumos-safety-v1");
    expect(capturedUrl).toContain("limit=10");
  });

  it("omits benchmarkId and limit when not provided", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse([]));
      }),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getResults({ agentId: "agent-001" });

    expect(capturedUrl).not.toContain("benchmark_id");
    expect(capturedUrl).not.toContain("limit=");
  });
});

// ---------------------------------------------------------------------------
// compareRuns()
// ---------------------------------------------------------------------------

describe("createAgentEvalClient — compareRuns()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with RunComparison on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_RUN_COMPARISON)));

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.compareRuns("run-001", "run-002");

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.baseline_run_id).toBe("run-001");
      expect(result.data.candidate_run_id).toBe("run-002");
      expect(result.data.composite_delta).toBe(0.05);
      expect(result.data.significant_improvement).toBe(true);
    }
  });

  it("passes baseline and candidate as query params", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_RUN_COMPARISON));
      }),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.compareRuns("run-001", "run-002");

    expect(capturedUrl).toContain("baseline=run-001");
    expect(capturedUrl).toContain("candidate=run-002");
    expect(capturedUrl).toContain("/runs/compare");
  });

  it("returns ok:false with status 404 when a run does not exist", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(404, { error: "Run not found", detail: "run-999 does not exist" }),
      ),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.compareRuns("run-999", "run-002");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(404);
      expect(result.error.error).toBe("Run not found");
    }
  });
});

// ---------------------------------------------------------------------------
// getRunStatus()
// ---------------------------------------------------------------------------

describe("createAgentEvalClient — getRunStatus()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with BenchmarkRun on success", async () => {
    const runningRun: BenchmarkRun = {
      ...SAMPLE_BENCHMARK_RUN,
      status: "running",
      started_at: "2024-01-01T00:01:00Z",
      progress_pct: 42,
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(runningRun)));

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getRunStatus("run-001");

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.run_id).toBe("run-001");
      expect(result.data.status).toBe("running");
      expect(result.data.progress_pct).toBe(42);
    }
  });

  it("URL-encodes the runId in the path", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_BENCHMARK_RUN));
      }),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getRunStatus("run/with-slash");

    expect(capturedUrl).toContain(encodeURIComponent("run/with-slash"));
  });

  it("returns ok:false with status 404 for unknown run", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(404, { error: "Run not found", detail: "" }),
      ),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getRunStatus("nonexistent-run");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(404);
      expect(result.error.error).toBe("Run not found");
    }
  });
});

// ---------------------------------------------------------------------------
// sdk-core error handling integration
// ---------------------------------------------------------------------------

describe("createAgentEvalClient — sdk-core error handling", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:false with status 429 on rate limit", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(429, { error: "Rate limit exceeded", detail: "" }, {
          "retry-after": "60",
        }),
      ),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getBenchmarks();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(429);
      expect(result.error.error).toBe("Rate limit exceeded");
    }
  });

  it("returns ok:false with status 500 on internal server error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(500, { error: "Internal error", detail: "DB unavailable" }),
      ),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.runEvaluation(SAMPLE_EVAL_CONFIG);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(500);
    }
  });

  it("returns ok:false with status 503 on service unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(503, { error: "Service unavailable", detail: "Maintenance" }),
      ),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getResults({ agentId: "agent-001" });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(503);
    }
  });

  it("returns ok:false with status 0 on network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getBenchmarks();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(0);
      expect(result.error.error).toMatch(/network error/i);
    }
  });

  it("exposes SdkEventEmitter on client.events", () => {
    const client = createAgentEvalClient({ baseUrl: BASE_URL });
    expect(typeof client.events.on).toBe("function");
    expect(typeof client.events.off).toBe("function");
    expect(typeof client.events.emit).toBe("function");
  });

  it("fires request:start and request:end on success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeSuccessResponse([SAMPLE_BENCHMARK_DESCRIPTOR])),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const fired: string[] = [];

    client.events.on("request:start", () => fired.push("start"));
    client.events.on("request:end", () => fired.push("end"));

    await client.getBenchmarks();

    expect(fired).toContain("start");
    expect(fired).toContain("end");
  });

  it("fires request:error on final failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(500, { error: "Crashed", detail: "" })),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const errorEvents: unknown[] = [];
    client.events.on("request:error", ({ payload }) => errorEvents.push(payload.error));

    await client.runEvaluation(SAMPLE_EVAL_CONFIG);

    expect(errorEvents).toHaveLength(1);
  });

  it("retries on 503 and succeeds on third attempt", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount += 1;
        if (callCount < 3) {
          return Promise.resolve(makeErrorResponse(503, { error: "Unavailable", detail: "" }));
        }
        return Promise.resolve(makeSuccessResponse([SAMPLE_BENCHMARK_DESCRIPTOR]));
      }),
    );

    const retried: number[] = [];
    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 3 });
    client.events.on("request:retry", ({ payload }) => retried.push(payload.attempt));

    const result = await client.getBenchmarks();

    expect(result.ok).toBe(true);
    expect(callCount).toBe(3);
    expect(retried).toHaveLength(2);
  });

  it("returns ok:false after all retries exhausted on persistent 503", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(503, { error: "Down", detail: "" })),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 2 });
    const result = await client.getResults({ agentId: "agent-001" });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(503);
    }
  });

  it("returns ok:false after all retries exhausted on persistent 502", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(502, { error: "Bad gateway", detail: "" })),
    );

    const client = createAgentEvalClient({ baseUrl: BASE_URL, maxRetries: 1 });
    const result = await client.compareRuns("run-001", "run-002");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(502);
    }
  });
});
