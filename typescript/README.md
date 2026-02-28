# @aumos/agent-eval

TypeScript client for the [AumOS agent-eval](https://github.com/invincible-jha/agent-eval)
evaluation framework. Run benchmarks, compute accuracy/safety/consistency/cost metrics,
and compare evaluation runs — all from TypeScript or JavaScript.

## Requirements

- Node.js 18+ (uses native Fetch API)
- TypeScript 5.3+ (strict mode)

## Installation

```bash
npm install @aumos/agent-eval
```

## Usage

### HTTP client

```ts
import { createAgentEvalClient } from "@aumos/agent-eval";

const client = createAgentEvalClient({
  baseUrl: "http://localhost:8090",
  timeoutMs: 30_000,
});

// Start an evaluation run
const runResult = await client.runEvaluation({
  eval_name: "safety-smoke-test",
  agent_id: "my-agent-v2",
  benchmark_id: "aumos-safety-v1",
  dimensions: ["accuracy", "safety"],
  parameters: { temperature: 0.0 },
});

if (runResult.ok) {
  const { run_id, status } = runResult.data;
  console.log(`Run ${run_id} queued with status: ${status}`);
}

// Poll for completion
const statusResult = await client.getRunStatus(runResult.ok ? runResult.data.run_id : "");
if (statusResult.ok && statusResult.data.status === "completed") {
  console.log("Composite score:", statusResult.data.result?.composite_score);
}

// List available benchmarks
const benchmarks = await client.getBenchmarks();
if (benchmarks.ok) {
  for (const b of benchmarks.data) {
    console.log(b.benchmark_id, "-", b.name);
  }
}

// Compare two runs
const comparison = await client.compareRuns("run-baseline-001", "run-candidate-002");
if (comparison.ok) {
  console.log("Composite delta:", comparison.data.composite_delta);
  console.log("Improved:", comparison.data.significant_improvement);
}

// Retrieve results for an agent
const results = await client.getResults({
  agentId: "my-agent-v2",
  benchmarkId: "aumos-safety-v1",
  limit: 10,
});
```

### Local metric calculator

```ts
import { createMetricCalculator } from "@aumos/agent-eval";

const calculator = createMetricCalculator();

// Accuracy
const accuracy = calculator.computeAccuracy([
  { prediction: "Paris", reference: "Paris" },
  { prediction: "London", reference: "Berlin" },
]);
console.log("Accuracy score:", accuracy.score);

// Safety
const safety = calculator.computeSafety([
  { text: "Hello", flagged: false },
  { text: "Bad content", flagged: true, classifierScore: 0.95 },
]);
console.log("Safety score:", safety.score);

// Consistency
const consistency = calculator.computeConsistency([
  {
    prompt: "What is the capital of France?",
    responses: ["Paris is the capital.", "The capital is Paris.", "Paris."],
  },
]);
console.log("Consistency score:", consistency.score);

// Cost efficiency
const cost = calculator.computeCost(
  [
    { cost_usd: 0.001, input_tokens: 500, output_tokens: 100 },
    { cost_usd: 0.002, input_tokens: 800, output_tokens: 200 },
  ],
  0.00001, // budget: $10/M output tokens
);
console.log("Cost efficiency score:", cost.score);
```

## API reference

### `createAgentEvalClient(config)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | `string` | required | agent-eval server URL |
| `timeoutMs` | `number` | `30000` | Request timeout (ms) |
| `headers` | `Record<string, string>` | `{}` | Extra HTTP headers |

#### Methods

| Method | Description |
|--------|-------------|
| `runEvaluation(config)` | Start a benchmark run |
| `getBenchmarks()` | List registered benchmarks |
| `getResults(options)` | Retrieve results for an agent |
| `compareRuns(baseline, candidate)` | Head-to-head run comparison |
| `getRunStatus(runId)` | Poll a run's current status |

### `createMetricCalculator()`

| Method | Description |
|--------|-------------|
| `computeAccuracy(pairs)` | Exact-match + token-F1 accuracy |
| `computeSafety(annotations)` | Flag-rate + classifier safety score |
| `computeConsistency(groups)` | Pairwise similarity across re-runs |
| `computeCost(records, budget?)` | Cost-efficiency relative to token budget |

## License

Apache-2.0. See [LICENSE](../../LICENSE) for details.
