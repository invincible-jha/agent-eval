/**
 * MetricCalculator — pure computation utilities for agent evaluation metrics.
 *
 * These functions operate entirely on typed data structures and produce
 * MetricScore records. They do not make network calls and have no side effects.
 *
 * All score values are normalised to the range [0, 1] unless documented otherwise.
 */

import type { EvalDimension, MetricScore } from "./types.js";

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/** A single (prediction, reference) pair used in accuracy calculations. */
export interface PredictionPair {
  /** Model prediction or agent response. */
  readonly prediction: string;
  /** Ground-truth reference answer. */
  readonly reference: string;
}

/** Raw safety annotation for a single response. */
export interface SafetyAnnotation {
  /** The text that was evaluated. */
  readonly text: string;
  /**
   * Whether the text was flagged as unsafe by an oracle or human reviewer.
   * true = unsafe, false = safe.
   */
  readonly flagged: boolean;
  /**
   * Optional float in [0, 1] from an automated safety classifier.
   * Higher = more likely unsafe.
   */
  readonly classifierScore?: number;
}

/** A group of responses to the same prompt, used for consistency scoring. */
export interface ConsistencyGroup {
  /** The prompt that produced all responses. */
  readonly prompt: string;
  /** Multiple responses generated for the same prompt. */
  readonly responses: readonly string[];
}

/** Cost data for a single LLM call or agent turn. */
export interface CostRecord {
  /** Cost in USD for this call. */
  readonly cost_usd: number;
  /** Number of input tokens consumed. */
  readonly input_tokens: number;
  /** Number of output tokens produced. */
  readonly output_tokens: number;
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

function buildMetricScore(
  metricId: string,
  metricName: string,
  dimension: EvalDimension,
  score: number,
  metadata: Readonly<Record<string, unknown>>,
  ciLower?: number,
  ciUpper?: number,
): MetricScore {
  const base: MetricScore = {
    metric_id: metricId,
    metric_name: metricName,
    dimension,
    score,
    metadata,
  };
  if (ciLower !== undefined && ciUpper !== undefined) {
    return { ...base, ci_lower: ciLower, ci_upper: ciUpper };
  }
  return base;
}

// ---------------------------------------------------------------------------
// MetricCalculator
// ---------------------------------------------------------------------------

/** Collection of pure metric-computation functions. */
export interface MetricCalculator {
  /**
   * Compute exact-match and token-overlap (F1) accuracy scores.
   *
   * @param pairs - Array of (prediction, reference) pairs.
   * @returns MetricScore for the accuracy dimension.
   */
  computeAccuracy(pairs: readonly PredictionPair[]): MetricScore;

  /**
   * Compute a safety score based on flagging rate and optional classifier scores.
   *
   * The returned score is the fraction of responses that are SAFE (i.e. not
   * flagged). A score of 1.0 means all responses passed safety checks.
   *
   * @param annotations - Per-response safety annotations.
   * @returns MetricScore for the safety dimension.
   */
  computeSafety(annotations: readonly SafetyAnnotation[]): MetricScore;

  /**
   * Compute response consistency across multiple generations for the same prompt.
   *
   * Uses normalised edit-distance similarity averaged across all response pairs
   * within each group and then across groups.
   *
   * @param groups - Arrays of responses grouped by shared prompt.
   * @returns MetricScore for the consistency dimension.
   */
  computeConsistency(groups: readonly ConsistencyGroup[]): MetricScore;

  /**
   * Compute a cost-efficiency score from raw cost records.
   *
   * The score is the inverse of the normalised mean cost per output token
   * relative to a reference budget. A score of 1.0 means all calls were
   * within budget; scores below 1.0 indicate over-budget usage.
   *
   * @param records - Per-call cost records.
   * @param budgetUsdPerOutputToken - Reference cost per output token in USD
   *   (e.g. 0.000010 for $10/M tokens). Defaults to 0.000010.
   * @returns MetricScore for the cost dimension.
   */
  computeCost(
    records: readonly CostRecord[],
    budgetUsdPerOutputToken?: number,
  ): MetricScore;
}

// ---------------------------------------------------------------------------
// Normalised token-level F1 (used inside computeAccuracy)
// ---------------------------------------------------------------------------

function tokenF1(prediction: string, reference: string): number {
  const predTokens = prediction.toLowerCase().split(/\s+/).filter(Boolean);
  const refTokens = reference.toLowerCase().split(/\s+/).filter(Boolean);

  if (predTokens.length === 0 && refTokens.length === 0) return 1.0;
  if (predTokens.length === 0 || refTokens.length === 0) return 0.0;

  const predSet = new Set(predTokens);
  const refSet = new Set(refTokens);
  let commonCount = 0;
  for (const token of predSet) {
    if (refSet.has(token)) commonCount += 1;
  }

  if (commonCount === 0) return 0.0;
  const precision = commonCount / predTokens.length;
  const recall = commonCount / refTokens.length;
  return (2 * precision * recall) / (precision + recall);
}

// ---------------------------------------------------------------------------
// Levenshtein similarity (used inside computeConsistency)
// ---------------------------------------------------------------------------

function normalisedSimilarity(a: string, b: string): number {
  if (a === b) return 1.0;
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1.0;

  // Simple character-level Levenshtein bounded to avoid O(n^2) on long strings.
  const truncA = a.slice(0, 512);
  const truncB = b.slice(0, 512);
  const lenA = truncA.length;
  const lenB = truncB.length;

  const previousRow = Array.from({ length: lenB + 1 }, (_, i) => i);

  for (let i = 1; i <= lenA; i++) {
    let previousDiagonal = previousRow[0];
    previousRow[0] = i;
    for (let j = 1; j <= lenB; j++) {
      const temp = previousRow[j];
      if (truncA[i - 1] === truncB[j - 1]) {
        previousRow[j] = previousDiagonal;
      } else {
        previousRow[j] =
          1 + Math.min(previousDiagonal, previousRow[j], previousRow[j - 1]);
      }
      previousDiagonal = temp;
    }
  }

  const editDistance = previousRow[lenB];
  return 1 - editDistance / Math.max(lenA, lenB);
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create a MetricCalculator instance.
 *
 * @returns A MetricCalculator with all computation methods.
 */
export function createMetricCalculator(): MetricCalculator {
  return {
    computeAccuracy(pairs: readonly PredictionPair[]): MetricScore {
      if (pairs.length === 0) {
        return buildMetricScore(
          "accuracy_f1",
          "Token-Level F1 Accuracy",
          "accuracy",
          0,
          { sample_count: 0, exact_match_count: 0 },
        );
      }

      let exactMatchCount = 0;
      let totalF1 = 0;

      for (const { prediction, reference } of pairs) {
        const normalised_pred = prediction.trim().toLowerCase();
        const normalised_ref = reference.trim().toLowerCase();
        if (normalised_pred === normalised_ref) exactMatchCount += 1;
        totalF1 += tokenF1(prediction, reference);
      }

      const exactMatchRate = exactMatchCount / pairs.length;
      const meanF1 = totalF1 / pairs.length;
      // Composite: equal weight of exact-match rate and token F1.
      const compositeScore = (exactMatchRate + meanF1) / 2;

      return buildMetricScore(
        "accuracy_f1",
        "Token-Level F1 Accuracy",
        "accuracy",
        compositeScore,
        {
          sample_count: pairs.length,
          exact_match_count: exactMatchCount,
          exact_match_rate: exactMatchRate,
          mean_token_f1: meanF1,
        },
      );
    },

    computeSafety(annotations: readonly SafetyAnnotation[]): MetricScore {
      if (annotations.length === 0) {
        return buildMetricScore(
          "safety_pass_rate",
          "Safety Pass Rate",
          "safety",
          1.0,
          { sample_count: 0, flagged_count: 0 },
        );
      }

      const flaggedCount = annotations.filter((a) => a.flagged).length;
      const safeRate = 1 - flaggedCount / annotations.length;

      // If classifier scores are available, incorporate the mean.
      const withClassifier = annotations.filter(
        (a) => a.classifierScore !== undefined,
      );
      let meanClassifierSafeScore = 1.0;
      if (withClassifier.length > 0) {
        const sumUnsafe = withClassifier.reduce(
          (sum, a) => sum + (a.classifierScore ?? 0),
          0,
        );
        meanClassifierSafeScore = 1 - sumUnsafe / withClassifier.length;
      }

      // Composite: flag-based rate weighted 70%, classifier 30%.
      const hasClassifier = withClassifier.length > 0;
      const compositeScore = hasClassifier
        ? 0.7 * safeRate + 0.3 * meanClassifierSafeScore
        : safeRate;

      return buildMetricScore(
        "safety_pass_rate",
        "Safety Pass Rate",
        "safety",
        compositeScore,
        {
          sample_count: annotations.length,
          flagged_count: flaggedCount,
          flag_based_safe_rate: safeRate,
          mean_classifier_safe_score: hasClassifier
            ? meanClassifierSafeScore
            : null,
          classifier_coverage: withClassifier.length,
        },
      );
    },

    computeConsistency(groups: readonly ConsistencyGroup[]): MetricScore {
      if (groups.length === 0) {
        return buildMetricScore(
          "consistency_similarity",
          "Response Consistency",
          "consistency",
          1.0,
          { group_count: 0 },
        );
      }

      let totalGroupScore = 0;

      for (const group of groups) {
        const responses = group.responses;
        if (responses.length < 2) {
          totalGroupScore += 1.0;
          continue;
        }

        let pairSum = 0;
        let pairCount = 0;
        for (let i = 0; i < responses.length; i++) {
          for (let j = i + 1; j < responses.length; j++) {
            pairSum += normalisedSimilarity(
              responses[i] ?? "",
              responses[j] ?? "",
            );
            pairCount += 1;
          }
        }

        totalGroupScore += pairCount > 0 ? pairSum / pairCount : 1.0;
      }

      const meanGroupScore = totalGroupScore / groups.length;

      return buildMetricScore(
        "consistency_similarity",
        "Response Consistency",
        "consistency",
        meanGroupScore,
        {
          group_count: groups.length,
          mean_pairwise_similarity: meanGroupScore,
        },
      );
    },

    computeCost(
      records: readonly CostRecord[],
      budgetUsdPerOutputToken: number = 0.00001,
    ): MetricScore {
      if (records.length === 0) {
        return buildMetricScore(
          "cost_efficiency",
          "Cost Efficiency",
          "cost",
          1.0,
          { record_count: 0, total_cost_usd: 0 },
        );
      }

      const totalCostUsd = records.reduce((sum, r) => sum + r.cost_usd, 0);
      const totalOutputTokens = records.reduce(
        (sum, r) => sum + r.output_tokens,
        0,
      );
      const totalInputTokens = records.reduce(
        (sum, r) => sum + r.input_tokens,
        0,
      );

      const actualCostPerOutputToken =
        totalOutputTokens > 0 ? totalCostUsd / totalOutputTokens : 0;

      // Score of 1.0 when at or under budget; degrades linearly above budget.
      const rawEfficiency =
        actualCostPerOutputToken === 0
          ? 1.0
          : Math.min(1.0, budgetUsdPerOutputToken / actualCostPerOutputToken);

      return buildMetricScore(
        "cost_efficiency",
        "Cost Efficiency",
        "cost",
        rawEfficiency,
        {
          record_count: records.length,
          total_cost_usd: totalCostUsd,
          total_input_tokens: totalInputTokens,
          total_output_tokens: totalOutputTokens,
          actual_cost_per_output_token: actualCostPerOutputToken,
          budget_usd_per_output_token: budgetUsdPerOutputToken,
        },
      );
    },
  };
}
