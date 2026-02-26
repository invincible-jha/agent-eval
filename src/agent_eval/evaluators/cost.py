"""Basic cost evaluator for agent-eval.

Estimates token usage from agent output and checks it against a budget.

NOTE: This is a commodity cost estimator. It uses approximate token counting
(word count * 1.3) and a static pricing table. It is NOT an exact token
counter, does NOT integrate with provider billing APIs, and does NOT
perform cost optimization analysis. Advanced cost tracking with exact
token counts and real-time pricing is available via the plugin system.
"""
from __future__ import annotations

from agent_eval.core.evaluator import Dimension, DimensionScore, Evaluator
from agent_eval.core.exceptions import EvaluatorError

# Static pricing table: USD per 1000 tokens (approximate public rates)
# These are approximate values for planning purposes only.
# Do NOT use for billing or financial reporting.
_PRICING_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.010, "output": 0.030},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
}

_DEFAULT_MAX_TOKENS = 4_096
_WORDS_TO_TOKENS_RATIO = 1.3


def _estimate_tokens(text: str) -> int:
    """Approximate token count from text using a word-count heuristic.

    Uses the rule: tokens ≈ words * 1.3

    This is an approximation for planning and budget purposes only.
    It is NOT the actual token count from any specific tokenizer (BPE, etc.).

    Parameters
    ----------
    text:
        The text to estimate tokens for.

    Returns
    -------
    int
        Estimated token count. Minimum of 1 for non-empty text.
    """
    if not text.strip():
        return 0
    word_count = len(text.split())
    return max(1, int(word_count * _WORDS_TO_TOKENS_RATIO))


class BasicCostEvaluator(Evaluator):
    """Evaluates whether agent token usage stays within budget.

    Token counts are estimated from the input and output text using
    the approximation: tokens = words * 1.3.

    Optionally estimates USD cost using a static pricing table.
    The max_tokens budget is taken from (in priority order):
    1. ``max_cost_tokens`` key in metadata (from TestCase or suite default)
    2. The ``max_tokens`` parameter passed to this constructor

    NOTE: This is NOT an exact cost calculator. Token estimates are
    approximate and do NOT reflect actual billing. The pricing table
    is static and may be out of date. Advanced cost analysis with
    exact token counts and real-time API integration is available via
    the plugin system.

    Parameters
    ----------
    max_tokens:
        Default maximum acceptable token count (input + output).
        Default: 4096.
    model:
        Model name for USD cost estimation. Must be a key in the
        pricing table, or None to skip cost estimation.
        Default: None.
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        model: str | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise EvaluatorError(
                self.name,
                f"max_tokens must be positive, got {max_tokens}",
            )
        if model is not None and model not in _PRICING_TABLE:
            raise EvaluatorError(
                self.name,
                f"Unknown model {model!r}. Known models: {sorted(_PRICING_TABLE.keys())}",
            )
        self.max_tokens = max_tokens
        self.model = model

    @property
    def dimension(self) -> Dimension:
        return Dimension.COST

    @property
    def name(self) -> str:
        return "BasicCostEvaluator"

    def evaluate(
        self,
        case_id: str,
        agent_output: str,
        expected_output: str | None,
        metadata: dict[str, str | int | float | bool],
    ) -> DimensionScore:
        """Estimate token cost and check against budget.

        Parameters
        ----------
        case_id:
            Test case identifier.
        agent_output:
            The agent's output text (used for output token estimation).
        expected_output:
            Not used by this evaluator.
        metadata:
            May contain:
            - ``max_cost_tokens``: override budget
            - ``input_text``: original input for input token estimation
            - ``input_tokens``: exact input token count (overrides estimation)
            - ``output_tokens``: exact output token count (overrides estimation)

        Returns
        -------
        DimensionScore
            Score proportional to budget utilization. Passes at 1.0 if
            total tokens <= budget.
        """
        # Get effective budget
        budget_raw = metadata.get("max_cost_tokens", self.max_tokens)
        budget = int(budget_raw) if isinstance(budget_raw, (int, float)) else self.max_tokens

        # Estimate token counts
        input_tokens_raw = metadata.get("input_tokens")
        output_tokens_raw = metadata.get("output_tokens")

        if isinstance(input_tokens_raw, (int, float)):
            input_tokens = int(input_tokens_raw)
        else:
            input_text_raw = metadata.get("input_text", "")
            input_text = str(input_text_raw) if input_text_raw else ""
            input_tokens = _estimate_tokens(input_text)

        if isinstance(output_tokens_raw, (int, float)):
            output_tokens = int(output_tokens_raw)
        else:
            output_tokens = _estimate_tokens(agent_output)

        total_tokens = input_tokens + output_tokens

        # Score: 1.0 at 0 tokens, 0.5 at budget, 0.0 at 2x budget
        if total_tokens <= budget:
            score = 1.0 - (total_tokens / (budget * 2)) * 0.5
            score = round(max(0.5, min(1.0, score)), 4)
            passed = True
            reason = f"Token usage {total_tokens} within budget {budget}"
        else:
            overage = total_tokens / budget
            score = max(0.0, 1.0 - (overage - 1.0))
            score = round(score, 4)
            passed = False
            reason = (
                f"Token usage {total_tokens} exceeds budget {budget} "
                f"({total_tokens - budget} tokens over)"
            )

        # Append cost estimate if model is configured
        if self.model and self.model in _PRICING_TABLE:
            pricing = _PRICING_TABLE[self.model]
            input_cost = (input_tokens / 1000) * pricing["input"]
            output_cost = (output_tokens / 1000) * pricing["output"]
            total_cost = input_cost + output_cost
            reason += f" | Est. cost ${total_cost:.4f} ({self.model})"

        return DimensionScore(
            dimension=self.dimension,
            score=score,
            passed=passed,
            reason=reason,
            raw_value=float(total_tokens),
        )
