"""Importers for external eval framework formats into the eval protocol.

Supported formats:
- DeepEval: test case format with metric scores
- Ragas: retrieval-augmented generation evaluation format

Neither DeepEval nor Ragas is required as a dependency; the importers
work from raw dict/JSON data matching those frameworks' output schemas.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from agent_eval.protocol.eval_protocol import (
    EvalMetric,
    EvalRequest,
    EvalResponse,
    MetricStatus,
)

logger = logging.getLogger(__name__)


class ProtocolImporter(ABC):
    """Abstract base class for eval format importers.

    Subclasses must implement:
    - ``import_request(data)`` — parse framework data into EvalRequest
    - ``import_response(data)`` — parse framework data into EvalResponse
    """

    @abstractmethod
    def import_request(self, data: dict[str, object]) -> EvalRequest:
        """Import a framework-specific eval record as an EvalRequest.

        Parameters
        ----------
        data:
            Framework-specific dict representing an evaluation input.

        Returns
        -------
        EvalRequest
        """
        ...

    @abstractmethod
    def import_response(self, data: dict[str, object]) -> EvalResponse:
        """Import a framework-specific eval result as an EvalResponse.

        Parameters
        ----------
        data:
            Framework-specific dict representing an evaluation result.

        Returns
        -------
        EvalResponse
        """
        ...

    def import_batch(
        self,
        records: list[dict[str, object]],
        *,
        import_as: str = "response",
    ) -> list[EvalRequest] | list[EvalResponse]:
        """Import a batch of records.

        Parameters
        ----------
        records:
            List of framework-specific dicts.
        import_as:
            Whether to import as "request" or "response".

        Returns
        -------
        list[EvalRequest] | list[EvalResponse]
        """
        results: list[EvalRequest | EvalResponse] = []
        for index, record in enumerate(records):
            try:
                if import_as == "request":
                    results.append(self.import_request(record))
                else:
                    results.append(self.import_response(record))
            except Exception as exc:
                logger.warning(
                    "Failed to import record at index %d: %s", index, exc
                )
        return results  # type: ignore[return-value]


class DeepEvalImporter(ProtocolImporter):
    """Importer for DeepEval test case format.

    DeepEval test case schema (simplified):
    ::

        {
          "input": "...",
          "actual_output": "...",
          "expected_output": "...",  # optional
          "context": ["doc1", "doc2"],  # optional list of context strings
          "metric_scores": {
            "answer_relevancy": 0.9,
            "faithfulness": 0.85,
          },
          "model": "gpt-4",
          "test_case_id": "tc_001"
        }
    """

    def __init__(self, default_agent_id: str = "deepeval_agent") -> None:
        """Initialise the DeepEval importer.

        Parameters
        ----------
        default_agent_id:
            Default agent ID used when not present in the data.
        """
        self.default_agent_id = default_agent_id

    def import_request(self, data: dict[str, object]) -> EvalRequest:
        """Import a DeepEval test case as an EvalRequest.

        Parameters
        ----------
        data:
            DeepEval test case dict.

        Returns
        -------
        EvalRequest
        """
        input_text = str(data.get("input", ""))
        if not input_text:
            raise ValueError("DeepEval test case missing 'input' field.")

        context_raw = data.get("context", [])
        context: dict[str, object] = {}
        if isinstance(context_raw, list):
            context["retrieved_contexts"] = context_raw
        elif isinstance(context_raw, dict):
            context = context_raw

        agent_id = str(
            data.get("model") or data.get("agent_id") or self.default_agent_id
        )
        request_id = str(data.get("test_case_id") or data.get("id") or "")

        metric_names = list(data.get("metric_scores", {}).keys())

        build_kwargs: dict[str, object] = {
            "agent_id": agent_id,
            "input": input_text,
            "context": context,
            "metrics": metric_names,
        }
        if data.get("expected_output"):
            build_kwargs["expected_output"] = str(data["expected_output"])
        if request_id:
            build_kwargs["request_id"] = request_id

        return EvalRequest(**build_kwargs)

    def import_response(self, data: dict[str, object]) -> EvalResponse:
        """Import a DeepEval test case result as an EvalResponse.

        Parameters
        ----------
        data:
            DeepEval test case dict (with metric_scores populated).

        Returns
        -------
        EvalResponse
        """
        actual_output = str(data.get("actual_output", ""))
        request_id = str(data.get("test_case_id") or data.get("id") or "unknown")

        metrics: list[EvalMetric] = []
        metric_scores = data.get("metric_scores", {})
        metric_thresholds = data.get("metric_thresholds", {})
        metric_reasons = data.get("metric_reasons", {})

        if isinstance(metric_scores, dict):
            for metric_name, raw_score in metric_scores.items():
                try:
                    score = float(raw_score)
                except (ValueError, TypeError):
                    score = None

                threshold = None
                if metric_name in metric_thresholds:
                    try:
                        threshold = float(metric_thresholds[metric_name])
                    except (ValueError, TypeError):
                        pass

                if score is not None and threshold is not None:
                    status = MetricStatus.PASS if score >= threshold else MetricStatus.FAIL
                elif score is not None:
                    status = MetricStatus.PASS if score >= 0.5 else MetricStatus.FAIL
                else:
                    status = MetricStatus.ERROR

                reason = str(metric_reasons.get(metric_name, ""))

                metrics.append(
                    EvalMetric(
                        name=metric_name,
                        score=max(0.0, min(1.0, score)) if score is not None else None,
                        status=status,
                        threshold=threshold,
                        reason=reason,
                    )
                )

        return EvalResponse(
            request_id=request_id,
            agent_output=actual_output,
            metrics=metrics,
            metadata={"source_format": "deepeval"},
        )


class RagasImporter(ProtocolImporter):
    """Importer for Ragas evaluation format.

    Ragas evaluation result schema (simplified):
    ::

        {
          "question": "...",
          "answer": "...",
          "contexts": ["doc1", "doc2"],
          "ground_truth": "...",   # optional
          "faithfulness": 0.92,
          "answer_relevancy": 0.88,
          "context_precision": 0.76,
          "context_recall": 0.81,
          "row_id": "r001"
        }
    """

    _RAGAS_METRIC_NAMES: frozenset[str] = frozenset({
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevancy",
        "answer_correctness",
        "answer_similarity",
        "harmfulness",
        "maliciousness",
        "coherence",
        "conciseness",
    })

    def __init__(
        self,
        default_agent_id: str = "ragas_agent",
        passing_threshold: float = 0.5,
    ) -> None:
        """Initialise the Ragas importer.

        Parameters
        ----------
        default_agent_id:
            Default agent ID used when not present in the data.
        passing_threshold:
            Score threshold for pass/fail classification.
        """
        self.default_agent_id = default_agent_id
        self.passing_threshold = passing_threshold

    def import_request(self, data: dict[str, object]) -> EvalRequest:
        """Import a Ragas evaluation record as an EvalRequest.

        Parameters
        ----------
        data:
            Ragas evaluation dict.

        Returns
        -------
        EvalRequest
        """
        question = str(data.get("question", ""))
        if not question:
            raise ValueError("Ragas record missing 'question' field.")

        contexts = data.get("contexts", [])
        context: dict[str, object] = {}
        if isinstance(contexts, list):
            context["contexts"] = contexts

        ground_truth = data.get("ground_truth")

        metric_names = [
            key for key in data
            if key in self._RAGAS_METRIC_NAMES
        ]

        request_id = str(data.get("row_id") or data.get("id") or "")

        build_kwargs: dict[str, object] = {
            "agent_id": self.default_agent_id,
            "input": question,
            "context": context,
            "metrics": metric_names,
        }
        if ground_truth:
            build_kwargs["expected_output"] = str(ground_truth)
        if request_id:
            build_kwargs["request_id"] = request_id

        return EvalRequest(**build_kwargs)

    def import_response(self, data: dict[str, object]) -> EvalResponse:
        """Import a Ragas evaluation result as an EvalResponse.

        Parameters
        ----------
        data:
            Ragas evaluation result dict with metric scores.

        Returns
        -------
        EvalResponse
        """
        answer = str(data.get("answer", ""))
        request_id = str(data.get("row_id") or data.get("id") or "unknown")

        metrics: list[EvalMetric] = []
        for metric_name in self._RAGAS_METRIC_NAMES:
            if metric_name not in data:
                continue

            raw_score = data[metric_name]
            try:
                score = float(raw_score)
                score = max(0.0, min(1.0, score))
                status = (
                    MetricStatus.PASS
                    if score >= self.passing_threshold
                    else MetricStatus.FAIL
                )
            except (ValueError, TypeError):
                score = None
                status = MetricStatus.ERROR

            metrics.append(
                EvalMetric(
                    name=metric_name,
                    score=score,
                    status=status,
                    threshold=self.passing_threshold,
                )
            )

        return EvalResponse(
            request_id=request_id,
            agent_output=answer,
            metrics=metrics,
            metadata={
                "source_format": "ragas",
                "contexts_count": len(data.get("contexts", [])),
            },
        )
