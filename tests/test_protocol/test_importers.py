"""Tests for agent_eval.protocol.importers."""
from __future__ import annotations

import pytest

from agent_eval.protocol.eval_protocol import (
    EvalRequest,
    EvalResponse,
    MetricStatus,
)
from agent_eval.protocol.importers import (
    DeepEvalImporter,
    RagasImporter,
)


_DEEPEVAL_RECORD = {
    "test_case_id": "tc_001",
    "input": "What is the capital of France?",
    "actual_output": "The capital of France is Paris.",
    "expected_output": "Paris",
    "context": ["France is a country in Europe.", "Paris is known as the City of Light."],
    "model": "gpt-4o",
    "metric_scores": {
        "answer_relevancy": 0.92,
        "faithfulness": 0.88,
    },
    "metric_thresholds": {
        "answer_relevancy": 0.7,
        "faithfulness": 0.7,
    },
    "metric_reasons": {
        "faithfulness": "Answer is grounded in context.",
    },
}

_RAGAS_RECORD = {
    "row_id": "r001",
    "question": "What is RAG?",
    "answer": "RAG stands for Retrieval Augmented Generation.",
    "contexts": ["RAG is a technique for grounding LLMs.", "It combines retrieval with generation."],
    "ground_truth": "Retrieval Augmented Generation",
    "faithfulness": 0.9,
    "answer_relevancy": 0.85,
    "context_precision": 0.78,
    "context_recall": 0.82,
}


class TestDeepEvalImporter:
    def test_import_request_basic(self) -> None:
        importer = DeepEvalImporter()
        req = importer.import_request(_DEEPEVAL_RECORD)
        assert isinstance(req, EvalRequest)
        assert req.input == "What is the capital of France?"
        assert req.agent_id == "gpt-4o"

    def test_import_request_sets_request_id(self) -> None:
        importer = DeepEvalImporter()
        req = importer.import_request(_DEEPEVAL_RECORD)
        assert req.request_id == "tc_001"

    def test_import_request_sets_expected_output(self) -> None:
        importer = DeepEvalImporter()
        req = importer.import_request(_DEEPEVAL_RECORD)
        assert req.expected_output == "Paris"

    def test_import_request_context_extracted(self) -> None:
        importer = DeepEvalImporter()
        req = importer.import_request(_DEEPEVAL_RECORD)
        assert "retrieved_contexts" in req.context
        assert len(req.context["retrieved_contexts"]) == 2

    def test_import_request_metric_names_extracted(self) -> None:
        importer = DeepEvalImporter()
        req = importer.import_request(_DEEPEVAL_RECORD)
        assert "answer_relevancy" in req.metrics
        assert "faithfulness" in req.metrics

    def test_import_request_missing_input_raises(self) -> None:
        importer = DeepEvalImporter()
        with pytest.raises(ValueError, match="input"):
            importer.import_request({"actual_output": "answer"})

    def test_import_response_basic(self) -> None:
        importer = DeepEvalImporter()
        resp = importer.import_response(_DEEPEVAL_RECORD)
        assert isinstance(resp, EvalResponse)
        assert resp.agent_output == "The capital of France is Paris."

    def test_import_response_metrics_parsed(self) -> None:
        importer = DeepEvalImporter()
        resp = importer.import_response(_DEEPEVAL_RECORD)
        metric_names = {m.name for m in resp.metrics}
        assert "answer_relevancy" in metric_names
        assert "faithfulness" in metric_names

    def test_import_response_pass_status_above_threshold(self) -> None:
        importer = DeepEvalImporter()
        resp = importer.import_response(_DEEPEVAL_RECORD)
        rel_metric = next(m for m in resp.metrics if m.name == "answer_relevancy")
        assert rel_metric.status == MetricStatus.PASS

    def test_import_response_reason_preserved(self) -> None:
        importer = DeepEvalImporter()
        resp = importer.import_response(_DEEPEVAL_RECORD)
        faith_metric = next(m for m in resp.metrics if m.name == "faithfulness")
        assert "grounded" in faith_metric.reason

    def test_import_response_metadata_source_format(self) -> None:
        importer = DeepEvalImporter()
        resp = importer.import_response(_DEEPEVAL_RECORD)
        assert resp.metadata.get("source_format") == "deepeval"

    def test_import_batch_responses(self) -> None:
        importer = DeepEvalImporter()
        responses = importer.import_batch([_DEEPEVAL_RECORD, _DEEPEVAL_RECORD])
        assert len(responses) == 2


class TestRagasImporter:
    def test_import_request_basic(self) -> None:
        importer = RagasImporter()
        req = importer.import_request(_RAGAS_RECORD)
        assert isinstance(req, EvalRequest)
        assert req.input == "What is RAG?"

    def test_import_request_sets_request_id(self) -> None:
        importer = RagasImporter()
        req = importer.import_request(_RAGAS_RECORD)
        assert req.request_id == "r001"

    def test_import_request_ground_truth_as_expected_output(self) -> None:
        importer = RagasImporter()
        req = importer.import_request(_RAGAS_RECORD)
        assert req.expected_output == "Retrieval Augmented Generation"

    def test_import_request_contexts_in_context(self) -> None:
        importer = RagasImporter()
        req = importer.import_request(_RAGAS_RECORD)
        assert "contexts" in req.context
        assert len(req.context["contexts"]) == 2

    def test_import_request_metric_names_detected(self) -> None:
        importer = RagasImporter()
        req = importer.import_request(_RAGAS_RECORD)
        assert "faithfulness" in req.metrics
        assert "answer_relevancy" in req.metrics

    def test_import_request_missing_question_raises(self) -> None:
        importer = RagasImporter()
        with pytest.raises(ValueError, match="question"):
            importer.import_request({"answer": "something"})

    def test_import_response_basic(self) -> None:
        importer = RagasImporter()
        resp = importer.import_response(_RAGAS_RECORD)
        assert isinstance(resp, EvalResponse)
        assert "RAG" in resp.agent_output

    def test_import_response_metrics_parsed(self) -> None:
        importer = RagasImporter()
        resp = importer.import_response(_RAGAS_RECORD)
        metric_names = {m.name for m in resp.metrics}
        assert "faithfulness" in metric_names
        assert "context_precision" in metric_names

    def test_import_response_pass_above_threshold(self) -> None:
        importer = RagasImporter(passing_threshold=0.5)
        resp = importer.import_response(_RAGAS_RECORD)
        faith_metric = next(m for m in resp.metrics if m.name == "faithfulness")
        assert faith_metric.status == MetricStatus.PASS

    def test_import_response_fail_below_threshold(self) -> None:
        record = {**_RAGAS_RECORD, "faithfulness": 0.2}
        importer = RagasImporter(passing_threshold=0.5)
        resp = importer.import_response(record)
        faith_metric = next(m for m in resp.metrics if m.name == "faithfulness")
        assert faith_metric.status == MetricStatus.FAIL

    def test_import_response_metadata_source_format(self) -> None:
        importer = RagasImporter()
        resp = importer.import_response(_RAGAS_RECORD)
        assert resp.metadata.get("source_format") == "ragas"

    def test_import_batch_requests(self) -> None:
        importer = RagasImporter()
        requests = importer.import_batch(
            [_RAGAS_RECORD, _RAGAS_RECORD], import_as="request"
        )
        assert len(requests) == 2

    def test_import_batch_skips_invalid(self) -> None:
        importer = RagasImporter()
        records = [
            _RAGAS_RECORD,
            {"answer": "no question field here"},
            _RAGAS_RECORD,
        ]
        requests = importer.import_batch(records, import_as="request")
        assert len(requests) == 2
