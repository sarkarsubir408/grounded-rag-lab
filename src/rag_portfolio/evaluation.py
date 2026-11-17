from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .abstention import abstention_signal, calibrate_threshold
from .chunking import create_chunk_records
from .embeddings import TextEncoder
from .ingestion import documents_fingerprint
from .metrics import DEFAULT_K_VALUES, macro_average, retrieval_metrics
from .models import ChunkRecord, SourceDocument
from .retrieval import RETRIEVAL_METHODS, build_index, search

CHUNK_CONFIGURATIONS = ((100, 20), (150, 20), (100, 0), (100, 40))
CALIBRATION_CONFIGURATION = (100, 20)
CALIBRATION_TOP_K = 5


def load_questions(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = payload["questions"]
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation question IDs must be unique.")
    splits = {item["split"] for item in questions}
    if splits != {"validation", "test"}:
        raise ValueError("Questions must include separate validation and test splits.")
    validation_ids = {item["id"] for item in questions if item["split"] == "validation"}
    test_ids = {item["id"] for item in questions if item["split"] == "test"}
    if validation_ids & test_ids:
        raise ValueError("Validation and test questions must be disjoint.")
    return payload


def relevant_chunks_for_question(
    question: dict[str, Any], records: list[ChunkRecord]
) -> set[str]:
    evidence = {
        (reference["doc_id"], reference["passage_id"])
        for reference in question.get("evidence", [])
    }
    record_evidence = {
        (record.document_id, passage_id)
        for record in records
        for passage_id in record.passage_ids
    }
    missing = evidence - record_evidence
    if missing:
        raise ValueError(f"Gold evidence not found for {question['id']}: {sorted(missing)}")
    return {
        record.chunk_id
        for record in records
        if any((record.document_id, passage_id) in evidence for passage_id in record.passage_ids)
    }


def calibrate_abstention_thresholds(
    documents: list[SourceDocument],
    questions: list[dict[str, Any]],
    model: TextEncoder,
    rrf_constant: int,
) -> dict[str, Any]:
    """Calibrate all methods from labeled validation questions for this corpus."""
    if len(questions) < 2 or any(question.get("split") != "validation" for question in questions):
        raise ValueError("Calibration requires at least two questions, all marked validation.")
    ids = [question.get("id") for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Validation question IDs must be unique.")
    if any(not isinstance(question.get("answerable"), bool) for question in questions):
        raise ValueError("Each validation question must have a boolean answerable label.")
    if len({question["answerable"] for question in questions}) < 2:
        raise ValueError("Validation labels must include answerable and unanswerable questions.")
    if any(not str(question.get("review_status", "")).startswith("human-reviewed") for question in questions):
        raise ValueError("Every calibration question must be human-reviewed before threshold calibration.")

    records = create_chunk_records(documents, *CALIBRATION_CONFIGURATION)
    if not records:
        raise ValueError("The calibration corpus produced no chunks.")
    index = build_index(records, model)
    thresholds: dict[str, dict[str, Any]] = {}
    for method in RETRIEVAL_METHODS:
        scores = []
        labels = []
        for question in questions:
            hits = search(
                question["question"], index, model, method,
                top_k=CALIBRATION_TOP_K, rrf_constant=rrf_constant,
            )
            scores.append(abstention_signal(method, hits))
            labels.append(bool(question["answerable"]))
        thresholds[method] = calibrate_threshold(scores, labels)
    for method, result in thresholds.items():
        result["signal_definition"] = {
            "dense": f"maximum cosine similarity among the top-{CALIBRATION_TOP_K} dense results",
            "bm25": f"maximum BM25 score among the top-{CALIBRATION_TOP_K} BM25 results",
            "hybrid": f"maximum dense cosine similarity among the top-{CALIBRATION_TOP_K} RRF results; RRF score itself is not treated as confidence",
        }[method]
    return {
        "dataset_status": (
            "human-reviewed validation annotations; threshold calibration remains provisional "
            f"because it uses only {len(questions)} validation questions"
        ),
        "split_used": "validation only",
        "test_split_used_for_calibration": False,
        "calibration_configuration": {"chunk_size": 100, "overlap": 20},
        "calibration_top_k": CALIBRATION_TOP_K,
        "rrf_constant": rrf_constant,
        "corpus_fingerprint": documents_fingerprint(documents),
        "retrievers": thresholds,
    }


def run_benchmark(
    documents: list[SourceDocument],
    questions_payload: dict[str, Any],
    model: TextEncoder,
    rrf_constant: int = 60,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    questions = questions_payload["questions"]
    validation = [question for question in questions if question["split"] == "validation"]
    test = [question for question in questions if question["split"] == "test"]
    if {question["id"] for question in validation} & {question["id"] for question in test}:
        raise ValueError("Calibration and test data overlap.")

    per_query_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    calibration = calibrate_abstention_thresholds(documents, validation, model, rrf_constant)

    for chunk_size, overlap in CHUNK_CONFIGURATIONS:
        records = create_chunk_records(documents, chunk_size, overlap)
        if not records:
            raise ValueError(f"{chunk_size}/{overlap} produced no chunks.")
        index = build_index(records, model)
        metric_rows: dict[tuple[str, int], list[dict[str, float | None]]] = defaultdict(list)
        for question in test:
            relevant_ids = relevant_chunks_for_question(question, records)
            if question["answerable"] and not relevant_ids:
                raise ValueError(f"Answerable test question {question['id']} maps to no relevant chunks.")
            if not question["answerable"] and relevant_ids:
                raise ValueError(f"Unanswerable test question {question['id']} has gold relevant chunks.")
            for method in RETRIEVAL_METHODS:
                hits = search(question["question"], index, model, method, top_k=max(k_values), rrf_constant=rrf_constant)
                metrics = retrieval_metrics(hits, relevant_ids, k_values)
                for k in k_values:
                    row_metrics = {
                        name: value
                        for name, value in metrics.items()
                        if name.endswith(f"@{k}")
                    }
                    if question["answerable"]:
                        metric_rows[(method, k)].append(row_metrics)
                    per_query_rows.append(
                        {
                            "configuration": f"{chunk_size}/{overlap}",
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "method": method,
                            "question_id": question["id"],
                            "split": "test",
                            "answerable": question["answerable"],
                            "top_k": k,
                            "retrieved_chunk_ids": [hit.chunk.chunk_id for hit in hits[:k]],
                            "relevant_chunk_ids": sorted(relevant_ids),
                            **row_metrics,
                        }
                    )

        for method in RETRIEVAL_METHODS:
            for k in k_values:
                averages = macro_average(metric_rows[(method, k)])
                summary_rows.append(
                    {
                        "configuration": f"{chunk_size}/{overlap}",
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "method": method,
                        "top_k": k,
                        "answerable_test_queries": len(metric_rows[(method, k)]),
                        "unanswerable_test_queries": sum(not item["answerable"] for item in test),
                        **averages,
                    }
                )

    return per_query_rows, summary_rows, calibration
