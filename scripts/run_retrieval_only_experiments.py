from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_portfolio.chunking import create_chunk_records
from rag_portfolio.config import Settings
from rag_portfolio.embeddings import load_embedding_model
from rag_portfolio.evaluation import CHUNK_CONFIGURATIONS, load_questions, relevant_chunks_for_question
from rag_portfolio.ingestion import documents_fingerprint, load_documents_from_directory
from rag_portfolio.metrics import DEFAULT_K_VALUES, macro_average, retrieval_metrics
from rag_portfolio.models import SourceDocument
from rag_portfolio.retrieval import RETRIEVAL_METHODS, build_index, search


def run_retrieval_only_experiments(
    documents: list[SourceDocument],
    questions_payload: dict[str, Any],
    model: Any,
    rrf_constant: int = 60,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate only the held-out test split; this function never calibrates thresholds."""
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers.")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive.")

    questions = questions_payload["questions"]
    test_questions = [question for question in questions if question.get("split") == "test"]
    if not test_questions:
        raise ValueError("The evaluation dataset contains no held-out test questions.")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Evaluation question IDs must be unique.")

    for question in test_questions:
        if not isinstance(question.get("answerable"), bool):
            raise ValueError(f"Question {question['id']} must have a boolean answerability label.")
        evidence = question.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError(f"Question {question['id']} must have an evidence list.")
        if question["answerable"] and not evidence:
            raise ValueError(f"Answerable test question {question['id']} has no gold evidence.")
        if not question["answerable"] and evidence:
            raise ValueError(f"Unanswerable test question {question['id']} must have empty evidence.")

    per_query_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    unanswerable_count = sum(not question["answerable"] for question in test_questions)

    for chunk_size, overlap in CHUNK_CONFIGURATIONS:
        records = create_chunk_records(documents, chunk_size, overlap)
        if not records:
            raise ValueError(f"{chunk_size}/{overlap} produced no chunks.")
        index = build_index(records, model)
        metric_rows: dict[tuple[str, int], list[dict[str, float | None]]] = defaultdict(list)

        for question in test_questions:
            relevant_ids = relevant_chunks_for_question(question, records)
            if question["answerable"] and not relevant_ids:
                raise ValueError(
                    f"Answerable test question {question['id']} maps to no relevant chunks "
                    f"for {chunk_size}/{overlap}."
                )
            if not question["answerable"] and relevant_ids:
                raise ValueError(f"Unanswerable test question {question['id']} has gold relevant chunks.")

            for method in RETRIEVAL_METHODS:
                hits = search(
                    question["question"],
                    index,
                    model,
                    method,
                    top_k=max(k_values),
                    rrf_constant=rrf_constant,
                )
                metrics = retrieval_metrics(hits, relevant_ids, k_values)
                for k in k_values:
                    row_metrics = {
                        name: value for name, value in metrics.items() if name.endswith(f"@{k}")
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
                        "unanswerable_test_queries": unanswerable_count,
                        **averages,
                    }
                )

    return per_query_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def build_metadata(
    documents: list[SourceDocument],
    questions_payload: dict[str, Any],
    model: Any,
    rrf_constant: int,
    per_query_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    embedding_model_name: str,
    embedding_revision: str,
) -> dict[str, Any]:
    questions = questions_payload["questions"]
    validation = [question for question in questions if question["split"] == "validation"]
    test = [question for question in questions if question["split"] == "test"]
    return {
        "experiment_type": "retrieval_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint": documents_fingerprint(documents),
        "document_count": len(documents),
        "dataset_version": questions_payload.get("dataset_version", "unspecified"),
        "question_count": len(questions),
        "validation_question_count": len(validation),
        "validation_question_ids": [question["id"] for question in validation],
        "test_question_count": len(test),
        "test_question_ids": [question["id"] for question in test],
        "answerable_test_questions": sum(question["answerable"] for question in test),
        "unanswerable_test_questions": sum(not question["answerable"] for question in test),
        "retrieval_methods": list(RETRIEVAL_METHODS),
        "chunk_configurations": [
            {"chunk_size": size, "overlap": overlap} for size, overlap in CHUNK_CONFIGURATIONS
        ],
        "completed_combinations": len(RETRIEVAL_METHODS) * len(CHUNK_CONFIGURATIONS),
        "k_values": list(DEFAULT_K_VALUES),
        "metrics": {
            "precision": "Relevant retrieved chunks divided by the number returned (up to K).",
            "recall": "Unique relevant chunk IDs retrieved divided by all relevant chunk IDs for that configuration.",
            "hit_rate": "Whether at least one relevant chunk appears in the first K.",
            "mrr": "Reciprocal rank of the first relevant chunk, or zero.",
            "ndcg": "Binary-gain discounted ranking score normalized against the ideal ordering.",
        },
        "unanswerable_handling": (
            "Unanswerable test questions retain empty gold evidence. Existing retrieval_metrics returns null "
            "for ranking metrics on empty qrels; CSV cells are blank, these rows remain in per-query output, "
            "and they are excluded from answerable-query macro averages but counted in summaries."
        ),
        "validation_questions_used_for_test_metrics": False,
        "embedding_model": embedding_model_name,
        "embedding_revision": embedding_revision,
        "embedding_dimensions": model.get_embedding_dimension(),
        "rrf_constant": rrf_constant,
        "per_query_row_count": len(per_query_rows),
        "macro_summary_row_count": len(summary_rows),
        "failed_question_ids": [],
        "missing_evidence_mappings": [],
        "calibration_performed": False,
        "thresholds_calculated_or_changed": False,
        "calibration_artifacts_written_or_changed": False,
        "answer_generation_performed": False,
        "openai_api_calls_made": 0,
    }


def main() -> None:
    settings = Settings.from_environment(load_env_file=False)
    corpus_dir = PROJECT_ROOT / "data" / "corpus"
    questions_path = PROJECT_ROOT / "data" / "evaluation" / "questions.json"

    documents = load_documents_from_directory(corpus_dir)
    questions_payload = load_questions(questions_path)
    model = load_embedding_model(settings.embedding_model_name, settings.embedding_model_revision)
    per_query_rows, summary_rows = run_retrieval_only_experiments(
        documents,
        questions_payload,
        model,
        rrf_constant=settings.rrf_constant,
    )
    metadata = build_metadata(
        documents,
        questions_payload,
        model,
        settings.rrf_constant,
        per_query_rows,
        summary_rows,
        settings.embedding_model_name,
        settings.embedding_model_revision,
    )

    output_dir = PROJECT_ROOT / "reports" / "public_release"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "retrieval_per_query.csv", per_query_rows)
    write_csv(output_dir / "retrieval_macro_summary.csv", summary_rows)
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Completed {metadata['completed_combinations']} retrieval combinations; "
        f"evaluated {metadata['test_question_count']} held-out test questions "
        f"({metadata['answerable_test_questions']} answerable, "
        f"{metadata['unanswerable_test_questions']} unanswerable)."
    )
    print(f"Wrote retrieval-only reports to {output_dir}.")
    print("No abstention calibration or LLM API calls were performed.")


if __name__ == "__main__":
    main()
