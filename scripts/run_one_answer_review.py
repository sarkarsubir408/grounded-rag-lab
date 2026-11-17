from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_portfolio.chunking import create_chunk_records
from rag_portfolio.config import Settings
from rag_portfolio.embeddings import load_embedding_model
from rag_portfolio.evaluation import load_questions
from rag_portfolio.generation import OpenAIResponsesProvider
from rag_portfolio.ingestion import load_documents_from_directory
from rag_portfolio.pipeline import answer_question
from rag_portfolio.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one answer for human review. This can incur an OpenAI API charge."
    )
    parser.add_argument("--question-id", required=True, help="One held-out test question ID, such as T01.")
    parser.add_argument(
        "--authorize-billable-call",
        action="store_true",
        help="Explicitly authorize this one API call. The environment flag must also be enabled.",
    )
    args = parser.parse_args()
    if not args.authorize_billable_call:
        parser.error("No API request was sent. Pass --authorize-billable-call to authorize one billable request.")

    settings = Settings.from_environment()
    provider = OpenAIResponsesProvider.from_settings(settings, user_authorized_this_call=True)
    questions_payload = load_questions(PROJECT_ROOT / "data" / "evaluation" / "questions.json")
    question = next(
        (item for item in questions_payload["questions"] if item["id"] == args.question_id), None
    )
    if question is None or question["split"] != "test":
        parser.error("question-id must identify one question in the held-out test split.")

    calibration = json.loads(
        (PROJECT_ROOT / settings.calibration_file).read_text(encoding="utf-8")
    )
    thresholds = {
        method: values["threshold"]
        for method, values in calibration["retrievers"].items()
    }
    documents = load_documents_from_directory(PROJECT_ROOT / "data" / "corpus")
    model = load_embedding_model(settings.embedding_model_name, settings.embedding_model_revision)
    records = create_chunk_records(documents, chunk_size=100, overlap=20)
    index = build_index(records, model)
    result = answer_question(
        question["question"], index, model, method="hybrid", thresholds=thresholds,
        provider=provider, top_k=5, rrf_constant=settings.rrf_constant,
    )
    record = {
        "question_id": question["id"],
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "gold_evidence": question["evidence"],
        "annotation_status": question["review_status"],
        "model_name": settings.openai_model,
        "answer_result": result.as_dict(),
        "review": {
            "correctness_rating_1_to_5": None,
            "contextual_relevance_rating_1_to_5": None,
            "faithfulness_rating_1_to_5": None,
            "reviewer": None,
            "review_date": None,
            "notes": None,
        },
    }
    output_path = PROJECT_ROOT / "reports" / "answer_review_records.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    print(f"Review record appended to {output_path}.")


if __name__ == "__main__":
    main()
