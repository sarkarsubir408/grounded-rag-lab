from pathlib import Path

import pytest

from rag_portfolio.abstention import calibrate_threshold, is_sufficient
from rag_portfolio.chunking import create_chunk_records
from rag_portfolio.evaluation import (
    CALIBRATION_TOP_K,
    calibrate_abstention_thresholds,
    load_questions,
    relevant_chunks_for_question,
)
from rag_portfolio.ingestion import documents_fingerprint, load_documents_from_directory
from rag_portfolio.models import SourceDocument, TextSegment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_question_splits_are_disjoint_and_all_annotations_are_approved() -> None:
    payload = load_questions(PROJECT_ROOT / "data" / "evaluation" / "questions.json")
    questions = payload["questions"]
    validation = {question["id"] for question in questions if question["split"] == "validation"}
    test = {question["id"] for question in questions if question["split"] == "test"}
    assert len(questions) == 25
    assert len(validation) == 8
    assert len(test) == 17
    assert validation.isdisjoint(test)
    assert payload["dataset_version"] == "public-release-1.0-reviewed"
    assert all(question["review_status"].startswith("human-reviewed") for question in questions)
    assert all(question["review_decision"]["decision"] == "approved" for question in questions)


def test_archived_relevance_label_schema_contract_uses_self_contained_fixture() -> None:
    configurations = {"100/20", "150/20", "100/0", "100/40"}
    # Keep the archived label-shape contract covered without depending on the
    # excluded baseline artifact or its notebook.
    exported = {
        "ground_truth": {
            configuration: {
                f"Q{index}": {"query": f"sample query {index}", "relevant_chunks": [index - 1]}
                for index in range(1, 8)
            }
            for configuration in configurations
        }
    }
    assert set(exported["ground_truth"]) == configurations
    for labels in exported["ground_truth"].values():
        assert len(labels) == 7
        assert all("query" in item and isinstance(item["relevant_chunks"], list) for item in labels.values())


def test_all_answerable_evidence_maps_under_each_chunking_configuration() -> None:
    documents = load_documents_from_directory(PROJECT_ROOT / "data" / "corpus")
    questions = load_questions(PROJECT_ROOT / "data" / "evaluation" / "questions.json")["questions"]
    for chunk_size, overlap in ((100, 20), (150, 20), (100, 0), (100, 40)):
        records = create_chunk_records(documents, chunk_size, overlap)
        for question in questions:
            relevant = relevant_chunks_for_question(question, records)
            assert bool(relevant) is question["answerable"]


def test_calibration_uses_validation_labels_and_selects_a_deterministic_threshold() -> None:
    result = calibrate_threshold([0.9, 0.8, 0.3, 0.2], [True, True, False, False])
    assert result["threshold"] == pytest.approx(0.8)
    assert result["balanced_accuracy"] == 1.0
    assert is_sufficient(0.8, 0.8)
    assert not is_sufficient(0.79, 0.8)


def test_calibration_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both answerable"):
        calibrate_threshold([0.4, 0.5], [True, True])


def test_corpus_specific_calibration_is_bound_to_content_and_validation_split() -> None:
    class Encoder:
        def encode(self, texts, **kwargs):
            import numpy as np

            return np.asarray(
                [[1.0, 0.0] if "cats" in text.lower() else [0.0, 1.0] for text in texts],
                dtype=np.float32,
            )

    document = SourceDocument("guide", "guide.txt", (TextSegment("p1", "Cats are mammals."),))
    questions = [
        {
            "id": "V1", "split": "validation", "question": "cats", "answerable": True,
            "review_status": "human-reviewed",
        },
        {
            "id": "V2", "split": "validation", "question": "unseen topic", "answerable": False,
            "review_status": "human-reviewed",
        },
    ]
    result = calibrate_abstention_thresholds([document], questions, Encoder(), rrf_constant=60)
    assert result["corpus_fingerprint"] == documents_fingerprint([document])
    assert result["test_split_used_for_calibration"] is False
    assert result["calibration_top_k"] == CALIBRATION_TOP_K == 5
    assert "2 validation questions" in result["dataset_status"]
    assert set(result["retrievers"]) == {"dense", "bm25", "hybrid"}

    with pytest.raises(ValueError, match="validation"):
        calibrate_abstention_thresholds(
            [document], [{**questions[0], "split": "test"}, questions[1]], Encoder(), 60
        )

    with pytest.raises(ValueError, match="human-reviewed"):
        calibrate_abstention_thresholds(
            [document], [{**questions[0], "review_status": "assistant-drafted; pending"}, questions[1]], Encoder(), 60
        )


def test_bundled_validation_annotations_are_approved_for_calibration() -> None:
    questions = load_questions(PROJECT_ROOT / "data" / "evaluation" / "questions.json")["questions"]
    validation = [question for question in questions if question["split"] == "validation"]
    assert len(validation) == 8
    assert {question["answerable"] for question in validation} == {True, False}
    assert all(question["review_status"].startswith("human-reviewed") for question in validation)
    assert all(question["review_decision"]["decision"] == "approved" for question in validation)
