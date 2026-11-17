from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rag_portfolio.abstention as abstention
import rag_portfolio.evaluation as evaluation
from rag_portfolio.models import SourceDocument, TextSegment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_retrieval_only_experiments as retrieval_only


class SmallEncoder:
    model_name = "test-encoder"
    revision = "test-revision"

    def encode(self, texts, **kwargs):
        vocabulary = {"alpha": 0, "fact": 1, "beta": 2, "context": 3}
        vectors = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                if token in vocabulary:
                    vectors[row, vocabulary[token]] += 1.0
        return vectors

    def get_embedding_dimension(self):
        return 4


def _fail_if_called(*args, **kwargs):
    raise AssertionError("Retrieval-only execution must not call calibration or run_benchmark.")


def test_main_runs_retrieval_only_without_calibration_or_calibration_file_changes(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    reports = project_root / "reports"
    reports.mkdir(parents=True)
    protected_artifacts = {
        "abstention_calibration.json": b"saved-thresholds\n",
        "abstention_calibration_reviewed_candidate.json": b"review-candidate\n",
        "abstention_threshold_comparison.json": b"threshold-comparison\n",
    }
    for name, content in protected_artifacts.items():
        (reports / name).write_bytes(content)

    document = SourceDocument(
        "guide",
        "guide.md",
        (TextSegment("p1", "alpha fact"), TextSegment("p2", "beta context")),
    )
    questions_payload = {
        "dataset_version": "test-dataset",
        "questions": [
            {
                "id": "C01",
                "split": "validation",
                "question": "validation-only query",
                "answerable": True,
                "evidence": [{"doc_id": "guide", "passage_id": "p1"}],
            },
            {
                "id": "T01",
                "split": "test",
                "question": "alpha",
                "answerable": True,
                "evidence": [{"doc_id": "guide", "passage_id": "p1"}],
            },
            {
                "id": "T02",
                "split": "test",
                "question": "unknown moon",
                "answerable": False,
                "evidence": [],
            },
        ],
    }

    monkeypatch.setattr(retrieval_only, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        retrieval_only,
        "Settings",
        SimpleNamespace(
            from_environment=lambda load_env_file=False: SimpleNamespace(
                embedding_model_name="test-encoder",
                embedding_model_revision="test-revision",
                rrf_constant=60,
            )
        ),
    )
    monkeypatch.setattr(retrieval_only, "load_documents_from_directory", lambda _path: [document])
    monkeypatch.setattr(retrieval_only, "load_questions", lambda _path: questions_payload)
    monkeypatch.setattr(retrieval_only, "load_embedding_model", lambda *_args: SmallEncoder())
    monkeypatch.setattr(evaluation, "calibrate_abstention_thresholds", _fail_if_called)
    monkeypatch.setattr(evaluation, "run_benchmark", _fail_if_called)
    monkeypatch.setattr(abstention, "calibrate_threshold", _fail_if_called)

    retrieval_only.main()

    output_dir = reports / "public_release"
    assert {path.name for path in output_dir.iterdir()} == {
        "retrieval_per_query.csv",
        "retrieval_macro_summary.csv",
        "experiment_metadata.json",
    }
    with (output_dir / "retrieval_per_query.csv").open(encoding="utf-8", newline="") as handle:
        per_query = list(csv.DictReader(handle))
    with (output_dir / "retrieval_macro_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    metadata = json.loads((output_dir / "experiment_metadata.json").read_text(encoding="utf-8"))

    assert len(per_query) == 4 * 3 * 2 * 2
    assert {row["question_id"] for row in per_query} == {"T01", "T02"}
    assert all(row["split"] == "test" for row in per_query)
    assert len(summary) == 4 * 3 * 2
    unanswerable_rows = [row for row in per_query if row["question_id"] == "T02"]
    assert all(row["precision@3"] == "" and row["recall@5"] == "" for row in unanswerable_rows)
    assert metadata["completed_combinations"] == 12
    assert metadata["calibration_performed"] is False
    assert metadata["thresholds_calculated_or_changed"] is False
    assert metadata["calibration_artifacts_written_or_changed"] is False
    assert metadata["answer_generation_performed"] is False

    for name, original_content in protected_artifacts.items():
        assert (reports / name).read_bytes() == original_content
