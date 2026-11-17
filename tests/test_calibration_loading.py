from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag_portfolio.calibration import REQUIRED_RETRIEVERS, load_calibration_thresholds
from rag_portfolio.config import Settings
from rag_portfolio.ingestion import documents_fingerprint, load_documents_from_directory
from rag_portfolio.models import ChunkRecord
from rag_portfolio.pipeline import answer_question
from rag_portfolio.retrieval import build_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_THRESHOLDS = {
    "dense": 0.5586917400360107,
    "bm25": 9.107469618115733,
    "hybrid": 0.5586917400360107,
}


def app_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in (
        "ABSTENTION_CALIBRATION_FILE",
        "EMBEDDING_MODEL_NAME",
        "EMBEDDING_MODEL_REVISION",
        "RRF_CONSTANT",
    ):
        monkeypatch.delenv(name, raising=False)
    return Settings.from_environment(load_env_file=False)


def load_project_thresholds(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, float], str]:
    settings = app_settings(monkeypatch)
    documents = load_documents_from_directory(PROJECT_ROOT / "data" / "corpus")
    return load_calibration_thresholds(
        PROJECT_ROOT / settings.calibration_file,
        corpus_fingerprint=documents_fingerprint(documents),
        embedding_model=settings.embedding_model_name,
        embedding_revision=settings.embedding_model_revision,
        rrf_constant=settings.rrf_constant,
    )


class TokenVectorEncoder:
    def encode(self, texts, **kwargs):
        return np.asarray(
            [[1.0, 0.0] if "evidence" in text.lower() else [0.0, 0.0] for text in texts],
            dtype=np.float32,
        )


def test_public_release_calibration_loads_all_expected_thresholds(monkeypatch) -> None:
    settings = app_settings(monkeypatch)
    assert settings.calibration_file == "reports/public_release/abstention_calibration.json"

    thresholds, status = load_project_thresholds(monkeypatch)

    assert set(thresholds) == set(REQUIRED_RETRIEVERS)
    assert thresholds == EXPECTED_THRESHOLDS
    assert "provisional" in status


def test_each_retrieval_method_uses_its_loaded_threshold(monkeypatch) -> None:
    thresholds, _ = load_project_thresholds(monkeypatch)
    records = [
        ChunkRecord(
            "doc:chunk-0000", "doc", "evidence.txt", None, 0,
            "evidence about cats", ("p1",), (),
        ),
        ChunkRecord(
            "doc:chunk-0001", "doc", "evidence.txt", None, 1,
            "unrelated dogs", ("p2",), (),
        ),
    ]
    model = TokenVectorEncoder()
    index = build_index(records, model)

    for method in REQUIRED_RETRIEVERS:
        result = answer_question(
            "evidence about cats", index, model, method, thresholds, provider=None
        )
        assert result.threshold == EXPECTED_THRESHOLDS[method]
        assert result.status in {"abstained", "generation_disabled"}


def test_fingerprint_mismatch_returns_uncalibrated_thresholds(monkeypatch) -> None:
    settings = app_settings(monkeypatch)
    thresholds, status = load_calibration_thresholds(
        PROJECT_ROOT / settings.calibration_file,
        corpus_fingerprint="deliberately-wrong-fingerprint",
        embedding_model=settings.embedding_model_name,
        embedding_revision=settings.embedding_model_revision,
        rrf_constant=settings.rrf_constant,
    )

    assert thresholds == {}
    assert "fingerprint" in status

    model = TokenVectorEncoder()
    index = build_index(
        [
            ChunkRecord(
                "doc:chunk-0000", "doc", "evidence.txt", None, 0,
                "evidence about cats", ("p1",), (),
            )
        ],
        model,
    )
    result = answer_question("evidence about cats", index, model, "dense", thresholds)
    assert result.status == "uncalibrated"
    assert result.abstained
    assert result.retrieved_sources


def test_missing_public_calibration_does_not_fall_back_to_historical_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = app_settings(monkeypatch)
    project_root = tmp_path
    public_path = project_root / settings.calibration_file
    historical_path = project_root / "reports" / "abstention_calibration.json"
    historical_path.parent.mkdir(parents=True)
    historical_path.write_text(
        json.dumps(
            {
                "corpus_fingerprint": "current-corpus",
                "embedding_model": settings.embedding_model_name,
                "embedding_revision": settings.embedding_model_revision,
                "calibration_configuration": {"chunk_size": 100, "overlap": 20},
                "calibration_top_k": 5,
                "rrf_constant": settings.rrf_constant,
                "retrievers": {
                    method: {"threshold": 123.0} for method in REQUIRED_RETRIEVERS
                },
            }
        ),
        encoding="utf-8",
    )

    thresholds, status = load_calibration_thresholds(
        public_path,
        corpus_fingerprint="current-corpus",
        embedding_model=settings.embedding_model_name,
        embedding_revision=settings.embedding_model_revision,
        rrf_constant=settings.rrf_constant,
    )

    assert not public_path.exists()
    assert historical_path.is_file()
    assert thresholds == {}
    assert "not found" in status
