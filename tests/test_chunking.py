from pathlib import Path

import pytest

from rag_portfolio.chunking import create_chunk_records, create_chunks
from rag_portfolio.ingestion import load_documents_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_corpus_chunks_with_all_four_configurations() -> None:
    documents = load_documents_from_directory(PROJECT_ROOT / "data" / "corpus")
    assert len(documents) == 6

    for chunk_size, overlap in ((100, 20), (150, 20), (100, 0), (100, 40)):
        records = create_chunk_records(documents, chunk_size, overlap)
        assert records
        assert len({record.chunk_id for record in records}) == len(records)
        assert {record.document_id for record in records} == {
            document.document_id for document in documents
        }
        assert all(0 < len(record.text.split()) <= chunk_size for record in records)


def test_create_chunks_keeps_baseline_boundaries() -> None:
    assert create_chunks("a b c d e f", 4, 1) == ["a b c d", "d e f"]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"), [(0, 0), (5, -1), (5, 5), (5, 7)]
)
def test_invalid_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        create_chunks("some words", chunk_size, overlap)
