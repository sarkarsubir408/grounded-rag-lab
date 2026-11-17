from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_portfolio.ingestion import (
    DocumentIngestionError,
    load_documents_from_directory,
    load_uploaded_documents,
    parse_document,
)


def make_text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_txt_and_markdown_ingestion_preserve_stable_passage_ids() -> None:
    text = parse_document("notes.txt", b"First paragraph.\n\nSecond paragraph.")
    assert text.document_id.startswith("notes_")
    assert len(text.segments) == 2

    markdown = b"""---
doc_id: sample_doc
source_url: https://example.org/doc
---
# Heading
<!-- passage_id: sample-p1 -->
Evidence paragraph.
"""
    parsed = parse_document("sample.md", markdown)
    assert parsed.document_id == "sample_doc"
    assert parsed.source_url == "https://example.org/doc"
    assert parsed.segments[0].passage_id == "sample-p1"
    assert parsed.segments[0].text.startswith("Heading Evidence paragraph.")


def test_plain_text_fallback_id_uses_filename_and_content_hash() -> None:
    source = b"# Notes title\n\nFirst evidence paragraph."
    document = parse_document("chunking-notes.txt", source)
    expected_id = f"chunking_notes_{hashlib.sha256(source).hexdigest()[:8]}"
    assert document.document_id == expected_id
    assert document.segments[0].text == "Notes title First evidence paragraph."


def test_bundled_corpus_uses_expected_document_ids() -> None:
    corpus_dir = Path(__file__).resolve().parents[1] / "data" / "corpus"
    documents = load_documents_from_directory(corpus_dir)
    expected_ids = {
        "bm25_retrieval.md": "bm25_retrieval",
        "chunking_guide.md": "chunking_guide_0dab41fe",
        "dense_embeddings.md": "dense_embeddings",
        "hybrid_rrf.md": "hybrid_rrf",
        "rag_workflow.md": "rag_workflow",
        "retrieval_evaluation.md": "retrieval_evaluation",
    }
    assert {document.name: document.document_id for document in documents} == expected_ids


def test_pdf_text_is_extracted_with_page_metadata() -> None:
    document = parse_document("sample.pdf", make_text_pdf("retrieval evidence"))
    assert "retrieval evidence" in document.segments[0].text
    assert document.segments[0].page_number == 1


def test_scanned_or_empty_pdf_is_rejected_with_ocr_limit() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(DocumentIngestionError, match="OCR"):
        parse_document("blank.pdf", output.getvalue())


def test_multiple_uploads_and_unsupported_files() -> None:
    docs = load_uploaded_documents([("one.txt", b"one"), ("two.md", b"two")])
    assert len(docs) == 2
    with pytest.raises(DocumentIngestionError, match="Unsupported"):
        parse_document("image.png", b"not text")


def test_file_size_limit_is_enforced_with_clear_error() -> None:
    with pytest.raises(DocumentIngestionError, match=r"exceeds the 15 MiB upload limit"):
        parse_document("huge.txt", b"x" * (15 * 1024 * 1024 + 1))
