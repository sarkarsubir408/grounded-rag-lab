from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from pypdf import PdfReader

from .models import SourceDocument, TextSegment

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}
MAX_DOCUMENT_BYTES = 15 * 1024 * 1024
PASSAGE_MARKER = re.compile(r"^\s*<!--\s*passage_id:\s*([\w:.-]+)\s*-->\s*$")


class DocumentIngestionError(ValueError):
    """Raised when an uploaded document cannot be safely read as text."""


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result or "document"


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    closing = text.find("\n---", 4)
    if closing < 0:
        return {}, text
    metadata: dict[str, str] = {}
    for line in text[4:closing].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[closing + 4 :].lstrip("\r\n")


def _segments(text: str, document_id: str, page_number: int | None = None) -> tuple[TextSegment, ...]:
    segments: list[TextSegment] = []
    lines: list[str] = []
    heading_prefix: list[str] = []
    pending_id: str | None = None

    def flush() -> None:
        nonlocal lines, pending_id
        content = " ".join(part.strip() for part in lines if part.strip()).strip()
        if content:
            passage_id = pending_id or f"{document_id}:p{len(segments) + 1:04d}"
            segments.append(TextSegment(passage_id, content, page_number))
        lines = []
        pending_id = None

    for line in text.splitlines():
        if match := PASSAGE_MARKER.match(line):
            flush()
            pending_id = match.group(1)
            if heading_prefix:
                lines.extend(heading_prefix)
                heading_prefix.clear()
        elif not line.strip():
            if lines:
                flush()
        elif line.lstrip().startswith("#"):
            flush()
            heading_prefix.append(line.lstrip("# ").strip())
        else:
            if heading_prefix:
                lines.extend(heading_prefix)
                heading_prefix.clear()
            lines.append(line)
    if heading_prefix:
        lines.extend(heading_prefix)
    flush()
    return tuple(segments)


def parse_document(name: str, data: bytes) -> SourceDocument:
    """Parse TXT, Markdown, or text-based PDF bytes without writing uploads to disk."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentIngestionError(
            f"Unsupported file type {suffix or '(no extension)'}. Use TXT, Markdown, or PDF."
        )
    if not data:
        raise DocumentIngestionError(f"{name!r} is empty.")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocumentIngestionError(f"{name!r} exceeds the 15 MiB upload limit.")

    metadata: dict[str, str] = {}
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if reader.is_encrypted:
                raise DocumentIngestionError(f"{name!r} is password-protected.")
            document_id = _slug(Path(name).stem) + "_" + hashlib.sha256(data).hexdigest()[:8]
            pages: list[TextSegment] = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for segment in _segments(page_text, document_id, page_number):
                    pages.append(segment)
            if not pages:
                raise DocumentIngestionError(
                    f"No selectable text was found in {name!r}; scanned PDFs need OCR, which is not supported."
                )
            return SourceDocument(document_id, Path(name).name, tuple(pages))
        except DocumentIngestionError:
            raise
        except Exception as error:
            raise DocumentIngestionError(f"Could not read PDF {name!r}: {error}") from error

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DocumentIngestionError(f"{name!r} must be UTF-8 text.") from error
    metadata, text = _frontmatter(text)
    document_id = metadata.get("doc_id") or f"{_slug(Path(name).stem)}_{hashlib.sha256(data).hexdigest()[:8]}"
    segments = _segments(text, document_id)
    if not segments:
        raise DocumentIngestionError(f"No readable text was found in {name!r}.")
    return SourceDocument(
        document_id=document_id,
        name=Path(name).name,
        segments=segments,
        source_url=(metadata.get("source_url") if (metadata.get("source_url", "").startswith("https://")) else None),
    )


def load_documents_from_directory(directory: str | Path) -> list[SourceDocument]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory not found: {root.resolve()}")
    documents = [
        parse_document(path.name, path.read_bytes())
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not documents:
        raise DocumentIngestionError(f"No supported documents found in {root.resolve()}.")
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        raise DocumentIngestionError("Document IDs must be unique within an index.")
    return documents


def load_uploaded_documents(files: list[tuple[str, bytes]]) -> list[SourceDocument]:
    if not files:
        raise DocumentIngestionError("Upload at least one document.")
    documents = [parse_document(name, data) for name, data in files]
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        raise DocumentIngestionError("Uploaded documents must have unique content/document IDs.")
    return documents


def documents_fingerprint(documents: list[SourceDocument]) -> str:
    """Return a stable digest for the corpus content and provenance used by an index."""
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.document_id):
        digest.update(document.document_id.encode("utf-8"))
        digest.update(document.name.encode("utf-8"))
        digest.update((document.source_url or "").encode("utf-8"))
        for segment in document.segments:
            digest.update(segment.passage_id.encode("utf-8"))
            digest.update((segment.page_number or 0).to_bytes(4, "big"))
            digest.update(segment.text.encode("utf-8"))
    return digest.hexdigest()
