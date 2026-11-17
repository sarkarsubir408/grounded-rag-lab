from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    passage_id: str
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    name: str
    segments: tuple[TextSegment, ...]
    source_url: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    document_name: str
    source_url: str | None
    chunk_index: int
    text: str
    passage_ids: tuple[str, ...]
    page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class RetrievalHit:
    chunk: ChunkRecord
    method: str
    score: float
    rank: int
    dense_score: float
    bm25_score: float
    dense_rank: int
    bm25_rank: int


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    document_name: str
    chunk_id: str
    page_numbers: tuple[int, ...]
    source_url: str | None
