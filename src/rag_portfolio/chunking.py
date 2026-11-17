from __future__ import annotations

from .models import ChunkRecord, SourceDocument


def create_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Preserve the baseline notebook's whitespace-word chunking behavior."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step)]


def create_chunk_records(
    documents: list[SourceDocument], chunk_size: int, overlap: int
) -> list[ChunkRecord]:
    """Chunk each document independently and retain passage/page provenance."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    records: list[ChunkRecord] = []
    for document in documents:
        words: list[str] = []
        owners: list[tuple[str, int | None]] = []
        for segment in document.segments:
            segment_words = segment.text.split()
            words.extend(segment_words)
            owners.extend([(segment.passage_id, segment.page_number)] * len(segment_words))

        step = chunk_size - overlap
        for chunk_index, start in enumerate(range(0, len(words), step)):
            end = min(start + chunk_size, len(words))
            chunk_owners = owners[start:end]
            passage_ids = tuple(dict.fromkeys(passage_id for passage_id, _ in chunk_owners))
            page_numbers = tuple(sorted({page for _, page in chunk_owners if page is not None}))
            records.append(
                ChunkRecord(
                    chunk_id=f"{document.document_id}:chunk-{chunk_index:04d}",
                    document_id=document.document_id,
                    document_name=document.name,
                    source_url=document.source_url,
                    chunk_index=chunk_index,
                    text=" ".join(words[start:end]),
                    passage_ids=passage_ids,
                    page_numbers=page_numbers,
                )
            )
    return records
