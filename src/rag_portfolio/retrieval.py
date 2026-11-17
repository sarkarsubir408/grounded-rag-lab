from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import TextEncoder, encode_texts
from .models import ChunkRecord, RetrievalHit

TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)
RETRIEVAL_METHODS = ("dense", "bm25", "hybrid")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


@dataclass
class RetrievalIndex:
    records: tuple[ChunkRecord, ...]
    embeddings: np.ndarray
    bm25: BM25Okapi


def build_index(records: list[ChunkRecord], model: TextEncoder) -> RetrievalIndex:
    if not records:
        raise ValueError("Cannot build an index without chunks.")
    ids = [record.chunk_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique within an index.")
    embeddings = encode_texts(model, [record.text for record in records])
    tokenized_corpus = [tokenize(record.text) for record in records]
    if any(not terms for terms in tokenized_corpus):
        raise ValueError("Chunks must contain at least one searchable token.")
    return RetrievalIndex(tuple(records), embeddings, BM25Okapi(tokenized_corpus))


def _rank(scores: np.ndarray, records: tuple[ChunkRecord, ...]) -> list[int]:
    return sorted(range(len(records)), key=lambda index: (-float(scores[index]), records[index].chunk_id))


def reciprocal_rank_fusion(
    rankings: list[list[int]], rrf_constant: int = 60
) -> dict[int, float]:
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive")
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_index in enumerate(ranking, start=1):
            fused[item_index] = fused.get(item_index, 0.0) + 1.0 / (rrf_constant + rank)
    return fused


def search(
    query: str,
    index: RetrievalIndex,
    model: TextEncoder,
    method: str,
    top_k: int = 5,
    rrf_constant: int = 60,
) -> list[RetrievalHit]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if method not in RETRIEVAL_METHODS:
        raise ValueError(f"method must be one of {RETRIEVAL_METHODS}")

    query_vector = encode_texts(model, [query])
    if query_vector.shape[1] != index.embeddings.shape[1]:
        raise ValueError("Query and corpus embedding dimensions do not match.")
    dense_scores = cosine_similarity(query_vector, index.embeddings)[0]
    bm25_scores = np.asarray(index.bm25.get_scores(tokenize(query)), dtype=np.float64)
    dense_order = _rank(dense_scores, index.records)
    bm25_order = _rank(bm25_scores, index.records)
    dense_ranks = {item: rank for rank, item in enumerate(dense_order, start=1)}
    bm25_ranks = {item: rank for rank, item in enumerate(bm25_order, start=1)}

    if method == "dense":
        order = dense_order
        scores = {item: float(dense_scores[item]) for item in order}
    elif method == "bm25":
        order = bm25_order
        scores = {item: float(bm25_scores[item]) for item in order}
    else:
        fused_scores = reciprocal_rank_fusion([dense_order, bm25_order], rrf_constant)
        order = sorted(
            fused_scores,
            key=lambda item: (-fused_scores[item], index.records[item].chunk_id),
        )
        scores = fused_scores

    return [
        RetrievalHit(
            chunk=index.records[item],
            method=method,
            score=float(scores[item]),
            rank=rank,
            dense_score=float(dense_scores[item]),
            bm25_score=float(bm25_scores[item]),
            dense_rank=dense_ranks[item],
            bm25_rank=bm25_ranks[item],
        )
        for rank, item in enumerate(order[:top_k], start=1)
    ]
