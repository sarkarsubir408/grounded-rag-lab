from __future__ import annotations

import numpy as np
import pytest

from rag_portfolio.metrics import retrieval_metrics
from rag_portfolio.models import ChunkRecord, RetrievalHit
from rag_portfolio.retrieval import build_index, reciprocal_rank_fusion, search


class TokenVectorEncoder:
    vocabulary = {"cat": 0, "dog": 1, "bird": 2, "animal": 3}

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                if token in self.vocabulary:
                    vectors[row, self.vocabulary[token]] += 1
        return vectors


def records() -> list[ChunkRecord]:
    return [
        ChunkRecord("a:0000", "a", "a.txt", None, 0, "cat animal", ("p1",), ()),
        ChunkRecord("b:0000", "b", "b.txt", None, 0, "dog animal", ("p2",), ()),
        ChunkRecord("c:0000", "c", "c.txt", None, 0, "bird", ("p3",), ()),
    ]


def test_dense_bm25_and_hybrid_retrieval_are_deterministic() -> None:
    model = TokenVectorEncoder()
    index = build_index(records(), model)
    dense = search("cat", index, model, "dense", top_k=2)
    lexical = search("cat", index, model, "bm25", top_k=2)
    hybrid = search("cat", index, model, "hybrid", top_k=2, rrf_constant=60)
    assert dense[0].chunk.chunk_id == "a:0000"
    assert lexical[0].chunk.chunk_id == "a:0000"
    assert hybrid[0].chunk.chunk_id == "a:0000"
    assert hybrid[0].score == pytest.approx(2 / 61)


def test_rrf_uses_one_based_ranks_and_rejects_invalid_constant() -> None:
    fused = reciprocal_rank_fusion([[0, 1], [1, 0]], rrf_constant=1)
    assert fused[0] == pytest.approx(1 / 2 + 1 / 3)
    assert fused[1] == pytest.approx(1 / 3 + 1 / 2)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[0]], rrf_constant=0)


def hit(chunk_id: str, rank: int) -> RetrievalHit:
    chunk = ChunkRecord(chunk_id, "doc", "doc.txt", None, rank, chunk_id, (), ())
    return RetrievalHit(chunk, "dense", 1.0 / rank, rank, 1.0 / rank, 0.0, rank, rank)


def test_retrieval_metrics_have_expected_binary_definitions() -> None:
    values = retrieval_metrics([hit("a", 1), hit("b", 2), hit("c", 3)], {"a", "c"}, (3,))
    assert values["precision@3"] == pytest.approx(2 / 3)
    assert values["recall@3"] == pytest.approx(1.0)
    assert values["hit_rate@3"] == 1.0
    assert values["mrr@3"] == 1.0
    assert values["ndcg@3"] < 1.0


def test_unanswerable_queries_are_not_assigned_retrieval_qrels() -> None:
    assert retrieval_metrics([hit("a", 1)], set(), (3,))["recall@3"] is None
