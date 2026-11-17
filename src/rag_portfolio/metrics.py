from __future__ import annotations

import math
from collections.abc import Sequence

from .models import RetrievalHit

DEFAULT_K_VALUES = (3, 5)


def retrieval_metrics(
    hits: Sequence[RetrievalHit], relevant_ids: set[str], k_values: Sequence[int] = DEFAULT_K_VALUES
) -> dict[str, float | None]:
    """Compute binary relevance metrics; empty qrels are not ranking-evaluable."""
    if any(k <= 0 for k in k_values):
        raise ValueError("All K values must be positive.")
    if not relevant_ids:
        return {
            f"{metric}@{k}": None
            for k in k_values
            for metric in ("precision", "recall", "hit_rate", "mrr", "ndcg")
        }

    result: dict[str, float | None] = {}
    hit_ids = [hit.chunk.chunk_id for hit in hits]
    for k in k_values:
        selected = hit_ids[:k]
        found = [chunk_id for chunk_id in selected if chunk_id in relevant_ids]
        denominator = max(1, len(selected))
        result[f"precision@{k}"] = len(found) / denominator
        result[f"recall@{k}"] = len(set(found)) / len(relevant_ids)
        result[f"hit_rate@{k}"] = float(bool(found))
        reciprocal_rank = next((1.0 / rank for rank, chunk_id in enumerate(selected, 1) if chunk_id in relevant_ids), 0.0)
        result[f"mrr@{k}"] = reciprocal_rank
        dcg = sum(1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(selected, 1) if chunk_id in relevant_ids)
        ideal_hits = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        result[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return result


def macro_average(rows: Sequence[dict[str, float | None]]) -> dict[str, float | None]:
    if not rows:
        return {}
    keys = rows[0]
    averages: dict[str, float | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        averages[key] = sum(values) / len(values) if values else None
    return averages
