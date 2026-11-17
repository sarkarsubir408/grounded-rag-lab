from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray: ...


def load_embedding_model(model_name: str = MODEL_NAME, revision: str = MODEL_REVISION) -> TextEncoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, revision=revision)


def encode_texts(model: TextEncoder, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        raise ValueError("At least one text is required for encoding.")
    vectors = np.asarray(model.encode(list(texts), show_progress_bar=False), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("The encoder must return one 2D vector per input text.")
    if not np.isfinite(vectors).all():
        raise ValueError("The encoder returned non-finite values.")
    return vectors
