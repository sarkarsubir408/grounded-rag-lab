from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .evaluation import CALIBRATION_CONFIGURATION, CALIBRATION_TOP_K

REQUIRED_RETRIEVERS = ("dense", "bm25", "hybrid")


def load_calibration_thresholds(
    path: str | Path,
    *,
    corpus_fingerprint: str,
    embedding_model: str,
    embedding_revision: str,
    rrf_constant: int,
) -> tuple[dict[str, float], str]:
    """Load a complete, corpus-matched calibration artifact or remain uncalibrated."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        return {}, f"calibration artifact not found: {calibration_path}"
    try:
        payload: Any = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, "calibration artifact could not be read as valid JSON"
    if not isinstance(payload, dict):
        return {}, "calibration artifact has an invalid structure"

    if payload.get("corpus_fingerprint") != corpus_fingerprint:
        return {}, "calibration corpus fingerprint does not match the selected corpus"
    if payload.get("embedding_model") != embedding_model:
        return {}, "calibration embedding model does not match the configured model"
    if payload.get("embedding_revision") != embedding_revision:
        return {}, "calibration embedding revision does not match the configured revision"
    if payload.get("calibration_configuration") != {
        "chunk_size": CALIBRATION_CONFIGURATION[0],
        "overlap": CALIBRATION_CONFIGURATION[1],
    }:
        return {}, "calibration chunking configuration does not match the application"
    # Older validated artifacts omitted this field but used K=5. The public-release
    # artifact records it explicitly; retain the existing compatibility behavior.
    if payload.get("calibration_top_k", CALIBRATION_TOP_K) != CALIBRATION_TOP_K:
        return {}, "calibration K does not match the application"
    if payload.get("rrf_constant") != rrf_constant:
        return {}, "calibration RRF constant does not match the application"

    retrievers = payload.get("retrievers")
    if not isinstance(retrievers, dict):
        return {}, "calibration artifact has no retriever thresholds"

    thresholds: dict[str, float] = {}
    for method in REQUIRED_RETRIEVERS:
        result = retrievers.get(method)
        if not isinstance(result, dict) or "threshold" not in result:
            return {}, f"calibration artifact has no {method} threshold"
        raw_threshold = result["threshold"]
        if isinstance(raw_threshold, bool):
            return {}, f"calibration artifact has an invalid {method} threshold"
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            return {}, f"calibration artifact has an invalid {method} threshold"
        if not math.isfinite(threshold):
            return {}, f"calibration artifact has an invalid {method} threshold"
        thresholds[method] = threshold

    return thresholds, str(payload.get("dataset_status", "calibrated validation set"))
