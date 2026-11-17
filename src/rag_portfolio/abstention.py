from __future__ import annotations

from collections.abc import Sequence

from .models import RetrievalHit


def abstention_signal(method: str, hits: Sequence[RetrievalHit]) -> float:
    """Return a retriever-specific score feature, not a calibrated probability."""
    if not hits:
        return float("-inf")
    if method == "dense":
        return max(hit.dense_score for hit in hits)
    if method == "bm25":
        return max(hit.bm25_score for hit in hits)
    if method == "hybrid":
        # RRF scores are rank-constant-dependent; calibrate the dense component instead.
        return max(hit.dense_score for hit in hits)
    raise ValueError("method must be dense, bm25, or hybrid")


def _binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, float]:
    tp = sum(label and pred for label, pred in zip(labels, predictions))
    tn = sum(not label and not pred for label, pred in zip(labels, predictions))
    fp = sum(not label and pred for label, pred in zip(labels, predictions))
    fn = sum(label and not pred for label, pred in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def calibrate_threshold(scores: Sequence[float], answerable: Sequence[bool]) -> dict[str, float | int | str]:
    """Select a validation-only threshold maximizing balanced accuracy; ties reduce false alarms."""
    if len(scores) != len(answerable) or len(scores) < 2:
        raise ValueError("Threshold calibration needs paired validation scores and labels.")
    if len(set(answerable)) < 2:
        raise ValueError("Validation labels must include both answerable and unanswerable queries.")
    candidates = sorted(set(float(score) for score in scores))
    candidates = [candidates[0] - 1e-9, *candidates, candidates[-1] + 1e-9]
    evaluations = []
    for threshold in candidates:
        predictions = [float(score) >= threshold for score in scores]
        metrics = _binary_metrics(answerable, predictions)
        evaluations.append((metrics["balanced_accuracy"], -metrics["false_positive_rate"], threshold, metrics))
    _, _, threshold, metrics = max(evaluations, key=lambda item: (item[0], item[1], item[2]))
    return {
        "threshold": float(threshold),
        "objective": "validation balanced accuracy; tie-break minimizes false-positive rate",
        "validation_count": len(scores),
        **metrics,
        "status": "provisional; validation score is a decision signal, not a probability",
    }


def is_sufficient(signal: float, threshold: float) -> bool:
    return signal >= threshold
