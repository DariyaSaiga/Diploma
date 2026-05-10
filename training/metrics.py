"""Multi-label metrics for CMU-MOSEI emotion recognition.

All functions accept logits (raw, pre-sigmoid) and labels in {0,1}.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
)

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class MultiLabelMetrics:
    micro_f1: float
    macro_f1: float
    weighted_f1: float
    samples_f1: float
    hamming_loss: float
    subset_accuracy: float
    per_class_f1: List[float]
    per_class_precision: List[float]
    per_class_recall: List[float]
    threshold: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_metrics(
    logits,
    labels,
    threshold: float = 0.5,
    label_names: Sequence[str] = EMOTIONS,
) -> MultiLabelMetrics:
    """Compute all multi-label metrics from logits + binary labels."""
    logits = to_numpy(logits).astype(np.float32, copy=False)
    if not np.isfinite(logits).all():
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    labels = (to_numpy(labels) > 0.5).astype(np.int32)
    probs = sigmoid_np(logits)
    preds = (probs >= threshold).astype(np.int32)

    return MultiLabelMetrics(
        micro_f1=float(f1_score(labels, preds, average="micro", zero_division=0)),
        macro_f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(labels, preds, average="weighted", zero_division=0)),
        samples_f1=float(f1_score(labels, preds, average="samples", zero_division=0)),
        hamming_loss=float(hamming_loss(labels, preds)),
        subset_accuracy=float(accuracy_score(labels, preds)),
        per_class_f1=f1_score(labels, preds, average=None, zero_division=0).tolist(),
        per_class_precision=precision_score(labels, preds, average=None, zero_division=0).tolist(),
        per_class_recall=recall_score(labels, preds, average=None, zero_division=0).tolist(),
        threshold=float(threshold),
    )


def find_best_thresholds_on_validation(
    logits,
    labels,
    candidates: Sequence[float] = (0.30, 0.35, 0.40, 0.45, 0.50),
    monitor: str = "weighted_f1",
) -> tuple[float, float]:
    """Sweep a few global thresholds; return (best_threshold, best_score) by `monitor`."""
    if monitor not in {"micro_f1", "macro_f1", "weighted_f1", "samples_f1"}:
        raise ValueError(f"Unsupported monitor metric: {monitor}")

    best_t, best_score = 0.5, -1.0
    for t in candidates:
        m = compute_metrics(logits, labels, threshold=t)
        score = getattr(m, monitor)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


def format_per_class_report(metrics: MultiLabelMetrics, label_names: Sequence[str] = EMOTIONS) -> str:
    """Render a small per-class table as a multi-line string."""
    lines = [
        f"Per-class report (threshold={metrics.threshold:.2f}):",
        f"  {'class':<10} {'precision':>10} {'recall':>10} {'f1':>10}",
        f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}",
    ]
    for i, name in enumerate(label_names):
        lines.append(
            f"  {name:<10} "
            f"{metrics.per_class_precision[i]:>10.4f} "
            f"{metrics.per_class_recall[i]:>10.4f} "
            f"{metrics.per_class_f1[i]:>10.4f}"
        )
    return "\n".join(lines)
