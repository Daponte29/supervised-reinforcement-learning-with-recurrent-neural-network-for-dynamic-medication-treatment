from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score


class AverageMeter:
    """Tracks the running average of a streamed value."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jaccard_samples(targets: np.ndarray, preds: np.ndarray) -> float:
    """Sample-averaged Jaccard (IoU) over multi-label medication sets."""
    return float(jaccard_score(targets, preds, average="samples", zero_division=0.0))


def mean_sample_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    """Mean per-sample subset accuracy across the batch."""
    return float(np.mean([accuracy_score(targets[i], preds[i]) for i in range(len(targets))]))


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Aggregate Jaccard and accuracy from already-binarized predictions."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return {
        "jaccard": jaccard_samples(targets, preds),
        "accuracy": mean_sample_accuracy(targets, preds),
    }
