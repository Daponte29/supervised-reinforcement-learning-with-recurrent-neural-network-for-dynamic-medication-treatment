import numpy as np

from src.metrics import AverageMeter, compute_metrics, jaccard_samples, mean_sample_accuracy


def test_average_meter():
    meter = AverageMeter()
    meter.update(2.0)
    meter.update(4.0)
    assert meter.avg == 3.0
    assert meter.count == 2


def test_jaccard_samples_known_value():
    targets = np.array([[1, 1, 0]])
    preds = np.array([[1, 0, 0]])
    # intersection {0} = 1, union {0, 1} = 2 -> 0.5
    assert jaccard_samples(targets, preds) == 0.5


def test_jaccard_handles_empty_sets():
    targets = np.array([[0, 0, 0]])
    preds = np.array([[0, 0, 0]])
    assert jaccard_samples(targets, preds) == 0.0


def test_mean_sample_accuracy():
    targets = np.array([[1, 1, 0]])
    preds = np.array([[1, 0, 0]])
    assert abs(mean_sample_accuracy(targets, preds) - 2 / 3) < 1e-9


def test_compute_metrics_keys():
    targets = np.array([[1, 0], [0, 1]])
    preds = np.array([[1, 0], [0, 0]])
    metrics = compute_metrics(preds, targets)
    assert set(metrics) == {"jaccard", "accuracy"}
    assert 0.0 <= metrics["jaccard"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
