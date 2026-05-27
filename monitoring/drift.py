"""Data and prediction drift checks."""
from __future__ import annotations

import numpy as np
from scipy import stats


def detect_covariate_drift(ref: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test per feature column."""
    results = {}
    for i in range(ref.shape[1]):
        stat, p = stats.ks_2samp(ref[:, i], current[:, i])
        results[f"feature_{i}"] = {"statistic": stat, "p_value": p, "drift": p < alpha}
    return results


def detect_label_drift(ref_labels: np.ndarray, current_labels: np.ndarray, alpha: float = 0.05) -> dict:
    """Chi-squared test on predicted label distributions."""
    ref_counts = np.bincount(ref_labels)
    cur_counts = np.bincount(current_labels, minlength=len(ref_counts))
    stat, p = stats.chisquare(cur_counts, ref_counts)
    return {"statistic": float(stat), "p_value": float(p), "drift": p < alpha}


if __name__ == "__main__":
    # TODO: load reference and current data, run checks, log results
    pass
