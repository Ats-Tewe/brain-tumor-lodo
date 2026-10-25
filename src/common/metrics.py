from __future__ import annotations

import numpy as np

from common.config import ECE_N_BINS

def confidence_bins(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_N_BINS,
):
    """
    Bin predictions by confidence.

    Returns (bin_lowers, bin_uppers, bin_accuracy, bin_confidence, bin_count),
    each of length n_bins. Empty bins carry accuracy = confidence = nan and
    count = 0, so plotting code can skip them rather than drawing a spurious
    point at the origin.
    """
    probs = np.asarray(probs)
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    confidences = probs.max(axis=1)
    correct = (preds == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    lowers, uppers = edges[:-1], edges[1:]

    acc = np.full(n_bins, np.nan)
    conf = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(lowers, uppers)):
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        n = int(mask.sum())
        count[i] = n
        if n:
            acc[i] = correct[mask].mean()
            conf[i] = confidences[mask].mean()

    return lowers, uppers, acc, conf, count

def compute_ece(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> float:
    """
    Expected Calibration Error (paper Section IV-F, Eq. 6).

        ECE = sum_b (n_b / N) * | acc(b) - conf(b) |
    """
    _, _, acc, conf, count = confidence_bins(probs, preds, labels, n_bins)
    total = count.sum()
    if total == 0:
        return float("nan")
    mask = count > 0
    return float(np.sum(count[mask] / total * np.abs(acc[mask] - conf[mask])))

def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Mean NLL, reported alongside ECE because temperature scaling optimizes NLL."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, eps, 1.0)).mean())

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score. A proper scoring rule, useful as an ECE cross-check."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())
