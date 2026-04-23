from __future__ import annotations

import numpy as np


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_f1(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    if not pred.any() and not true.any():
        return 1.0
    if not pred.any() or not true.any():
        return 0.0

    true_positive = np.logical_and(pred, true).sum()
    false_positive = np.logical_and(pred, np.logical_not(true)).sum()
    false_negative = np.logical_and(np.logical_not(pred), true).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def summarize_results(results: list[dict[str, float | str | None]]) -> dict[str, int | float | None]:
    if not results:
        return {
            "num_samples": 0,
            "num_samples_with_gt": 0,
            "mean_iou": None,
            "mean_f1": None,
        }

    with_ground_truth = [item for item in results if item.get("iou") is not None]
    return {
        "num_samples": len(results),
        "num_samples_with_gt": len(with_ground_truth),
        "mean_iou": float(np.mean([item["iou"] for item in with_ground_truth])) if with_ground_truth else None,
        "mean_f1": float(np.mean([item["f1"] for item in with_ground_truth])) if with_ground_truth else None,
    }
