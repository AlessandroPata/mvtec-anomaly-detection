from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


@dataclass
class ImageLevelMetrics:
    auroc: float
    auprc: float
    best_f1: float
    best_threshold: float


def compute_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) == 0:
        return 0.0, 0.5

    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_values))
    best_f1 = float(f1_values[best_idx])
    best_threshold = float(thresholds[best_idx])
    return best_f1, best_threshold


def compute_image_level_metrics(y_true: np.ndarray, y_score: np.ndarray) -> ImageLevelMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    unique = np.unique(y_true)
    if len(unique) < 2:
        return ImageLevelMetrics(
            auroc=float("nan"),
            auprc=float("nan"),
            best_f1=float("nan"),
            best_threshold=0.5,
        )

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    best_f1, best_threshold = compute_best_f1_threshold(y_true, y_score)

    return ImageLevelMetrics(
        auroc=auroc,
        auprc=auprc,
        best_f1=best_f1,
        best_threshold=best_threshold,
    )


def compute_f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred))
