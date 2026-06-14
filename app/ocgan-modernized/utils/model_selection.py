from __future__ import annotations

from typing import Any


def compute_selection_score(cfg: Any, val_metrics: dict[str, float]) -> float:
    split = cfg.model_selection.monitor_split

    if not cfg.model_selection.use_composite_score:
        key = f"{split}_auroc"
        return float(val_metrics[key])

    weights = cfg.model_selection.composite_weights
    auroc = float(val_metrics[f"{split}_auroc"])
    auprc = float(val_metrics[f"{split}_auprc"])

    score = float(weights.auroc) * auroc + float(weights.auprc) * auprc

    # Optional FPR@95TPR penalty (lower FPR = better, so we add (1 - fpr))
    fpr_weight = float(getattr(weights, "fpr95_penalty", 0.0))
    if fpr_weight > 0.0:
        fpr = float(val_metrics.get(f"{split}_fpr_at_95_tpr", 1.0))
        score += fpr_weight * (1.0 - fpr)
    else:
        best_f1 = float(val_metrics.get(f"{split}_best_f1", 0.0))
        score += float(getattr(weights, "best_f1", 0.2)) * best_f1

    return float(score)
