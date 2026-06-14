"""Self-contained binary-classification metrics for arena summaries."""
from __future__ import annotations


def auroc(labels: list[int], scores: list[float]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U) with average ranks for ties."""
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    u = sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def verdict_of(gt_anomaly: bool, pred_anomaly: bool) -> str:
    if gt_anomaly and pred_anomaly:
        return "tp"
    if not gt_anomaly and not pred_anomaly:
        return "tn"
    if not gt_anomaly and pred_anomaly:
        return "fp"
    return "fn"


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("verdict") != "error"]
    labels = [1 if r["ground_truth_anomaly"] else 0 for r in ok]
    preds = [1 if r["is_anomaly"] else 0 for r in ok]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    n = len(ok)
    times = sorted(r["inference_ms"] for r in ok)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)

    # per-defect-type breakdown: which defects get caught vs missed
    by_defect: dict[str, dict] = {}
    for r in ok:
        d = r.get("defect_type", "?")
        e = by_defect.setdefault(d, {"n": 0, "correct": 0, "is_anomaly": bool(r.get("ground_truth_anomaly"))})
        e["n"] += 1
        if bool(r.get("ground_truth_anomaly")) == bool(r.get("is_anomaly")):
            e["correct"] += 1
    for e in by_defect.values():
        e["accuracy"] = e["correct"] / e["n"] if e["n"] else None

    return {
        "n": n,
        "errors": len(results) - n,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc(labels, [r["anomaly_score"] for r in ok]),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "by_defect": by_defect,
        "mean_ms": sum(times) / n if n else None,
        "p95_ms": times[min(int(round(0.95 * n)), n - 1)] if n else None,
    }
