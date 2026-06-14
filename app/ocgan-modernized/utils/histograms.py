from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_score_histogram(
    normal_scores,
    anomaly_scores,
    output_path: str | Path,
    title: str,
    bins: int = 30,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normal_scores = np.asarray(normal_scores, dtype=float)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    plt.figure(figsize=(6, 4))
    if normal_scores.size > 0:
        plt.hist(normal_scores, bins=bins, alpha=0.6, density=False, label="normal")
    if anomaly_scores.size > 0:
        plt.hist(anomaly_scores, bins=bins, alpha=0.6, density=False, label="anomaly")
    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
