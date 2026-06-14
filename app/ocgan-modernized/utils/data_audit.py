from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def tensor_sha1(x: torch.Tensor) -> str:
    x = x.detach().cpu().contiguous()
    return hashlib.sha1(x.numpy().tobytes()).hexdigest()


def find_exact_duplicates(samples: list[dict[str, Any]]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for i, sample in enumerate(samples):
        h = tensor_sha1(sample["image"])
        buckets.setdefault(h, []).append(i)
    return {k: v for k, v in buckets.items() if len(v) > 1}


def audit_split_contamination(
    split_to_samples: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    split_hashes: dict[str, set[str]] = {}
    for split_name, samples in split_to_samples.items():
        split_hashes[split_name] = {tensor_sha1(s["image"]) for s in samples}

    overlaps: dict[str, int] = {}
    split_names = list(split_hashes.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            inter = split_hashes[a].intersection(split_hashes[b])
            overlaps[f"{a}__{b}"] = len(inter)

    return overlaps


def summarize_labels(samples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(s["label"]) for s in samples]
    anomalies = [int(s["is_anomaly"]) for s in samples]
    return {
        "num_samples": len(samples),
        "label_counts": dict(Counter(labels)),
        "anomaly_count": int(sum(anomalies)),
    }


def compute_channel_stats(samples: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    xs = torch.stack([s["image"] for s in samples], dim=0)  # [N,C,H,W]
    mean = xs.mean(dim=(0, 2, 3))
    std = xs.std(dim=(0, 2, 3))
    return mean.tolist(), std.tolist()
