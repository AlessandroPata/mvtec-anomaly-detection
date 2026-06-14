"""Deterministic stratified sampling of MVTec test images for arena runs."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@dataclass(frozen=True, order=True)
class TestImage:
    __test__ = False  # not a pytest class despite the Test* name

    defect: str
    filename: str

    @property
    def is_anomaly(self) -> bool:
        return self.defect != "good"


def list_test_images(dataset_root: Path, category: str) -> dict[str, list[TestImage]]:
    test_dir = Path(dataset_root) / category / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")
    groups: dict[str, list[TestImage]] = {}
    for d in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        imgs = [TestImage(d.name, f.name) for f in sorted(d.iterdir())
                if f.suffix.lower() in IMAGE_EXTS]
        if imgs:
            groups[d.name] = imgs
    return groups


def sample_test_images(dataset_root: Path, category: str, n: int, seed: int) -> list[TestImage]:
    """Proportional stratified sample (largest-remainder), ≥1 per defect type
    when n allows, deterministic for a given seed."""
    groups = list_test_images(dataset_root, category)
    names = sorted(groups)
    total = sum(len(v) for v in groups.values())
    n = min(n, total)

    quotas = {g: n * len(groups[g]) / total for g in names}
    alloc = {g: int(quotas[g]) for g in names}
    if n >= len(names):
        for g in names:
            alloc[g] = max(alloc[g], 1)

    # fix rounding so sum(alloc) == n, never exceeding group sizes
    remainder_order = sorted(names, key=lambda g: (quotas[g] - int(quotas[g])), reverse=True)
    guard = 0
    while sum(alloc.values()) != n and guard < 10 * len(names) + n:
        guard += 1
        diff = n - sum(alloc.values())
        if diff > 0:
            for g in remainder_order:
                if alloc[g] < len(groups[g]):
                    alloc[g] += 1
                    break
        else:
            for g in sorted(names, key=lambda g: alloc[g], reverse=True):
                if alloc[g] > (1 if n >= len(names) else 0):
                    alloc[g] -= 1
                    break

    rng = random.Random(seed)
    picked: list[TestImage] = []
    for g in names:
        take = min(alloc[g], len(groups[g]))
        if take:
            picked.extend(rng.sample(groups[g], take))
    rng.shuffle(picked)
    return picked
