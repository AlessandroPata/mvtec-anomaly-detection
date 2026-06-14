"""Historical PatchCore variants reconstructed from the production memory banks.

A variant is the production bank restricted to a k-center coreset plus the
aggregation settings of that era (configs read off the eval CSVs):
  v1 = coreset 10000, topk_mean  k=3   (logs/patchcore_pure.csv)
  v2 = coreset 10000, topk_reweighted k=9 (logs/patchcore_v2.csv)
Thresholds are recalibrated offline (scripts/calibrate_variant_thresholds.py)
on the same val_normal split (seed 43, 15%) as the original export, p99.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from models.patchcore_common import kcenter_greedy_select

CORESET_K = 10000
# Same pool cap the production export used for its (rare) coreset builds.
CANDIDATE_POOL_SIZE = 20000


@dataclass(frozen=True)
class VariantSpec:
    id: str
    label: str
    kind: str                  # "production" | "reconstructed"
    aggregation: str | None    # None → keep checkpoint settings
    topk: int | None
    coreset: int | None        # None → full bank
    description: str


VARIANT_SPECS: dict[str, VariantSpec] = {
    "production": VariantSpec(
        "production", "Production — PatchCore v3", "production", None, None, None,
        "Full memory bank (≤70k patches), topk_reweighted k=9. The shipped model.",
    ),
    "patchcore_v2": VariantSpec(
        "patchcore_v2", "PatchCore v2 — reconstructed", "reconstructed",
        "topk_reweighted", 9, CORESET_K,
        "k-center coreset 10k of the production bank, topk_reweighted k=9.",
    ),
    "patchcore_v1": VariantSpec(
        "patchcore_v1", "PatchCore v1 — reconstructed", "reconstructed",
        "topk_mean", 3, CORESET_K,
        "k-center coreset 10k of the production bank, topk_mean k=3.",
    ),
}

# screw's production bank uses layer1+2+3 features; the original v1/v2 ran on
# layer2+3, so its reconstructions are approximate.
APPROXIMATE_CATEGORIES = {"screw"}


def coreset_indices_path(models_dir: Path, category: str, k: int) -> Path:
    return Path(models_dir) / category / "variants" / f"coreset{k}_idx.pt"


def get_coreset_indices(models_dir: Path, category: str, bank: torch.Tensor, k: int = CORESET_K) -> torch.Tensor:
    path = coreset_indices_path(models_dir, category, k)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    idx = kcenter_greedy_select(bank, k, init="mean",
                                candidate_pool_size=CANDIDATE_POOL_SIZE).cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(idx, path)
    return idx


def load_calibration(models_dir: Path) -> dict:
    p = Path(models_dir) / "variant_thresholds.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def variant_stats(scores: list[float]) -> dict:
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "threshold": float(np.percentile(arr, 99)),
        "score_mean": float(arr.mean()),
        "score_std": float(arr.std()),
        "n_val": int(arr.size),
    }


def available_variants(category: str, calibration: dict) -> list[dict]:
    out = []
    for spec in VARIANT_SPECS.values():
        if spec.kind == "production":
            available = True
        else:
            available = calibration.get(category, {}).get(spec.id) is not None
        out.append({
            **asdict(spec),
            "available": available,
            "approximate": spec.kind == "reconstructed" and category in APPROXIMATE_CATEGORIES,
        })
    return out


def build_variant_model(base, spec: VariantSpec, calibration_entry: dict | None, models_dir: Path):
    """base: PatchCoreInference (or any object with bank/aggregation/topk/threshold/
    train_score_mean/train_score_std). Returns base itself for production, else a
    shallow copy sharing the backbone but with bank subset + era settings."""
    if spec.kind == "production":
        return base
    if calibration_entry is None:
        raise ValueError(f"Variant {spec.id} not calibrated for {base.category}")
    m = copy.copy(base)
    idx = get_coreset_indices(models_dir, base.category, base.bank, spec.coreset)
    m.bank = base.bank[idx.to(base.bank.device)]
    m.aggregation = spec.aggregation
    m.topk = spec.topk
    m.threshold = float(calibration_entry["threshold"])
    m.train_score_mean = float(calibration_entry["score_mean"])
    m.train_score_std = float(calibration_entry["score_std"])
    return m
