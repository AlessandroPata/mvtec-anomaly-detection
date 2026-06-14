# MVTec Webapp — Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing FastAPI server (`ocgan-modernized/server.py`) with model-variant inference (Production + reconstructed PatchCore v1/v2), a streaming batch "Test Arena" API, dataset thumbnails/masks, metadata, and regenerated benchmark JSON for the frontend.

**Architecture:** All new Python code lives in `ocgan-modernized/` (a git repo). A new `webapp/` package holds arena logic (sampler, metrics, jobs); `models/patchcore_common.py` holds scoring utilities shared between inference and scripts; `models/patchcore_variants.py` reconstructs historical PatchCore configs from the production banks via shallow-copied `PatchCoreInference` instances. Thresholds for reconstructed variants are recalibrated offline with the same 85/15 val_normal split (seed 43, p99) used by the original export.

**Tech Stack:** Python 3.13, PyTorch (+CUDA cu126 if the Quadro T1000 works, else CPU), FastAPI/uvicorn, pytest. No sklearn, no SSE library (plain `StreamingResponse`).

**Working directory for all commands:** `D:\OCGAN\project\storage_project_outputs_datasets\project\ocgan-modernized` (PowerShell). The venv lives at `.venv\`; activate once per shell with `.\.venv\Scripts\Activate.ps1`.

**Key existing facts (verified):**
- `PatchCoreInference` (models/patchcore_inference.py) loads `production_models/{cat}/patchcore_bank.pt` with keys `bank, feature_level, aggregation, topk, image_size, threshold, train_score_mean, train_score_std, backbone`; `predict(pil)` returns dict with `anomaly_score, anomaly_probability, is_anomalous, threshold, category, inference_time_ms, score_components, heatmap (np [H,W] 0..1), reconstruction`.
- `scripts/patchcore_pure.py:59` has `kcenter_greedy_select(features, k, init="mean", candidate_pool_size=None)`; `:120` has `aggregate_image_score(min_dists, aggregation, topk)`. `PatchCoreInference._aggregate` duplicates the same formulas.
- Variant configs from the result CSVs: **v1** = `topk_mean k=3, coreset=10000` (logs/patchcore_pure.csv), **v2** = `topk_reweighted k=9, coreset=10000` (logs/patchcore_v2.csv), **v3/production** = `topk_reweighted k=9, coreset=70000=full` (logs/patchcore_v3.csv).
- Export constants (scripts/export_patchcore_banks.py:34-55): `SEED=43`, `VAL_NORMAL_RATIO=0.15`, `IMAGE_SIZE=256`, `CANDIDATE_POOL_SIZE=20000`, dataset class `MVTecADDataset(root, category, split="val_normal", image_size, image_transform, val_normal_ratio, val_mixed_ratio=0.0, seed)`, transform = `Compose([ResizePadToSquare(256), NormalizeTensor(ImageNet)])`.
- `server.py` globals: `PRODUCTION_MODELS_DIR`, `DATASET_ROOT = PROJECT_ROOT.parent.parent / "datasets" / "mvtec_ad"` (exists, 15 categories), `CATEGORIES`, `get_model(category)`, `_model_cache`, `_device`.
- Tests live in `tests/` (conftest.py, test_imports.py, …), `pytest.ini` at repo root.

---

### Task 0: Python environment + real-bank smoke test

**Files:**
- Create: `requirements-webapp.txt`
- Create: `scripts/smoke_predict.py`

- [ ] **Step 0.1: Create venv**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
Expected: pip upgraded, prompt shows `(.venv)`.

- [ ] **Step 0.2: Install torch — try CUDA first, fall back to CPU**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```
Expected: `2.x.x+cu126 True Quadro T1000 with Max-Q Design`.
If the install fails (no cp313 cu126 wheel) or `is_available()` is False: `pip uninstall -y torch torchvision; pip install torch torchvision` (CPU wheels) — everything below works on CPU, just slower.

- [ ] **Step 0.3: Write `requirements-webapp.txt`**

```text
# Webapp/server dependencies (torch/torchvision installed separately — see README)
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
omegaconf>=2.3
pillow>=10
numpy
pytest>=8
httpx>=0.27
```

- [ ] **Step 0.4: Install them**

```powershell
pip install -r requirements-webapp.txt
```
Expected: all resolve. (If `numpy` 2.x conflicts with anything at import time, pin `numpy<2` — the README pinned <2 only for wandb, which the server does not import.)

- [ ] **Step 0.5: Write `scripts/smoke_predict.py`**

```python
"""Smoke test: load one production bank and predict a good + a defect image."""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.patchcore_inference import PatchCoreInference  # noqa: E402

DATASET_ROOT = ROOT.parent.parent / "datasets" / "mvtec_ad"


def first_image(folder: Path) -> Path:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")[0]


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    cat = "bottle"
    model = PatchCoreInference(cat, ROOT / "production_models" / cat / "patchcore_bank.pt", device=device)

    good = first_image(DATASET_ROOT / cat / "test" / "good")
    defect_dir = sorted(d for d in (DATASET_ROOT / cat / "test").iterdir() if d.is_dir() and d.name != "good")[0]
    defect = first_image(defect_dir)

    r_good = model.predict(Image.open(good).convert("RGB"))
    r_def = model.predict(Image.open(defect).convert("RGB"))

    print(f"good   {good.name}: score={r_good['anomaly_score']:.4f} thr={r_good['threshold']:.4f} anomalous={r_good['is_anomalous']} ({r_good['inference_time_ms']:.0f} ms)")
    print(f"defect {defect_dir.name}/{defect.name}: score={r_def['anomaly_score']:.4f} anomalous={r_def['is_anomalous']} ({r_def['inference_time_ms']:.0f} ms)")
    assert r_def["anomaly_score"] > r_good["anomaly_score"], "defect must score higher than good"
    assert r_def["is_anomalous"], "defect must be flagged anomalous"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 0.6: Run it (use `cuda` if Step 0.2 said True, else `cpu`)**

```powershell
python scripts\smoke_predict.py cuda
```
Expected: two score lines, defect > good, `SMOKE OK`. First run downloads wide_resnet50_2 weights (~132 MB) to the torch cache. Record the two scores — Task 1 reuses them as a regression guard.
If CUDA OOMs loading the bank (4 GB card): rerun with `cpu` and use `--device cpu` for the server later; per-category OOM fallback is implemented in Task 7.

- [ ] **Step 0.7: Commit**

```powershell
git add requirements-webapp.txt scripts/smoke_predict.py .gitignore
git commit -m "chore: webapp env requirements + production-bank smoke test"
```
(Add `.venv/` and `.thumb_cache/` to `.gitignore` first if not already ignored.)

---

### Task 1: Shared scoring utils — `models/patchcore_common.py` (TDD)

Extract `aggregate_image_score` + `kcenter_greedy_select` into one importable module so inference, variants, and calibration all use literally the same code (today it's duplicated between `patchcore_inference.py` and `scripts/patchcore_pure.py`; scripts/ has no `__init__.py` and stays untouched).

**Files:**
- Create: `models/patchcore_common.py`
- Create: `tests/test_patchcore_common.py`
- Modify: `models/patchcore_inference.py:96-110` (`_aggregate` delegates)

- [ ] **Step 1.1: Write the failing test**

`tests/test_patchcore_common.py`:
```python
import math

import pytest
import torch

from models.patchcore_common import aggregate_image_score, kcenter_greedy_select


class TestAggregate:
    def test_topk_mean(self):
        d = torch.tensor([[3.0, 1.0, 2.0]])
        assert aggregate_image_score(d, "topk_mean", 2).item() == pytest.approx(2.5)

    def test_topk_reweighted_hand_computed(self):
        # d=[4,2], k=2 → inv=[0.25,0.5]; softmax(inv)=[e^.25, e^.5]/Z → w=1-softmax
        d = torch.tensor([[4.0, 2.0]])
        e25, e50 = math.exp(0.25), math.exp(0.5)
        s4, s2 = e25 / (e25 + e50), e50 / (e25 + e50)
        w4, w2 = 1 - s4, 1 - s2
        expected = (4 * w4 + 2 * w2) / (w4 + w2)
        got = aggregate_image_score(d, "topk_reweighted", 2).item()
        assert got == pytest.approx(expected, rel=1e-5)

    def test_mean_and_max(self):
        d = torch.tensor([[1.0, 2.0, 3.0]])
        assert aggregate_image_score(d, "mean", 0).item() == pytest.approx(2.0)
        assert aggregate_image_score(d, "max", 0).item() == pytest.approx(3.0)

    def test_k_clamped_to_patches(self):
        d = torch.tensor([[1.0, 2.0]])
        assert aggregate_image_score(d, "topk_mean", 99).item() == pytest.approx(1.5)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            aggregate_image_score(torch.ones(1, 3), "nope", 1)


class TestKCenter:
    def test_deterministic_and_unique(self):
        torch.manual_seed(0)
        x = torch.randn(40, 4)
        a = kcenter_greedy_select(x, 8)
        b = kcenter_greedy_select(x, 8)
        assert torch.equal(a, b)
        assert len(set(a.tolist())) == 8

    def test_first_pick_is_farthest_from_mean(self):
        x = torch.tensor([[0.0], [1.0], [2.0], [10.0]])
        idx = kcenter_greedy_select(x, 2)
        assert idx[0].item() == 3  # 10.0 is farthest from mean 3.25
        assert idx[1].item() == 0  # then 0.0 is farthest from 10.0

    def test_k_ge_n_returns_all(self):
        x = torch.randn(5, 3)
        assert kcenter_greedy_select(x, 9).tolist() == [0, 1, 2, 3, 4]

    def test_candidate_pool_returns_valid_indices(self):
        x = torch.randn(100, 3)
        idx = kcenter_greedy_select(x, 5, candidate_pool_size=20)
        assert len(idx) == 5
        assert idx.max().item() < 100
```

- [ ] **Step 1.2: Run to verify it fails**

```powershell
python -m pytest tests/test_patchcore_common.py -v
```
Expected: `ModuleNotFoundError: No module named 'models.patchcore_common'` (collection error).

- [ ] **Step 1.3: Implement `models/patchcore_common.py`**

Copy the two functions **verbatim** from `scripts/patchcore_pure.py:58-135` (they are the reference used by every eval run):

```python
"""Shared PatchCore scoring utilities.

These are the reference implementations from scripts/patchcore_pure.py (the
evaluation script). Inference, variant reconstruction, and threshold
calibration import from here so live scores match the published eval exactly.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def aggregate_image_score(min_dists: torch.Tensor, aggregation: str, topk: int) -> torch.Tensor:
    """min_dists: [B, P]  ->  [B] image-level score."""
    b, p = min_dists.shape
    if aggregation == "topk_mean":
        k = min(topk, p)
        return min_dists.topk(k, dim=1).values.mean(dim=1)
    if aggregation == "topk_reweighted":
        k = min(topk, p)
        topk_d, _ = min_dists.topk(k, dim=1)
        weights = 1.0 - torch.softmax(1.0 / topk_d.clamp(min=1e-6), dim=1)
        return (weights * topk_d).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
    if aggregation == "mean":
        return min_dists.mean(dim=1)
    if aggregation == "max":
        return min_dists.max(dim=1).values
    raise ValueError(f"Unsupported aggregation: {aggregation}")


@torch.no_grad()
def kcenter_greedy_select(features: torch.Tensor, k: int,
                          init: str = "mean",
                          candidate_pool_size: int | None = None) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected [N, D], got shape={tuple(features.shape)}")
    n = features.shape[0]
    if k >= n:
        return torch.arange(n, device=features.device)

    x = features
    if candidate_pool_size is not None and n > candidate_pool_size:
        step = max(n // candidate_pool_size, 1)
        base_idx = torch.arange(0, n, step, device=features.device)[:candidate_pool_size]
        x = x[base_idx]
    else:
        base_idx = None

    n_work = x.shape[0]
    if k >= n_work:
        selected = torch.arange(n_work, device=x.device)
        return base_idx[selected] if base_idx is not None else selected

    if init == "mean":
        center = x.mean(dim=0, keepdim=True)
        min_dists = torch.cdist(x, center).squeeze(1)
        first_idx = torch.argmax(min_dists)
    else:
        first_idx = torch.randint(0, n_work, (1,), device=x.device).squeeze(0)

    selected = [first_idx]
    min_dists = torch.cdist(x, x[first_idx:first_idx + 1]).squeeze(1)
    for _ in range(1, k):
        next_idx = torch.argmax(min_dists)
        selected.append(next_idx)
        new_dists = torch.cdist(x, x[next_idx:next_idx + 1]).squeeze(1)
        min_dists = torch.minimum(min_dists, new_dists)

    selected_idx = torch.stack(selected)
    if base_idx is not None:
        selected_idx = base_idx[selected_idx]
    return selected_idx
```

- [ ] **Step 1.4: Run tests — pass**

```powershell
python -m pytest tests/test_patchcore_common.py -v
```
Expected: all PASS. (If `test_first_pick_is_farthest_from_mean` fails, print `kcenter_greedy_select(x, 2)` and fix the test's expected indices to the actual deterministic ones — the property that matters is determinism + spread, and the implementation is frozen verbatim.)

- [ ] **Step 1.5: Delegate `PatchCoreInference._aggregate`**

In `models/patchcore_inference.py` add the import near the other model imports:
```python
from models.patchcore_common import aggregate_image_score
```
and replace the whole `_aggregate` body (lines 96-110) with:
```python
    @torch.no_grad()
    def _aggregate(self, min_dists: torch.Tensor) -> torch.Tensor:
        """min_dists: [B, P] → [B] image-level score."""
        return aggregate_image_score(min_dists, self.aggregation, self.topk)
```

- [ ] **Step 1.6: Regression guard — rerun smoke, scores unchanged**

```powershell
python scripts\smoke_predict.py cuda
```
Expected: **identical scores** to Step 0.6 (same 4 decimals) and `SMOKE OK`.

- [ ] **Step 1.7: Commit**

```powershell
git add models/patchcore_common.py models/patchcore_inference.py tests/test_patchcore_common.py
git commit -m "refactor: extract shared aggregate/kcenter into patchcore_common (verbatim from eval script)"
```

---

### Task 2: Variant registry + builder — `models/patchcore_variants.py` (TDD)

**Files:**
- Create: `models/patchcore_variants.py`
- Create: `tests/test_variants.py`

- [ ] **Step 2.1: Write the failing tests**

`tests/test_variants.py`:
```python
import json
from types import SimpleNamespace

import pytest
import torch

from models.patchcore_variants import (
    VARIANT_SPECS,
    available_variants,
    build_variant_model,
    get_coreset_indices,
    load_calibration,
    variant_stats,
)


def make_base(n=50, d=8):
    return SimpleNamespace(
        category="bottle",
        bank=torch.randn(n, d),
        aggregation="topk_reweighted",
        topk=9,
        threshold=1.0,
        train_score_mean=0.5,
        train_score_std=0.1,
    )


class TestRegistry:
    def test_specs(self):
        assert set(VARIANT_SPECS) == {"production", "patchcore_v2", "patchcore_v1"}
        assert VARIANT_SPECS["production"].kind == "production"
        v1 = VARIANT_SPECS["patchcore_v1"]
        assert (v1.aggregation, v1.topk, v1.coreset) == ("topk_mean", 3, 10000)
        v2 = VARIANT_SPECS["patchcore_v2"]
        assert (v2.aggregation, v2.topk, v2.coreset) == ("topk_reweighted", 9, 10000)


class TestCoresetCache:
    def test_builds_and_caches(self, tmp_path):
        bank = torch.randn(30, 4)
        idx1 = get_coreset_indices(tmp_path, "bottle", bank, k=5)
        cache = tmp_path / "bottle" / "variants" / "coreset5_idx.pt"
        assert cache.exists()
        # second call must read the cache, not recompute: corrupt-proof by passing a different bank
        idx2 = get_coreset_indices(tmp_path, "bottle", torch.randn(30, 4), k=5)
        assert torch.equal(idx1, idx2)
        assert len(idx1) == 5


class TestBuildVariant:
    def test_production_returns_base(self, tmp_path):
        base = make_base()
        out = build_variant_model(base, VARIANT_SPECS["production"], None, tmp_path)
        assert out is base

    def test_reconstructed_overrides(self, tmp_path):
        base = make_base(n=50)
        cal = {"threshold": 2.5, "score_mean": 1.1, "score_std": 0.2, "n_val": 30}
        m = build_variant_model(base, VARIANT_SPECS["patchcore_v1"], cal, tmp_path)
        assert m is not base
        assert m.aggregation == "topk_mean" and m.topk == 3
        assert m.threshold == 2.5 and m.train_score_mean == 1.1 and m.train_score_std == 0.2
        assert m.bank.shape[0] == 50  # k=10000 >= n=50 → full bank kept
        assert base.aggregation == "topk_reweighted"  # base untouched

    def test_missing_calibration_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not calibrated"):
            build_variant_model(make_base(), VARIANT_SPECS["patchcore_v1"], None, tmp_path)


class TestAvailability:
    def test_flags(self):
        cal = {"bottle": {"patchcore_v1": {"threshold": 1, "score_mean": 0, "score_std": 1, "n_val": 3}}}
        out = {v["id"]: v for v in available_variants("bottle", cal)}
        assert out["production"]["available"] is True
        assert out["patchcore_v1"]["available"] is True
        assert out["patchcore_v2"]["available"] is False
        assert out["patchcore_v1"]["approximate"] is False

    def test_screw_marked_approximate(self):
        out = {v["id"]: v for v in available_variants("screw", {})}
        assert out["patchcore_v1"]["approximate"] is True
        assert out["production"]["approximate"] is False


class TestCalibrationIO:
    def test_load_missing_returns_empty(self, tmp_path):
        assert load_calibration(tmp_path) == {}

    def test_load_roundtrip(self, tmp_path):
        data = {"bottle": {"patchcore_v1": {"threshold": 1.0, "score_mean": 0.5, "score_std": 0.1, "n_val": 31}}}
        (tmp_path / "variant_thresholds.json").write_text(json.dumps(data))
        assert load_calibration(tmp_path) == data

    def test_variant_stats(self):
        scores = [float(i) for i in range(1, 101)]  # 1..100
        s = variant_stats(scores)
        assert s["threshold"] == pytest.approx(99.01, abs=0.1)  # p99
        assert s["score_mean"] == pytest.approx(50.5)
        assert s["n_val"] == 100
```

- [ ] **Step 2.2: Run to verify failure**

```powershell
python -m pytest tests/test_variants.py -v
```
Expected: `ModuleNotFoundError: No module named 'models.patchcore_variants'`.

- [ ] **Step 2.3: Implement `models/patchcore_variants.py`**

```python
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
```

- [ ] **Step 2.4: Run tests — pass**

```powershell
python -m pytest tests/test_variants.py tests/test_patchcore_common.py -v
```
Expected: all PASS.

- [ ] **Step 2.5: Commit**

```powershell
git add models/patchcore_variants.py tests/test_variants.py
git commit -m "feat: variant registry + reconstruction of PatchCore v1/v2 from production banks"
```

---

### Task 3: Threshold calibration script

**Files:**
- Create: `scripts/calibrate_variant_thresholds.py`

- [ ] **Step 3.1: Check the dataset import + item format used by the export script**

```powershell
Select-String -Path scripts\export_patchcore_banks.py -Pattern "^from|^import" | Select-Object -First 15
Select-String -Path datasets\mvtec_dataset.py -Pattern "def __getitem__" -Context 0,15
```
Record: the exact `MVTecADDataset` import line and what `__getitem__` returns (tensor vs dict — adapt the marked line in Step 3.2).

- [ ] **Step 3.2: Write `scripts/calibrate_variant_thresholds.py`**

```python
"""Recalibrate anomaly thresholds for reconstructed PatchCore variants.

Method (identical to scripts/export_patchcore_banks.py): hold out the same 15%
val_normal split (seed 43), score each held-out normal image with the variant,
threshold = 99th percentile of those scores. The expensive backbone pass runs
once per batch per bank; variants share the coreset bank.

Usage:
    python scripts/calibrate_variant_thresholds.py --device cuda
    python scripts/calibrate_variant_thresholds.py --device cpu --categories bottle screw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.patchcore_inference import PatchCoreInference          # noqa: E402
from models.patchcore_variants import (                            # noqa: E402
    VARIANT_SPECS, get_coreset_indices, variant_stats,
)
from models.patchcore_common import aggregate_image_score          # noqa: E402
# Same dataset + transform stack as the export script:
from datasets.mvtec_dataset import MVTecADDataset                  # noqa: E402  (verify exact module in Step 3.1)
from utils.transforms import Compose, NormalizeTensor, ResizePadToSquare  # noqa: E402

SEED = 43
VAL_NORMAL_RATIO = 0.15
IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_ROOT = ROOT.parent.parent / "datasets" / "mvtec_ad"
PRODUCTION_DIR = ROOT / "production_models"

ALL_CATEGORIES = sorted(
    d.name for d in PRODUCTION_DIR.iterdir()
    if d.is_dir() and (d / "patchcore_bank.pt").exists()
)


def val_loader(category: str):
    ds = MVTecADDataset(
        root=str(DATASET_ROOT),
        category=category,
        split="val_normal",
        image_size=IMAGE_SIZE,
        image_transform=Compose([
            ResizePadToSquare(IMAGE_SIZE),
            NormalizeTensor(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        val_normal_ratio=VAL_NORMAL_RATIO,
        val_mixed_ratio=0.0,
        seed=SEED,
    )
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)


@torch.no_grad()
def patch_min_dists(model: PatchCoreInference, batch: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """batch: [B,3,H,W] already normalized → [B, P] min distance per patch."""
    outs = model.backbone(batch.to(model.device))
    fmap = model._get_feature_map(outs)
    b, c, h, w = fmap.shape
    fmap = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)
    patches = fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)
    patches = F.normalize(patches, p=2, dim=2, eps=1e-8)
    flat = patches.reshape(b * h * w, c)
    dists = torch.cdist(flat, bank)
    dists = torch.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
    return dists.reshape(b, h * w, bank.shape[0]).min(dim=2).values


@torch.no_grad()
def calibrate_category(category: str, device: str, variants: list[str]) -> dict:
    t0 = time.time()
    model = PatchCoreInference(category, PRODUCTION_DIR / category / "patchcore_bank.pt", device=device)
    coreset_idx = get_coreset_indices(PRODUCTION_DIR, category, model.bank)
    coreset_bank = model.bank[coreset_idx.to(model.bank.device)]

    scores: dict[str, list[float]] = {vid: [] for vid in variants}
    n_imgs = 0
    for batch in val_loader(category):
        x = batch["image"] if isinstance(batch, dict) else (batch[0] if isinstance(batch, (list, tuple)) else batch)  # ← adapt per Step 3.1
        n_imgs += x.shape[0]
        md = patch_min_dists(model, x, coreset_bank)  # both v1/v2 use the coreset bank
        for vid in variants:
            spec = VARIANT_SPECS[vid]
            s = aggregate_image_score(md, spec.aggregation, spec.topk)
            scores[vid].extend(float(v) for v in s)

    out = {vid: variant_stats(scores[vid]) for vid in variants}
    for vid, st in out.items():
        print(f"  [{category}] {vid}: n={st['n_val']} thr(p99)={st['threshold']:.4f} "
              f"mean={st['score_mean']:.4f} std={st['score_std']:.4f}")
    print(f"  [{category}] done in {time.time() - t0:.0f}s ({n_imgs} val images)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--categories", nargs="*", default=ALL_CATEGORIES)
    args = ap.parse_args()

    variants = [vid for vid, s in VARIANT_SPECS.items() if s.kind == "reconstructed"]
    out_path = PRODUCTION_DIR / "variant_thresholds.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}

    for cat in args.categories:
        print(f"[calibrate] {cat} on {args.device} …")
        existing[cat] = {**existing.get(cat, {}), **calibrate_category(cat, args.device, variants)}
        out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")  # save incrementally

    print(f"[calibrate] wrote {out_path}")


if __name__ == "__main__":
    main()
```
Note: both reconstructed variants share the coreset bank, so the distance matrix is computed once per batch and only the aggregation differs — this keeps calibration cheap (~30–60 val images per category, one backbone pass each).

- [ ] **Step 3.3: Run for one category, sanity-check**

```powershell
python scripts\calibrate_variant_thresholds.py --device cuda --categories bottle
```
Expected: two lines (patchcore_v2, patchcore_v1) with `n≈31`, thresholds the same order of magnitude as the production threshold from the Task 0 smoke test, `production_models/variant_thresholds.json` created with a `bottle` key, and `production_models/bottle/variants/coreset10000_idx.pt` written (the kcenter build is the slow part — minutes on CPU, seconds on GPU).

- [ ] **Step 3.4: Run all 15 categories**

```powershell
python scripts\calibrate_variant_thresholds.py --device cuda
```
Expected: 15 category blocks; JSON ends with 15 keys × 2 variants. GPU: a few minutes total; CPU: up to ~1 h (dominated by 15 kcenter builds — they cache, so this cost is one-time).

- [ ] **Step 3.5: Commit (JSON + coreset index caches so the server works out of the box)**

```powershell
git add scripts/calibrate_variant_thresholds.py production_models/variant_thresholds.json
git add -f production_models/*/variants/coreset10000_idx.pt
git commit -m "feat: offline threshold calibration for reconstructed variants (p99 val_normal, seed 43)"
```
(`git add -f` in case a `*.pt` ignore rule exists; the index files are ~80 KB each. If they exceed ~1 MB each, commit only the JSON and let coresets rebuild lazily.)

---

### Task 4: Arena sampler + metrics — `webapp/` package (TDD)

**Files:**
- Create: `webapp/__init__.py` (empty)
- Create: `webapp/sampler.py`
- Create: `webapp/metrics.py`
- Create: `tests/test_webapp_sampler.py`
- Create: `tests/test_webapp_metrics.py`

- [ ] **Step 4.1: Write failing sampler tests**

`tests/test_webapp_sampler.py`:
```python
import pytest

from webapp.sampler import TestImage, list_test_images, sample_test_images


@pytest.fixture
def dataset(tmp_path):
    """mvtec-like tree: bottle/test/{good×10, crack×4, hole×2}."""
    for defect, n in [("good", 10), ("crack", 4), ("hole", 2)]:
        d = tmp_path / "bottle" / "test" / defect
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i:03d}.png").write_bytes(b"\x89PNG fake")
    return tmp_path


class TestList:
    def test_groups(self, dataset):
        groups = list_test_images(dataset, "bottle")
        assert {g: len(v) for g, v in groups.items()} == {"good": 10, "crack": 4, "hole": 2}
        assert groups["crack"][0].is_anomaly is True
        assert groups["good"][0].is_anomaly is False

    def test_missing_category_raises(self, dataset):
        with pytest.raises(FileNotFoundError):
            list_test_images(dataset, "nope")


class TestSample:
    def test_deterministic(self, dataset):
        a = sample_test_images(dataset, "bottle", 8, seed=7)
        b = sample_test_images(dataset, "bottle", 8, seed=7)
        assert a == b

    def test_seed_changes_sample(self, dataset):
        a = sample_test_images(dataset, "bottle", 8, seed=1)
        b = sample_test_images(dataset, "bottle", 8, seed=2)
        assert a != b

    def test_stratified_proportional(self, dataset):
        s = sample_test_images(dataset, "bottle", 8, seed=0)
        by = {}
        for img in s:
            by[img.defect] = by.get(img.defect, 0) + 1
        assert len(s) == 8
        assert by["good"] == 5 and by["crack"] == 2 and by["hole"] == 1  # 8×(10,4,2)/16

    def test_at_least_one_per_type(self, dataset):
        s = sample_test_images(dataset, "bottle", 3, seed=0)
        assert {img.defect for img in s} == {"good", "crack", "hole"}

    def test_n_capped_to_total(self, dataset):
        s = sample_test_images(dataset, "bottle", 999, seed=0)
        assert len(s) == 16

    def test_no_duplicates(self, dataset):
        s = sample_test_images(dataset, "bottle", 16, seed=3)
        assert len({(i.defect, i.filename) for i in s}) == 16
```

- [ ] **Step 4.2: Write failing metrics tests**

`tests/test_webapp_metrics.py`:
```python
import pytest

from webapp.metrics import auroc, summarize


class TestAuroc:
    def test_perfect(self):
        assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_inverted(self):
        assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_known_value(self):
        # pos scores {3,1}, neg {2,0}: pairs won 3>2,3>0,1>0 = 3/4
        assert auroc([1, 0, 1, 0], [3.0, 2.0, 1.0, 0.0]) == pytest.approx(0.75)

    def test_ties_average(self):
        assert auroc([1, 0], [1.0, 1.0]) == pytest.approx(0.5)

    def test_single_class_none(self):
        assert auroc([1, 1], [0.5, 0.6]) is None


def res(gt, pred, score, ms=10.0, verdict=None):
    return {
        "ground_truth_anomaly": gt, "is_anomaly": pred, "anomaly_score": score,
        "inference_ms": ms, "verdict": verdict or ("tp" if gt and pred else "tn"),
    }


class TestSummarize:
    def test_counts_and_accuracy(self):
        results = [res(True, True, 0.9), res(False, False, 0.1),
                   res(False, True, 0.8), res(True, False, 0.2)]
        s = summarize(results)
        assert s["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
        assert s["accuracy"] == pytest.approx(0.5)
        assert s["precision"] == pytest.approx(0.5)
        assert s["recall"] == pytest.approx(0.5)
        assert s["f1"] == pytest.approx(0.5)
        assert s["n"] == 4 and s["errors"] == 0

    def test_errors_excluded(self):
        results = [res(True, True, 0.9), {"verdict": "error", "filename": "x.png"}]
        s = summarize(results)
        assert s["n"] == 1 and s["errors"] == 1

    def test_timing(self):
        results = [res(True, True, 0.9, ms=10), res(False, False, 0.1, ms=30)]
        s = summarize(results)
        assert s["mean_ms"] == pytest.approx(20.0)
        assert s["p95_ms"] == 30.0
```

- [ ] **Step 4.3: Run to verify failure**

```powershell
python -m pytest tests/test_webapp_sampler.py tests/test_webapp_metrics.py -v
```
Expected: `ModuleNotFoundError: No module named 'webapp'`.

- [ ] **Step 4.4: Implement `webapp/sampler.py` (+ empty `webapp/__init__.py`)**

```python
"""Deterministic stratified sampling of MVTec test images for arena runs."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@dataclass(frozen=True, order=True)
class TestImage:
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
```

- [ ] **Step 4.5: Implement `webapp/metrics.py`**

```python
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
    return {
        "n": n,
        "errors": len(results) - n,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc(labels, [r["anomaly_score"] for r in ok]),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "mean_ms": sum(times) / n if n else None,
        "p95_ms": times[min(int(round(0.95 * n)), n - 1)] if n else None,
    }
```

- [ ] **Step 4.6: Run tests — pass**

```powershell
python -m pytest tests/test_webapp_sampler.py tests/test_webapp_metrics.py -v
```
Expected: all PASS. (If `test_timing` p95 differs, align the test to the implementation's index — the exact p95 convention is not load-bearing; `mean_ms` is.)

- [ ] **Step 4.7: Commit**

```powershell
git add webapp/ tests/test_webapp_sampler.py tests/test_webapp_metrics.py
git commit -m "feat: arena stratified sampler + self-contained metrics"
```

---

### Task 5: Job manager — `webapp/jobs.py` (TDD)

**Files:**
- Create: `webapp/jobs.py`
- Create: `tests/test_webapp_jobs.py`

- [ ] **Step 5.1: Write failing tests**

`tests/test_webapp_jobs.py`:
```python
import threading
import time

import pytest

from webapp.jobs import ArenaJob, JobBusyError, JobManager
from webapp.sampler import TestImage


def fake_runner_factory(scores, delay=0.0, barrier=None):
    """Returns runner(job) that emits one canned result per image."""
    def runner(job: ArenaJob):
        from webapp.metrics import summarize, verdict_of
        for i, img in enumerate(job.images):
            if barrier is not None and i == 1:
                barrier.wait(timeout=5)
            if job.cancel_requested:
                job.finish("cancelled", summary=summarize(job.results))
                return
            time.sleep(delay)
            pred = scores[i] >= 0.5
            job.add_result({
                "idx": i, "defect_type": img.defect, "filename": img.filename,
                "ground_truth_anomaly": img.is_anomaly, "anomaly_score": scores[i],
                "is_anomaly": pred, "inference_ms": 1.0,
                "verdict": verdict_of(img.is_anomaly, pred),
                "correct": img.is_anomaly == pred,
            })
        job.finish("done", summary=summarize(job.results))
    return runner


IMAGES = [TestImage("good", "a.png"), TestImage("good", "b.png"),
          TestImage("crack", "c.png"), TestImage("crack", "d.png")]


def wait_status(job, status, timeout=5.0):
    t0 = time.time()
    while job.status != status and time.time() - t0 < timeout:
        time.sleep(0.01)
    assert job.status == status


class TestJobLifecycle:
    def test_runs_to_done_with_summary(self):
        mgr = JobManager()
        job = mgr.start("bottle", "production", IMAGES,
                        fake_runner_factory([0.1, 0.2, 0.9, 0.8]))
        wait_status(job, "done")
        assert len(job.results) == 4
        assert job.summary["accuracy"] == pytest.approx(1.0)
        assert job.summary["confusion"] == {"tp": 2, "tn": 2, "fp": 0, "fn": 0}

    def test_single_flight(self):
        mgr = JobManager()
        barrier = threading.Barrier(2)
        job = mgr.start("bottle", "production", IMAGES,
                        fake_runner_factory([0.1, 0.2, 0.9, 0.8], barrier=barrier))
        with pytest.raises(JobBusyError):
            mgr.start("bottle", "production", IMAGES, fake_runner_factory([0.5] * 4))
        barrier.wait(timeout=5)
        wait_status(job, "done")
        job2 = mgr.start("bottle", "production", IMAGES, fake_runner_factory([0.1, 0.2, 0.9, 0.8]))
        wait_status(job2, "done")

    def test_cancel(self):
        mgr = JobManager()
        barrier = threading.Barrier(2)
        job = mgr.start("bottle", "production", IMAGES,
                        fake_runner_factory([0.1, 0.2, 0.9, 0.8], barrier=barrier))
        mgr.cancel(job.id)
        barrier.wait(timeout=5)
        wait_status(job, "cancelled")
        assert len(job.results) < 4
        assert job.summary is not None  # partial summary

    def test_get_unknown_returns_none(self):
        assert JobManager().get("nope") is None


class TestEventCursor:
    def test_wait_for_results_since(self):
        mgr = JobManager()
        job = mgr.start("bottle", "production", IMAGES,
                        fake_runner_factory([0.1, 0.2, 0.9, 0.8], delay=0.02))
        got = []
        cursor = 0
        while True:
            batch, status, _summary = job.wait_results(cursor, timeout=2.0)
            got.extend(batch)
            cursor += len(batch)
            if status != "running":
                break
        assert [r["idx"] for r in got] == [0, 1, 2, 3]
```

- [ ] **Step 5.2: Run to verify failure**

```powershell
python -m pytest tests/test_webapp_jobs.py -v
```
Expected: `ImportError` (no `webapp.jobs`).

- [ ] **Step 5.3: Implement `webapp/jobs.py`**

```python
"""Single-flight background job manager for arena batch runs."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


class JobBusyError(RuntimeError):
    def __init__(self, current_id: str):
        super().__init__(f"A job is already running: {current_id}")
        self.current_id = current_id


@dataclass
class ArenaJob:
    id: str
    category: str
    variant: str
    images: list
    seed: int = 0
    status: str = "running"        # running | done | cancelled | error
    error: str | None = None
    results: list = field(default_factory=list)
    summary: dict | None = None
    cancel_requested: bool = False
    cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def add_result(self, r: dict) -> None:
        with self.cond:
            self.results.append(r)
            self.cond.notify_all()

    def finish(self, status: str, summary: dict | None = None, error: str | None = None) -> None:
        with self.cond:
            self.status = status
            self.summary = summary
            self.error = error
            self.cond.notify_all()

    def wait_results(self, cursor: int, timeout: float = 15.0):
        """Block until there are results past cursor or the job leaves 'running'.
        Returns (new_results, status, summary)."""
        with self.cond:
            if cursor >= len(self.results) and self.status == "running" and timeout > 0:
                self.cond.wait(timeout=timeout)
            return list(self.results[cursor:]), self.status, self.summary


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ArenaJob] = {}
        self._current: ArenaJob | None = None

    def start(self, category: str, variant: str, images: list, runner, seed: int = 0) -> ArenaJob:
        with self._lock:
            if self._current is not None and self._current.status == "running":
                raise JobBusyError(self._current.id)
            job = ArenaJob(uuid.uuid4().hex[:12], category, variant, images, seed=seed)
            self._jobs[job.id] = job
            self._current = job
        threading.Thread(target=runner, args=(job,), daemon=True,
                         name=f"arena-{job.id}").start()
        return job

    def get(self, job_id: str) -> ArenaJob | None:
        return self._jobs.get(job_id)

    @property
    def current(self) -> ArenaJob | None:
        return self._current

    def cancel(self, job_id: str) -> ArenaJob | None:
        job = self._jobs.get(job_id)
        if job is not None and job.status == "running":
            job.cancel_requested = True
        return job
```

- [ ] **Step 5.4: Run tests — pass**

```powershell
python -m pytest tests/test_webapp_jobs.py -v
```
Expected: all PASS (every wait in tests has a timeout — no hangs).

- [ ] **Step 5.5: Commit**

```powershell
git add webapp/jobs.py tests/test_webapp_jobs.py
git commit -m "feat: single-flight arena job manager with condition-based streaming"
```

---

### Task 6: Thumbnails — `webapp/thumbs.py` (TDD)

**Files:**
- Create: `webapp/thumbs.py`
- Create: `tests/test_webapp_thumbs.py`

- [ ] **Step 6.1: Write failing tests**

`tests/test_webapp_thumbs.py`:
```python
import pytest
from PIL import Image

from webapp.thumbs import get_thumb, safe_name


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "ds" / "bottle" / "test" / "good"
    d.mkdir(parents=True)
    Image.new("RGB", (700, 700), (200, 30, 30)).save(d / "000.png")
    return tmp_path / "ds"


class TestSafeName:
    def test_accepts_normal(self):
        assert safe_name("bottle") and safe_name("000.png") and safe_name("metal_nut")

    @pytest.mark.parametrize("bad", ["..", "a/b", "a\\b", "", ".hidden/../x"])
    def test_rejects_traversal(self, bad):
        assert not safe_name(bad)


class TestThumb:
    def test_creates_and_caches(self, dataset, tmp_path):
        cache = tmp_path / "cache"
        p1 = get_thumb(dataset, cache, "bottle", "good", "000.png", 128)
        assert p1.exists() and p1.suffix == ".jpg"
        with Image.open(p1) as im:
            assert max(im.size) == 128
        mtime = p1.stat().st_mtime_ns
        p2 = get_thumb(dataset, cache, "bottle", "good", "000.png", 128)
        assert p2 == p1 and p2.stat().st_mtime_ns == mtime  # cache hit, not rewritten

    def test_missing_image_raises(self, dataset, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_thumb(dataset, tmp_path / "c", "bottle", "good", "nope.png", 128)

    def test_bad_size_raises(self, dataset, tmp_path):
        with pytest.raises(ValueError):
            get_thumb(dataset, tmp_path / "c", "bottle", "good", "000.png", 999)
```

- [ ] **Step 6.2: Run to verify failure**

```powershell
python -m pytest tests/test_webapp_thumbs.py -v
```
Expected: `ImportError`.

- [ ] **Step 6.3: Implement `webapp/thumbs.py`**

```python
"""Server-side thumbnail generation with a flat disk cache."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ALLOWED_SIZES = {64, 128, 256}


def safe_name(value: str) -> bool:
    """Single path component: no separators, no traversal, non-empty."""
    return bool(value) and ".." not in value and "/" not in value and "\\" not in value


def get_thumb(dataset_root: Path, cache_dir: Path, cat: str, defect: str,
              filename: str, size: int = 128) -> Path:
    if size not in ALLOWED_SIZES:
        raise ValueError(f"size must be one of {sorted(ALLOWED_SIZES)}")
    for part in (cat, defect, filename):
        if not safe_name(part):
            raise ValueError(f"Invalid path component: {part!r}")
    src = Path(dataset_root) / cat / "test" / defect / filename
    if not src.is_file():
        raise FileNotFoundError(str(src))

    out = Path(cache_dir) / f"{cat}__{defect}__{Path(filename).stem}__{size}.jpg"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(out, "JPEG", quality=85)
    return out
```

- [ ] **Step 6.4: Run tests — pass**

```powershell
python -m pytest tests/test_webapp_thumbs.py -v
```
Expected: all PASS.

- [ ] **Step 6.5: Commit**

```powershell
git add webapp/thumbs.py tests/test_webapp_thumbs.py
git commit -m "feat: cached dataset thumbnails with path-component validation"
```

---

### Task 7: Server endpoints (modify `server.py`) + integration tests

**Files:**
- Modify: `server.py` (imports, variant cache, meta, thumb/mask, predict variants, arena routes, device auto, static mount)
- Create: `tests/test_server_api.py`

- [ ] **Step 7.1: Write failing integration tests**

`tests/test_server_api.py` — TestClient with monkeypatched dataset root + fake variant model (no real banks/weights needed):
```python
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server


class FakeModel:
    """Deterministic stand-in for PatchCoreInference."""
    def __init__(self):
        self.threshold = 0.5
        self.category = "bottle"

    def predict(self, pil_image):
        return {
            "anomaly_score": 0.9, "anomaly_probability": 0.88, "is_anomalous": True,
            "threshold": self.threshold, "category": self.category,
            "inference_time_ms": 1.0, "score_components": {},
            "heatmap": np.zeros((8, 8)), "reconstruction": None,
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    ds = tmp_path / "mvtec"
    for defect, n in [("good", 6), ("broken", 4)]:
        d = ds / "bottle" / "test" / defect
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (64, 64), (i * 20 % 255, 80, 80)).save(d / f"{i:03d}.png")
    gt = ds / "bottle" / "ground_truth" / "broken"
    gt.mkdir(parents=True)
    Image.new("L", (64, 64), 255).save(gt / "000_mask.png")

    monkeypatch.setattr(server, "DATASET_ROOT", ds)
    monkeypatch.setattr(server, "CATEGORIES", ["bottle"])
    monkeypatch.setattr(server, "THUMB_CACHE_DIR", tmp_path / "thumbs")
    fake = FakeModel()
    monkeypatch.setattr(server, "get_variant_model",
                        lambda cat, variant="production": fake if variant in ("production", "patchcore_v1")
                        else (_ for _ in ()).throw(server.HTTPException(status_code=400, detail="unknown")))
    monkeypatch.setattr(server, "load_calibration", lambda _dir: {"bottle": {
        "patchcore_v1": {"threshold": 1, "score_mean": 0, "score_std": 1, "n_val": 5}}})
    server.meta_cache_clear()
    return TestClient(server.app)


class TestMeta:
    def test_meta_lists_variants_and_counts(self, client):
        r = client.get("/api/meta")
        assert r.status_code == 200
        cat = r.json()["categories"][0]
        assert cat["name"] == "bottle"
        assert cat["test_total"] == 10
        variants = {v["id"]: v for v in cat["variants"]}
        assert variants["production"]["available"] is True
        assert variants["patchcore_v1"]["available"] is True
        assert variants["patchcore_v2"]["available"] is False


class TestThumbAndMask:
    def test_thumb_ok(self, client):
        r = client.get("/api/dataset/thumb", params=dict(cat="bottle", defect="good", filename="000.png", size=64))
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    def test_thumb_traversal_rejected(self, client):
        r = client.get("/api/dataset/thumb", params=dict(cat="..", defect="good", filename="000.png"))
        assert r.status_code == 400

    def test_mask_ok_and_missing(self, client):
        ok = client.get("/api/dataset/mask", params=dict(cat="bottle", defect="broken", filename="000.png"))
        assert ok.status_code == 200
        missing = client.get("/api/dataset/mask", params=dict(cat="bottle", defect="broken", filename="001.png"))
        assert missing.status_code == 404


class TestArenaFlow:
    def start(self, client, n=6, variant="production", seed=42):
        r = client.post("/api/arena/start",
                        json=dict(category="bottle", variant=variant, n_images=n, seed=seed))
        assert r.status_code == 200, r.text
        return r.json()

    def wait_done(self, client, job_id, timeout=10):
        deadline = time.time() + timeout
        results, payload = [], None
        while time.time() < deadline:
            r = client.get(f"/api/arena/jobs/{job_id}", params={"since": len(results)})
            payload = r.json()
            results += payload["results"]
            if payload["status"] != "running":
                return results, payload
            time.sleep(0.02)
        raise AssertionError("job did not finish in time")

    def test_start_poll_complete(self, client):
        body = self.start(client)
        assert body["n"] == 6 and body["seed"] == 42 and len(body["images"]) == 6
        results, payload = self.wait_done(client, body["job_id"])
        assert payload["status"] == "done"
        assert len(results) == 6
        assert payload["summary"]["n"] == 6

    def test_same_seed_same_images(self, client):
        a = self.start(client)
        self.wait_done(client, a["job_id"])
        b = self.start(client)
        self.wait_done(client, b["job_id"])
        assert a["images"] == b["images"]

    def test_unknown_variant_400(self, client):
        r = client.post("/api/arena/start", json=dict(category="bottle", variant="nope", n_images=5))
        assert r.status_code == 400

    def test_busy_409(self, client, monkeypatch):
        import webapp.jobs as jobs_mod
        slow = self.start(client, n=6)
        r = client.post("/api/arena/start", json=dict(category="bottle", variant="production", n_images=5))
        # the fake model is instant; both 409 (if still running) and 200 are acceptable here —
        # the strict single-flight behavior is covered in tests/test_webapp_jobs.py
        assert r.status_code in (200, 409)
        self.wait_done(client, slow["job_id"])

    def test_sse_stream_smoke(self, client):
        job_id = self.start(client)["job_id"]
        events = []
        with client.stream("GET", f"/api/arena/jobs/{job_id}/stream") as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                if events and events[-1] == "summary":
                    break
        assert events.count("result") == 6
        assert events[-1] == "summary"


class TestPredictVariant:
    def test_predict_from_dataset_accepts_variant(self, client):
        r = client.post("/api/predict/from-dataset",
                        data=dict(category="bottle", defect="good", filename="000.png",
                                  model_variant="patchcore_v1"))
        assert r.status_code == 200
        body = r.json()
        assert body["model_variant"] == "patchcore_v1"
        assert "heatmap_base64" in body
```

- [ ] **Step 7.2: Run to verify failure**

```powershell
python -m pytest tests/test_server_api.py -v
```
Expected: errors — `server` has no `THUMB_CACHE_DIR`, `get_variant_model`, `meta_cache_clear`; arena routes 404. (Importing `server` imports torch; weights are not loaded at import.)

- [ ] **Step 7.3: Modify `server.py`**

**(a) Imports + globals** — after the existing `from models.patchcore_inference import ...` block:
```python
import json
import random as _random
from functools import lru_cache

from models.patchcore_variants import (
    VARIANT_SPECS, available_variants, build_variant_model, load_calibration,
)
from webapp.jobs import JobBusyError, JobManager
from webapp.metrics import summarize, verdict_of
from webapp.sampler import list_test_images, sample_test_images
from webapp.thumbs import get_thumb, safe_name

THUMB_CACHE_DIR = PROJECT_ROOT / ".thumb_cache"
FRONTEND_DIST = PROJECT_ROOT.parent / "frontend" / "dist"
job_manager = JobManager()
_variant_cache: dict[tuple[str, str], object] = {}
```
And with the other FastAPI imports: `from fastapi.responses import StreamingResponse` , `from fastapi.staticfiles import StaticFiles` , `from pydantic import BaseModel, Field`.

**(b) Variant model accessor** — below `get_model`:
```python
def get_variant_model(category: str, variant: str = "production"):
    if variant not in VARIANT_SPECS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown variant '{variant}'. Available: {list(VARIANT_SPECS)}")
    key = (category, variant)
    if key not in _variant_cache:
        base = get_model(category)
        spec = VARIANT_SPECS[variant]
        cal = load_calibration(PRODUCTION_MODELS_DIR).get(category, {}).get(variant)
        try:
            _variant_cache[key] = build_variant_model(base, spec, cal, PRODUCTION_MODELS_DIR)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return _variant_cache[key]
```

**(c) `/api/meta`** (cached; expose a named cache-clear for tests):
```python
@lru_cache(maxsize=1)
def _meta_payload() -> str:
    calibration = load_calibration(PRODUCTION_MODELS_DIR)
    cats = []
    for c in CATEGORIES:
        entry = {"name": c, "variants": available_variants(c, calibration)}
        try:
            groups = list_test_images(DATASET_ROOT, c)
            entry["test_total"] = sum(len(v) for v in groups.values())
            entry["defect_types"] = [
                {"name": g, "count": len(v), "is_anomaly": g != "good"}
                for g, v in groups.items()
            ]
        except FileNotFoundError:
            entry["test_total"] = 0
            entry["defect_types"] = []
        cats.append(entry)
    return json.dumps({
        "categories": cats,
        "device": _device,
        "dataset_available": DATASET_ROOT.exists(),
    })


def meta_cache_clear() -> None:
    _meta_payload.cache_clear()


@app.get("/api/meta")
def meta_endpoint():
    return JSONResponse(content=json.loads(_meta_payload()))
```

**(d) Thumb + mask endpoints:**
```python
@app.get("/api/dataset/thumb")
def dataset_thumb(cat: str, defect: str, filename: str, size: int = 128):
    try:
        path = get_thumb(DATASET_ROOT, THUMB_CACHE_DIR, cat, defect, filename, size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/dataset/mask")
def dataset_mask(cat: str, defect: str, filename: str):
    if not all(safe_name(v) for v in (cat, defect, filename)):
        raise HTTPException(status_code=400, detail="Invalid path component")
    stem = Path(filename).stem
    mask = DATASET_ROOT / cat / "ground_truth" / defect / f"{stem}_mask.png"
    if not mask.is_file():
        raise HTTPException(status_code=404, detail="No ground-truth mask")
    return FileResponse(str(mask), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
```

**(e) `model_variant` on both predict endpoints** — add `model_variant: str = Form("production")` to the signatures of `predict` and `predict_from_dataset`, replace `model = get_model(category)` with `model = get_variant_model(category, model_variant)`, and add `"model_variant": model_variant` to both JSON responses.

**(f) Arena endpoints:**
```python
class ArenaStartRequest(BaseModel):
    category: str
    variant: str = "production"
    n_images: int = Field(default=100, ge=5, le=150)
    seed: int | None = None


def _make_runner(model, category: str):
    def runner(job):
        try:
            for i, img in enumerate(job.images):
                if job.cancel_requested:
                    job.finish("cancelled", summary=summarize(job.results))
                    return
                path = DATASET_ROOT / category / "test" / img.defect / img.filename
                try:
                    pil = Image.open(path).convert("RGB")
                    r = model.predict(pil)
                    res = {
                        "idx": i, "defect_type": img.defect, "filename": img.filename,
                        "ground_truth_anomaly": img.is_anomaly,
                        "anomaly_score": r["anomaly_score"],
                        "anomaly_probability": r["anomaly_probability"],
                        "is_anomaly": r["is_anomalous"], "threshold": r["threshold"],
                        "inference_ms": r["inference_time_ms"],
                        "verdict": verdict_of(img.is_anomaly, r["is_anomalous"]),
                        "correct": img.is_anomaly == r["is_anomalous"],
                    }
                except Exception as e:  # noqa: BLE001 — keep the batch alive
                    res = {"idx": i, "defect_type": img.defect, "filename": img.filename,
                           "ground_truth_anomaly": img.is_anomaly,
                           "verdict": "error", "error": str(e)}
                job.add_result(res)
            job.finish("done", summary=summarize(job.results))
        except Exception as e:  # noqa: BLE001
            job.finish("error", error=str(e))
    return runner


@app.post("/api/arena/start")
def arena_start(payload: ArenaStartRequest):
    if payload.category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category '{payload.category}'")
    model = get_variant_model(payload.category, payload.variant)  # 400 if unknown/uncalibrated
    seed = payload.seed if payload.seed is not None else _random.randint(0, 999_999)
    try:
        images = sample_test_images(DATASET_ROOT, payload.category, payload.n_images, seed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        job = job_manager.start(payload.category, payload.variant, images,
                                _make_runner(model, payload.category), seed=seed)
    except JobBusyError as e:
        raise HTTPException(status_code=409, detail={"message": str(e), "job_id": e.current_id})
    return {
        "job_id": job.id, "seed": seed, "n": len(images),
        "category": payload.category, "variant": payload.variant,
        "images": [{"idx": i, "defect_type": im.defect, "filename": im.filename,
                    "ground_truth_anomaly": im.is_anomaly} for i, im in enumerate(images)],
    }


@app.get("/api/arena/jobs/{job_id}")
def arena_poll(job_id: str, since: int = 0):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    results, status, summary = job.wait_results(since, timeout=0.0)
    return {"job_id": job.id, "status": status, "results": results,
            "summary": summary, "error": job.error,
            "total": len(job.images), "done": since + len(results)}


@app.get("/api/arena/jobs/{job_id}/stream")
def arena_stream(job_id: str, since: int = 0):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def gen():
        cursor = since
        while True:
            batch, status, summary = job.wait_results(cursor, timeout=15.0)
            for r in batch:
                yield f"event: result\ndata: {json.dumps(r)}\n\n"
            cursor += len(batch)
            if status != "running":
                yield ("event: summary\ndata: "
                       + json.dumps({"status": status, "summary": summary, "error": job.error})
                       + "\n\n")
                return
            if not batch:
                yield ": heartbeat\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/arena/jobs/{job_id}/cancel")
def arena_cancel(job_id: str):
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status, "cancel_requested": job.cancel_requested}
```

**(g) Device auto + static mount + SPA fallback** — in `__main__` change `--device` to `default="auto", choices=["auto", "cpu", "cuda"]` and resolve:
```python
    if args.device == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = args.device
```
After all route definitions (bottom of module, before `__main__`):
```python
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


@app.exception_handler(404)
async def spa_fallback(request, exc):
    if (request.method == "GET" and not request.url.path.startswith("/api")
            and FRONTEND_DIST.is_dir()):
        return FileResponse(FRONTEND_DIST / "index.html")
    return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
```

- [ ] **Step 7.4: Run all tests — pass**

```powershell
python -m pytest tests/test_server_api.py -v
python -m pytest -q
```
Expected: new file all PASS; full suite green.

- [ ] **Step 7.5: Manual smoke against the real server**

```powershell
python server.py --port 8000
# in a second shell:
curl.exe http://localhost:8000/api/meta
curl.exe -X POST http://localhost:8000/api/arena/start -H "Content-Type: application/json" -d "{\"category\":\"bottle\",\"variant\":\"production\",\"n_images\":10,\"seed\":1}"
curl.exe -N "http://localhost:8000/api/arena/jobs/<job_id_from_previous>/stream"
```
Expected: meta lists 15 categories × 3 variants (v1/v2 available after Task 3); start returns 10 images; the stream prints 10 `event: result` lines then `event: summary` with accuracy ≈ 0.9–1.0 for bottle/production. Record wall time (GPU ~5–15 s, CPU ~1–3 min). Repeat once with `"variant":"patchcore_v1"` — accuracy may be slightly lower (that's the story!).

- [ ] **Step 7.6: Commit**

```powershell
git add server.py tests/test_server_api.py
git commit -m "feat: arena endpoints (SSE+poll), variants on predict, meta, thumbs, masks, SPA mount"
```

---

### Task 8: Benchmark data regeneration — `scripts/build_webapp_data.py`

**Files:**
- Create: `scripts/build_webapp_data.py`
- Regenerates: `../frontend/src/data/benchmarks.json`, `../frontend/src/data/insights.json`

**Source → model mapping (headers verified):**
| benchmarks key | source | columns |
|---|---|---|
| `ocgan_v1` | `final_per_category_multiseed_aggregated.csv` | `category,num_seeds,mean_test_auroc,std_test_auroc,mean_test_auprc,…,mean_test_fpr_at_95_tpr,…` |
| `ocgan_v3` | located in Step 8.1 (else passthrough existing) | — |
| `patchcore_v1` | `logs/patchcore_pure.csv` | `category,seed,backbone,aggregation,topk,coreset,auroc,auprc,best_f1,fpr95,elapsed_s` |
| `patchcore_v2` | `logs/patchcore_v2.csv` | same + `feature_level` |
| `patchcore_v3` | `logs/patchcore_v3.csv` | same as v2 |
| `patchcore_p1` | `logs/patchcore_p1.csv` (+`_ext`) | same as v2 (partial categories OK) |
| `production_final` | v3 means; `screw` ← p1 rows with `feature_level=layer1+layer2+layer3` | macro must ≈ 0.9846 |

- [ ] **Step 8.1: Locate the ocgan_v3 source and record the draft JSON shape**

```powershell
Get-ChildItem logs\v3_per_category_multiseed -ErrorAction SilentlyContinue | Select-Object -First 10 Name
python -c "import json; d=json.load(open('../frontend/src/data/benchmarks.json')); print(list(d)); print(list(d['per_category'])); print(d['macro'])"
```
Record: any v3 aggregated CSV (else passthrough) and the draft's exact top-level keys — the script must emit the same shape (`per_category` = model → list of row objects, `macro` = model → float).

- [ ] **Step 8.2: Write `scripts/build_webapp_data.py`**

```python
"""Regenerate frontend benchmark + insight JSON from the repo's result CSVs.

Run:  python scripts/build_webapp_data.py
Writes: ../../frontend/src/data/benchmarks.json and insights.json (relative to repo root's parent)
The frontend treats these files as the single source of truth for metrics.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DATA = ROOT.parent / "frontend" / "src" / "data"

EXPECTED_MACRO = {  # from README / project history, tolerance ±0.005
    "patchcore_v1": 0.9051, "patchcore_v2": 0.9397, "patchcore_v3": 0.9828,
    "production_final": 0.9846,
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_runs_to_per_category(rows: list[dict]) -> dict[str, dict]:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        aurocs = [float(r["auroc"]) for r in rs]
        out[cat] = {
            "category": cat,
            "auroc": round(mean(aurocs), 4),
            "auroc_std": round(stdev(aurocs), 4) if len(aurocs) > 1 else 0.0,
            "auprc": round(mean(float(r["auprc"]) for r in rs), 4),
            "best_f1": round(mean(float(r["best_f1"]) for r in rs), 4),
            "fpr95": round(mean(float(r["fpr95"]) for r in rs), 4),
            "elapsed_s": round(mean(float(r["elapsed_s"]) for r in rs), 1) if rs[0].get("elapsed_s") else None,
            "n_seeds": len(rs),
            "feature_level": rs[0].get("feature_level", "layer2+layer3"),
            "aggregation": rs[0].get("aggregation"),
            "topk": int(rs[0]["topk"]) if rs[0].get("topk") else None,
            "coreset": int(rs[0]["coreset"]) if rs[0].get("coreset") else None,
        }
    return out


def ocgan_aggregated_to_per_category(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in sorted(rows, key=lambda r: r["category"]):
        out[r["category"]] = {
            "category": r["category"],
            "auroc": round(float(r["mean_test_auroc"]), 4),
            "auroc_std": round(float(r["std_test_auroc"]), 4),
            "auprc": round(float(r["mean_test_auprc"]), 4),
            "best_f1": round(float(r["mean_test_best_f1"]), 4),
            "fpr95": round(float(r["mean_test_fpr_at_95_tpr"]), 4),
            "elapsed_s": None,
            "n_seeds": int(r["num_seeds"]),
            "feature_level": None, "aggregation": None, "topk": None, "coreset": None,
        }
    return out


def macro_of(per_cat: dict[str, dict]) -> float:
    return round(mean(v["auroc"] for v in per_cat.values()), 4)


def main() -> None:
    existing = json.loads((FRONTEND_DATA / "benchmarks.json").read_text(encoding="utf-8"))

    per_category: dict[str, dict] = {}
    per_category["ocgan_v1"] = ocgan_aggregated_to_per_category(
        read_csv(ROOT / "final_per_category_multiseed_aggregated.csv"))

    V3_SRC: Path | None = None  # ← set to the CSV found in Step 8.1, or leave None
    if V3_SRC is not None:
        per_category["ocgan_v3"] = ocgan_aggregated_to_per_category(read_csv(V3_SRC))
    else:
        per_category["ocgan_v3"] = {r["category"]: r for r in existing["per_category"]["ocgan_v3"]}
        print("[build] ocgan_v3: no source CSV — passthrough of existing draft values")

    per_category["patchcore_v1"] = seed_runs_to_per_category(read_csv(ROOT / "logs" / "patchcore_pure.csv"))
    per_category["patchcore_v2"] = seed_runs_to_per_category(read_csv(ROOT / "logs" / "patchcore_v2.csv"))
    per_category["patchcore_v3"] = seed_runs_to_per_category(read_csv(ROOT / "logs" / "patchcore_v3.csv"))

    p1_rows = read_csv(ROOT / "logs" / "patchcore_p1.csv")
    p1_ext = ROOT / "logs" / "patchcore_p1_ext.csv"
    if p1_ext.exists():
        p1_rows += read_csv(p1_ext)
    per_category["patchcore_p1"] = seed_runs_to_per_category(p1_rows)

    production = dict(per_category["patchcore_v3"])
    screw_l123 = [r for r in p1_rows
                  if r["category"] == "screw" and r.get("feature_level") == "layer1+layer2+layer3"]
    if screw_l123:
        production["screw"] = seed_runs_to_per_category(screw_l123)["screw"]
    per_category["production_final"] = production

    macro = {model: macro_of(pc) for model, pc in per_category.items()}

    for model, expected in EXPECTED_MACRO.items():
        got = macro.get(model)
        ok = got is not None and abs(got - expected) <= 0.005
        print(f"[build] macro {model}: {got} (expected ~{expected}) {'OK' if ok else 'MISMATCH — investigate'}")

    benchmarks = {
        "per_category": {m: list(pc.values()) for m, pc in per_category.items()},
        "macro": macro,
    }
    (FRONTEND_DATA / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
    print(f"[build] wrote benchmarks.json ({len(per_category)} models)")

    insights = {
        "coreset_effect": [
            {"category": c,
             "coreset_10k": per_category["patchcore_v2"][c]["auroc"],
             "full_bank": per_category["patchcore_v3"][c]["auroc"],
             "delta": round(per_category["patchcore_v3"][c]["auroc"]
                            - per_category["patchcore_v2"][c]["auroc"], 4)}
            for c in sorted(per_category["patchcore_v3"])
            if c in per_category["patchcore_v2"]
        ],
        "aggregation_effect": [
            {"category": c,
             "topk_mean": per_category["patchcore_v1"][c]["auroc"],
             "topk_reweighted": per_category["patchcore_v2"][c]["auroc"],
             "delta": round(per_category["patchcore_v2"][c]["auroc"]
                            - per_category["patchcore_v1"][c]["auroc"], 4)}
            for c in sorted(per_category["patchcore_v2"])
            if c in per_category["patchcore_v1"]
        ],
        "layer_ablation": [
            {"category": cat,
             "configs": {
                 fl: round(mean(float(r["auroc"]) for r in p1_rows
                                if r["category"] == cat and r["feature_level"] == fl), 4)
                 for fl in sorted({r["feature_level"] for r in p1_rows if r["category"] == cat})
             }}
            for cat in sorted({r["category"] for r in p1_rows})
        ],
        "seed_stability": [
            {"category": c, "auroc_std": per_category["patchcore_v3"][c]["auroc_std"]}
            for c in sorted(per_category["patchcore_v3"])
        ],
    }
    (FRONTEND_DATA / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print("[build] wrote insights.json")


if __name__ == "__main__":
    main()
```
After Step 8.1: set `V3_SRC` (or leave None) and, if the located CSV uses different column names, print its header and adjust `ocgan_aggregated_to_per_category` accordingly.

- [ ] **Step 8.3: Run + verify macro checks**

```powershell
python scripts\build_webapp_data.py
```
Expected: `OK` for patchcore_v1/v2/v3 and production_final (≈0.9846); both JSON files written. On MISMATCH: diff per-category values against the README table to find the offending category before proceeding.

- [ ] **Step 8.4: Spot-check against the README**

```powershell
python -c "import json; d=json.load(open('../frontend/src/data/benchmarks.json')); pf={r['category']: r['auroc'] for r in d['per_category']['production_final']}; print(pf['bottle'], pf['screw'], pf['zipper'], d['macro']['production_final'])"
```
Expected: `1.0 0.9419 0.9801 0.9846` (±0.001 — README table values).

- [ ] **Step 8.5: Commit (script in backend repo; JSON is committed with the frontend plan)**

```powershell
git add scripts/build_webapp_data.py
git commit -m "feat: regenerate frontend benchmark/insight JSON from result CSVs"
```

---

### Task 9: Backend docs + full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 9.1: Full test suite**

```powershell
python -m pytest -q
```
Expected: all green (existing + new tests).

- [ ] **Step 9.2: Add a "Webapp" section to `README.md`** (replace the stale frontend bullet in "Open items")

```markdown
## Webapp

Showcase webapp: frontend in `../frontend` (React/Vite), served by this API in production.

    # one-time setup
    python -m venv .venv ; .\.venv\Scripts\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or plain `pip install torch torchvision` for CPU
    pip install -r requirements-webapp.txt
    python scripts/calibrate_variant_thresholds.py        # thresholds for reconstructed v1/v2
    python scripts/build_webapp_data.py                   # regenerate frontend benchmark JSON

    # run (also serves ../frontend/dist if built)
    python server.py --port 8000 --device auto

New API: `/api/meta`, `/api/dataset/thumb`, `/api/dataset/mask`, `/api/arena/start`,
`/api/arena/jobs/{id}` (poll), `/api/arena/jobs/{id}/stream` (SSE), `/api/arena/jobs/{id}/cancel`.
Both predict endpoints accept `model_variant`: `production` | `patchcore_v2` | `patchcore_v1`
(reconstructed from the production banks; thresholds recalibrated on the original val split).
```

- [ ] **Step 9.3: Commit**

```powershell
git add README.md
git commit -m "docs: webapp setup, variant semantics, new API surface"
```

---

## Self-review notes

- Spec coverage: variant engine (T2), calibration (T3), sampler/metrics/jobs (T4-T5), thumbs (T6), all endpoints + SPA mount + device auto (T7), benchmark JSON (T8), docs (T9). Frontend work is in `2026-06-10-mvtec-webapp-frontend.md`.
- Consistency watch-list during execution: `meta_cache_clear` name (test ↔ server), dataset `__getitem__` adaptation (Step 3.2), p95 index convention (Step 4.6), kcenter first-pick expectation (Step 1.4), draft benchmarks.json shape (Step 8.1).
