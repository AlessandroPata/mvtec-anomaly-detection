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
