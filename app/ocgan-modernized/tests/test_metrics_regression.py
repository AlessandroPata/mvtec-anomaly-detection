"""Regression tests for the localization / calibration / fusion metric code.

The fast tests are pure NumPy/sklearn maths (no GPU, no dataset) and guard the
properties the relazione relies on. The live verification gate (verify_all over a
real category) is opt-in: it needs the GPU + the MVTec dataset, so it is skipped
unless RUN_GPU_VERIFY=1.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


# --- AUPRO ------------------------------------------------------------------

def _aupro():
    return pytest.importorskip("pixel_metrics").compute_aupro


class TestAupro:
    def test_perfect_localization_is_one(self):
        compute_aupro = _aupro()
        regions = [np.ones(50), np.ones(30)]
        normal = np.zeros(10_000)
        assert compute_aupro(regions, normal) == pytest.approx(1.0, abs=1e-6)

    def test_regions_weighted_equally(self):
        # one tiny perfectly-found region + one large fully-missed region -> ~0.5,
        # because AUPRO weights regions, not pixels (a 2px region == a 500px region).
        compute_aupro = _aupro()
        regions = [np.ones(2), np.zeros(500)]
        normal = np.zeros(10_000)
        assert compute_aupro(regions, normal) == pytest.approx(0.5, abs=0.1)

    def test_random_near_baseline(self):
        compute_aupro = _aupro()
        rng = np.random.default_rng(0)
        regions = [rng.random(50), rng.random(40)]
        normal = rng.random(20_000)
        # random scores: PRO tracks FPR, so normalized area ~ 0.15
        assert 0.08 < compute_aupro(regions, normal) < 0.30

    def test_empty_inputs_are_nan(self):
        compute_aupro = _aupro()
        assert np.isnan(compute_aupro([], np.zeros(10)))
        assert np.isnan(compute_aupro([np.ones(5)], np.array([])))


# --- GAN logistic fusion: version-independent applier -----------------------

class TestRawLogRegFusion:
    def test_matches_sklearn_predict_proba(self):
        gan_engine = pytest.importorskip("webapp.gan_engine")
        LogisticRegression = pytest.importorskip("sklearn.linear_model").LogisticRegression
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 7))
        y = (X[:, 0] + 0.5 * X[:, 3] - X[:, 5] > 0).astype(int)
        est = LogisticRegression(max_iter=500).fit(X, y)
        raw = gan_engine._as_raw_fusion(est)
        Xt = rng.normal(size=(25, 7))
        a = est.predict_proba(Xt)[:, 1]
        b = raw.predict_proba(Xt)[:, 1]
        assert np.max(np.abs(a - b)) < 1e-9

    def test_passthrough_none_and_idempotent(self):
        gan_engine = pytest.importorskip("webapp.gan_engine")
        assert gan_engine._as_raw_fusion(None) is None
        raw = gan_engine.RawLogRegFusion([1.0, -2.0], [0.5])
        assert gan_engine._as_raw_fusion(raw) is raw


# --- best-F1 threshold ------------------------------------------------------

class TestBestF1Threshold:
    def test_separable_gives_perfect_f1(self):
        mod = pytest.importorskip("recalibrate_thresholds")
        scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
        y = np.array([0, 0, 0, 1, 1, 1])
        thr, f1 = mod.best_f1_threshold(scores, y)
        assert f1 == pytest.approx(1.0)
        pred = (scores >= thr).astype(int)
        assert (pred == y).all()


# --- honest CV calibration --------------------------------------------------

class TestHonestCalibration:
    def test_honest_not_above_oracle_on_noisy_data(self):
        # oracle peeks at all labels to set the threshold; held-out CV cannot do
        # better in expectation -> honest_acc should not exceed oracle by much.
        mod = pytest.importorskip("honest_calibration")
        rng = np.random.default_rng(3)
        y = np.array([0] * 60 + [1] * 60)
        scores = np.where(y == 1, rng.normal(1.0, 1.0, size=120), rng.normal(0.0, 1.0, size=120))
        thr_oracle, _ = pytest.importorskip("recalibrate_thresholds").best_f1_threshold(scores, y)
        oracle_acc = float(((scores >= thr_oracle).astype(int) == y).mean())
        honest_acc, _std, _f1 = mod.honest_cv_accuracy(scores, y, k=5)
        assert honest_acc <= oracle_acc + 0.05
        assert 0.5 < honest_acc <= 1.0


# --- opt-in live GPU gate (verify_all) --------------------------------------

@pytest.mark.skipif(os.environ.get("RUN_GPU_VERIFY") != "1",
                    reason="set RUN_GPU_VERIFY=1 to run the live PatchCore verification gate")
def test_live_production_no_threshold_bug():
    """The bug this whole effort fixed: AUROC high but acc@thr collapses. Assert it
    stays fixed on one category through the exact server inference path."""
    import server
    from verify_all import evaluate
    server._device = "cuda"
    m = evaluate("bottle", "production", gan_cap=50)
    assert "skip" not in m, m.get("skip")
    assert m["auroc"] >= 0.95
    assert m["acc"] >= 0.85  # no THRESH_BUG: operating point tracks the ranking
