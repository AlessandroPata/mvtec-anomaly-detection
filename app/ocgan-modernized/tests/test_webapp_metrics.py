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
