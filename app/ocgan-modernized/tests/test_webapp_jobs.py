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
