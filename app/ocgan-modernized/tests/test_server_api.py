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

    def test_busy_409(self, client):
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
