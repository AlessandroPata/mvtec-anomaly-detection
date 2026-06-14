"""Smoke test: rebuild one OCGAN checkpoint and predict a good + a defect image.

Usage: python scripts/smoke_gan_predict.py [device] [variant] [category]
"""
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.gan_engine import GanInference  # noqa: E402

DATASET_ROOT = ROOT.parent.parent / "datasets" / "mvtec_ad"


def first_image(folder: Path) -> Path:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")[0]


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    variant = sys.argv[2] if len(sys.argv) > 2 else "ocgan_final"
    cat = sys.argv[3] if len(sys.argv) > 3 else "bottle"

    t0 = time.perf_counter()
    model = GanInference(cat, variant, device=device)
    print(f"build: {time.perf_counter() - t0:.1f}s threshold={model.threshold:.4f} "
          f"probability_scores={model.is_probability}")

    good = first_image(DATASET_ROOT / cat / "test" / "good")
    defect_dir = sorted(d for d in (DATASET_ROOT / cat / "test").iterdir() if d.is_dir() and d.name != "good")[0]
    defect = first_image(defect_dir)

    r_good = model.predict(Image.open(good).convert("RGB"))
    r_def = model.predict(Image.open(defect).convert("RGB"))

    print(f"good   {good.name}: score={r_good['anomaly_score']:.4f} thr={r_good['threshold']:.4f} "
          f"anomalous={r_good['is_anomalous']} ({r_good['inference_time_ms']:.0f} ms)")
    print(f"defect {defect_dir.name}/{defect.name}: score={r_def['anomaly_score']:.4f} "
          f"anomalous={r_def['is_anomalous']} ({r_def['inference_time_ms']:.0f} ms)")
    print(f"heatmap: shape={r_def['heatmap'].shape} range=({r_def['heatmap'].min():.2f},{r_def['heatmap'].max():.2f})")
    print(f"components(defect): {r_def['score_components']}")
    assert r_def["anomaly_score"] > r_good["anomaly_score"], "defect must score higher than good"
    print("GAN SMOKE OK")


if __name__ == "__main__":
    main()
