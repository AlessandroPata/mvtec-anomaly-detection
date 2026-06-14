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
