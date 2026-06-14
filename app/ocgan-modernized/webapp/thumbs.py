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
