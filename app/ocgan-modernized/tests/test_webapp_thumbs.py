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
