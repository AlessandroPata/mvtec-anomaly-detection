import pytest

from webapp.sampler import list_test_images, sample_test_images


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
