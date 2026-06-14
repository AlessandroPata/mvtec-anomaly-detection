import math

import pytest
import torch

from models.patchcore_common import aggregate_image_score, kcenter_greedy_select


class TestAggregate:
    def test_topk_mean(self):
        d = torch.tensor([[3.0, 1.0, 2.0]])
        assert aggregate_image_score(d, "topk_mean", 2).item() == pytest.approx(2.5)

    def test_topk_reweighted_hand_computed(self):
        # d=[4,2], k=2 → inv=[0.25,0.5]; softmax(inv)=[e^.25, e^.5]/Z → w=1-softmax
        d = torch.tensor([[4.0, 2.0]])
        e25, e50 = math.exp(0.25), math.exp(0.5)
        s4, s2 = e25 / (e25 + e50), e50 / (e25 + e50)
        w4, w2 = 1 - s4, 1 - s2
        expected = (4 * w4 + 2 * w2) / (w4 + w2)
        got = aggregate_image_score(d, "topk_reweighted", 2).item()
        assert got == pytest.approx(expected, rel=1e-5)

    def test_mean_and_max(self):
        d = torch.tensor([[1.0, 2.0, 3.0]])
        assert aggregate_image_score(d, "mean", 0).item() == pytest.approx(2.0)
        assert aggregate_image_score(d, "max", 0).item() == pytest.approx(3.0)

    def test_k_clamped_to_patches(self):
        d = torch.tensor([[1.0, 2.0]])
        assert aggregate_image_score(d, "topk_mean", 99).item() == pytest.approx(1.5)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            aggregate_image_score(torch.ones(1, 3), "nope", 1)


class TestKCenter:
    def test_deterministic_and_unique(self):
        torch.manual_seed(0)
        x = torch.randn(40, 4)
        a = kcenter_greedy_select(x, 8)
        b = kcenter_greedy_select(x, 8)
        assert torch.equal(a, b)
        assert len(set(a.tolist())) == 8

    def test_first_pick_is_farthest_from_mean(self):
        x = torch.tensor([[0.0], [1.0], [2.0], [10.0]])
        idx = kcenter_greedy_select(x, 2)
        assert idx[0].item() == 3  # 10.0 is farthest from mean 3.25
        assert idx[1].item() == 0  # then 0.0 is farthest from 10.0

    def test_k_ge_n_returns_all(self):
        x = torch.randn(5, 3)
        assert kcenter_greedy_select(x, 9).tolist() == [0, 1, 2, 3, 4]

    def test_candidate_pool_returns_valid_indices(self):
        x = torch.randn(100, 3)
        idx = kcenter_greedy_select(x, 5, candidate_pool_size=20)
        assert len(idx) == 5
        assert idx.max().item() < 100
