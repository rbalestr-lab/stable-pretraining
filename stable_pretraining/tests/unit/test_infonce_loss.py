import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch

from stable_pretraining.losses import InfoNCELoss

DDP_GATHER_PATH = "stable_pretraining.losses.all_gather"


@pytest.mark.unit
class TestInfoNCELoss:
    """Unit tests for the InfoNCELoss (CLIP loss) function."""

    @patch(DDP_GATHER_PATH, side_effect=lambda x: [x])
    def test_loss_is_low_for_perfect_match(self, mock_all_gather):
        """Loss should be near-zero when image_feats == text_feats and features are orthonormal."""
        torch.manual_seed(0)
        batch_size, dim = 4, 8
        # use orthonormal rows so that off-diagonal similarities are exactly 0
        feats = F.normalize(torch.eye(dim)[:batch_size], dim=-1)
        loss_fn = InfoNCELoss()

        loss = loss_fn(image_feats=feats, text_feats=feats)

        assert loss.ndim == 0
        # relax tolerance slightly to avoid rare numerical flakes
        assert loss.item() < 1e-5
        # check that all_gather was called twice (once for images, once for texts)
        assert mock_all_gather.call_count == 2

    @patch(DDP_GATHER_PATH, side_effect=lambda x: [x])
    def test_loss_is_high_for_mismatch(self, mock_all_gather):
        """Loss should be high when positive pairs are swapped/orthogonal."""
        torch.manual_seed(0)
        # image_feats are identity, text_feats are reversed identity
        image_feats = torch.eye(2)
        text_feats = torch.flip(torch.eye(2), dims=[1])
        loss_fn = InfoNCELoss()

        loss = loss_fn(image_feats=image_feats, text_feats=text_feats)

        # expected loss: cross entropy when true logit = 0 and wrong logit = large
        s = 1.0 / 0.07
        expected_loss = -F.log_softmax(torch.tensor([0.0, s]), dim=0)[0]
        assert torch.allclose(loss, expected_loss, atol=1e-7, rtol=0)
        mock_all_gather.assert_called()

    @patch(DDP_GATHER_PATH, side_effect=lambda x: [x])
    def test_invariance_to_feature_magnitude(self, mock_all_gather):
        """Loss should be identical regardless of input vector magnitude."""
        torch.manual_seed(123)
        batch_size, dim = 8, 256
        image_feats = torch.randn(batch_size, dim)
        text_feats = torch.randn(batch_size, dim)
        loss_fn = InfoNCELoss()

        loss1 = loss_fn(image_feats=image_feats, text_feats=text_feats)
        # rescale both features by 100x
        loss2 = loss_fn(image_feats=image_feats * 100.0, text_feats=text_feats * 100.0)

        # normalize cancels out magnitude, so losses should match
        assert torch.allclose(loss1, loss2, atol=1e-7, rtol=1e-6)
        mock_all_gather.assert_called()

    @patch(DDP_GATHER_PATH, side_effect=lambda x: [x])
    def test_logit_scale_overrides_temperature(self, mock_all_gather):
        """A provided logit_scale should be used instead of temperature."""
        torch.manual_seed(42)
        batch_size, dim = 4, 128
        # create normalized features
        image_feats = F.normalize(torch.randn(batch_size, dim), dim=-1)
        text_feats = F.normalize(torch.randn(batch_size, dim), dim=-1)

        # set temperature very low (implies scale = 100)
        loss_fn = InfoNCELoss(temperature=0.01)

        # compare three cases: no logit_scale, float logit_scale, tensor logit_scale
        loss_temp = loss_fn(
            image_feats=image_feats, text_feats=text_feats, logit_scale=None
        )
        loss_float = loss_fn(
            image_feats=image_feats, text_feats=text_feats, logit_scale=20.0
        )
        loss_tensor = loss_fn(
            image_feats=image_feats,
            text_feats=text_feats,
            logit_scale=torch.tensor(20.0, requires_grad=True),
        )

        # different scales => different loss
        assert not torch.allclose(loss_temp, loss_float)
        # float and tensor versions should match exactly
        assert torch.allclose(loss_float, loss_tensor, atol=1e-7, rtol=0)
        mock_all_gather.assert_called()

    @patch(DDP_GATHER_PATH, side_effect=lambda x: [x])
    def test_symmetry_image_text(self, mock_all_gather):
        """Loss(img, txt) ~= Loss(txt, img) within numerical tolerance."""
        torch.manual_seed(7)
        x = F.normalize(torch.randn(5, 64), dim=-1).cpu()
        y = F.normalize(torch.randn(5, 64), dim=-1).cpu()
        loss_fn = InfoNCELoss()

        a = loss_fn(x, y).cpu()
        b = loss_fn(y, x).cpu()

        # allow tiny fp drift
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-6), f"{a=} {b=}"

    def _two_rank_disjoint_side_effect(x):
        # use an even batch size so the split is clean
        B = x.size(0)
        m = B // 2
        return [x[:m], x[m:]]  # no duplicates across shards

    @patch(DDP_GATHER_PATH, side_effect=_two_rank_disjoint_side_effect)
    def test_ddp_like_concat_keeps_diagonal_targets(self, mock_all_gather):
        """Simulate two ranks without duplicate maxima: ensure diagonal targets remain uniquely best."""
        torch.manual_seed(0)
        B, D = 3, 32
        img = F.normalize(torch.randn(B, D), dim=-1)
        txt = img.clone()  # perfect pairing

        # concatenation of disjoint shards yields a global batch with no duplicates
        loss = InfoNCELoss()(img, txt)

        # should still be very close to zero since matches are preserved and unique
        assert loss.item() < 1e-5
