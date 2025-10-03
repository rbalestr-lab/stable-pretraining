import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses.dino import iBOTPatchLoss


@pytest.mark.unit
class TestiBOTPatchLoss:
    """Unit tests for the iBOTPatchLoss function."""

    def test_initialization(self):
        """Test proper initialization of iBOTPatchLoss."""
        loss_fn = iBOTPatchLoss(
            patch_out_dim=256, student_temp=0.1, center_momentum=0.9
        )

        assert loss_fn.student_temp == 0.1
        assert loss_fn.center_momentum == 0.9
        # Center should be None initially (lazy initialization)
        assert loss_fn.center is None

    def test_forward_with_perfect_match(self):
        """Loss with perfect match should be lower than with mismatch."""
        torch.manual_seed(0)
        batch_size, n_patches, dim = 2, 4, 16

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        # Create teacher distribution
        teacher_probs = F.softmax(torch.randn(batch_size, n_patches, dim), dim=-1)

        # Perfect match: student logits that produce same probabilities
        student_logits_perfect = torch.log(teacher_probs + 1e-8) * loss_fn.student_temp

        # Poor match: random student logits
        student_logits_random = torch.randn(batch_size, n_patches, dim)

        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)

        loss_perfect = loss_fn.forward(student_logits_perfect, teacher_probs, masks)
        loss_random = loss_fn.forward(student_logits_random, teacher_probs, masks)

        assert loss_perfect.ndim == 0  # scalar
        assert loss_random.ndim == 0  # scalar
        assert loss_perfect < loss_random  # perfect match should have lower loss

    def test_forward_with_mismatch(self):
        """Loss should be high when student predictions are wrong."""
        torch.manual_seed(0)
        batch_size, n_patches, dim = 2, 4, 16

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        # Teacher predicts one thing, student predicts opposite
        teacher_probs = torch.zeros(batch_size, n_patches, dim)
        teacher_probs[:, :, 0] = 1.0  # All probability on first class

        student_logits = torch.zeros(batch_size, n_patches, dim)
        student_logits[:, :, -1] = 100.0  # All probability on last class

        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)

        loss = loss_fn.forward(student_logits, teacher_probs, masks)

        assert loss.ndim == 0
        assert loss.item() > 1.0  # high loss for mismatch

    def test_forward_masked_with_subset(self):
        """Test forward_masked with only masked patches."""
        torch.manual_seed(42)
        batch_size, n_patches, dim = 4, 8, 32

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        # Create masks (some patches masked, some not)
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        masks[:, ::2] = True  # Mask every other patch

        # Extract only masked patches
        n_masked = masks.sum().item()
        teacher_probs_masked = F.softmax(torch.randn(n_masked, dim), dim=-1)
        student_logits_masked = torch.randn(n_masked, dim)

        loss = loss_fn.forward_masked(
            student_logits_masked, teacher_probs_masked, masks
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_masked_with_weights(self):
        """Test forward_masked with custom per-patch weights."""
        torch.manual_seed(123)
        batch_size, n_patches, dim = 2, 4, 16

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)
        n_masked = batch_size * n_patches

        teacher_probs_masked = F.softmax(torch.randn(n_masked, dim), dim=-1)
        student_logits_masked = torch.randn(n_masked, dim)

        # Custom weights (first half weighted more)
        custom_weights = torch.ones(n_masked)
        custom_weights[: n_masked // 2] = 2.0

        loss_with_weights = loss_fn.forward_masked(
            student_logits_masked,
            teacher_probs_masked,
            masks,
            masks_weight=custom_weights,
        )

        loss_without_weights = loss_fn.forward_masked(
            student_logits_masked, teacher_probs_masked, masks
        )

        # Losses should be different due to different weights
        assert not torch.allclose(loss_with_weights, loss_without_weights)

    def test_forward_masked_with_truncation(self):
        """Test forward_masked with n_masked_patches truncation."""
        torch.manual_seed(7)
        batch_size, n_patches, dim = 2, 8, 16

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)
        n_total = batch_size * n_patches

        teacher_probs_masked = F.softmax(torch.randn(n_total, dim), dim=-1)
        student_logits_masked = torch.randn(n_total, dim)

        # Use only first 8 patches
        n_truncate = 8
        loss_truncated = loss_fn.forward_masked(
            student_logits_masked,
            teacher_probs_masked,
            masks,
            n_masked_patches=n_truncate,
        )

        # Should compute loss over truncated patches
        assert loss_truncated.ndim == 0
        assert not torch.isnan(loss_truncated)

    def test_softmax_center_teacher(self):
        """Test centering of teacher predictions."""
        torch.manual_seed(0)
        n_samples, dim = 100, 64

        loss_fn = iBOTPatchLoss(patch_out_dim=dim, center_momentum=0.9)

        teacher_logits = torch.randn(n_samples, dim)

        # First call: center is None, should just do softmax without centering
        probs1 = loss_fn.softmax_center_teacher(
            teacher_logits, teacher_temp=0.1, update_centers=False
        )

        assert probs1.shape == (n_samples, dim)
        assert torch.allclose(probs1.sum(dim=-1), torch.ones(n_samples), atol=1e-5)

        # Update center
        loss_fn.update_center(teacher_logits.unsqueeze(0))
        loss_fn.apply_center_update()

        # Center should have been initialized (not None anymore)
        assert loss_fn.center is not None

        # Second call with updated center
        probs2 = loss_fn.softmax_center_teacher(
            teacher_logits, teacher_temp=0.1, update_centers=False
        )

        # Probabilities should be different due to centering
        assert not torch.allclose(probs1, probs2)

    def test_sinkhorn_knopp_teacher(self):
        """Test Sinkhorn-Knopp normalization of teacher predictions."""
        torch.manual_seed(42)
        n_samples, dim = 50, 32

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        teacher_logits = torch.randn(n_samples, dim)

        probs = loss_fn.sinkhorn_knopp_teacher(
            teacher_logits,
            teacher_temp=0.1,
            n_masked_patches_tensor=torch.tensor(n_samples),
            n_iterations=3,
        )

        assert probs.shape == (n_samples, dim)
        # Each sample should be a probability distribution
        assert torch.allclose(probs.sum(dim=-1), torch.ones(n_samples), atol=1e-4)
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_center_update_ema(self):
        """Test that center update uses exponential moving average."""
        torch.manual_seed(0)
        n_samples, dim = 10, 16
        momentum = 0.9

        loss_fn = iBOTPatchLoss(patch_out_dim=dim, center_momentum=momentum)

        # First batch - iBOT takes 2D input [n_masked, dim]
        teacher_logits1 = torch.randn(n_samples, dim)
        loss_fn.update_center(teacher_logits1)
        loss_fn.apply_center_update()
        center1 = loss_fn.center.clone()

        # Second batch
        teacher_logits2 = torch.randn(n_samples, dim)
        loss_fn.update_center(teacher_logits2)
        loss_fn.apply_center_update()
        center2 = loss_fn.center.clone()

        # Center should change but smoothly (due to momentum)
        assert not torch.allclose(center1, center2)

        # Verify EMA formula: new = old * momentum + batch * (1 - momentum)
        # teacher_logits2 is (n_samples, dim), mean over dim 0 with keepdim gives (1, dim)
        batch_mean = teacher_logits2.mean(dim=0, keepdim=True)
        expected_center2 = center1 * momentum + batch_mean * (1 - momentum)
        assert torch.allclose(center2, expected_center2, atol=1e-5)

    def test_loss_is_positive(self):
        """Cross-entropy loss should always be positive."""
        torch.manual_seed(0)
        batch_size, n_patches, dim = 3, 5, 20

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        teacher_probs = F.softmax(torch.randn(batch_size, n_patches, dim), dim=-1)
        student_logits = torch.randn(batch_size, n_patches, dim)
        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)

        loss = loss_fn.forward(student_logits, teacher_probs, masks)

        # Cross-entropy loss should be positive
        assert loss.item() >= 0

    def test_temperature_effect(self):
        """Higher temperature should make loss less sensitive to errors."""
        torch.manual_seed(0)
        batch_size, n_patches, dim = 2, 4, 16

        # High temperature (0.5)
        loss_fn_high_temp = iBOTPatchLoss(patch_out_dim=dim, student_temp=0.5)

        # Low temperature (0.05)
        loss_fn_low_temp = iBOTPatchLoss(patch_out_dim=dim, student_temp=0.05)

        # Create somewhat mismatched predictions
        teacher_probs = F.softmax(torch.randn(batch_size, n_patches, dim), dim=-1)
        student_logits = torch.randn(batch_size, n_patches, dim)
        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)

        loss_high = loss_fn_high_temp.forward(student_logits, teacher_probs, masks)
        loss_low = loss_fn_low_temp.forward(student_logits, teacher_probs, masks)

        # Low temperature should have higher loss (more sensitive to errors)
        assert loss_low.item() > loss_high.item()

    def test_mask_weighting_consistency(self):
        """Test that mask weighting is consistent across different mask patterns."""
        torch.manual_seed(0)
        batch_size, n_patches, dim = 4, 8, 16

        loss_fn = iBOTPatchLoss(patch_out_dim=dim)

        # Uniform mask: all patches masked equally
        masks_uniform = torch.ones(batch_size, n_patches, dtype=torch.bool)

        # Non-uniform mask: different number of patches per sample
        masks_nonuniform = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        masks_nonuniform[0, :2] = True  # 2 patches
        masks_nonuniform[1, :4] = True  # 4 patches
        masks_nonuniform[2, :6] = True  # 6 patches
        masks_nonuniform[3, :8] = True  # 8 patches

        teacher_probs = F.softmax(torch.randn(batch_size, n_patches, dim), dim=-1)
        student_logits = torch.randn(batch_size, n_patches, dim)

        loss_uniform = loss_fn.forward(student_logits, teacher_probs, masks_uniform)
        loss_nonuniform = loss_fn.forward(
            student_logits, teacher_probs, masks_nonuniform
        )

        # Both should produce valid scalar losses
        assert loss_uniform.ndim == 0
        assert loss_nonuniform.ndim == 0
        assert not torch.isnan(loss_uniform)
        assert not torch.isnan(loss_nonuniform)
