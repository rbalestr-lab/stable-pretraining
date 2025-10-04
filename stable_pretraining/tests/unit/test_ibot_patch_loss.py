import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses.dino import iBOTPatchLoss


@pytest.mark.unit
class TestiBOTPatchLoss:
    """Unit tests for the iBOTPatchLoss function."""

    def test_initialization(self):
        """Test proper initialization of iBOTPatchLoss."""
        loss_fn = iBOTPatchLoss(student_temp=0.1)

        assert loss_fn.student_temp == 0.1

    def test_forward_with_perfect_match(self):
        """Loss with perfect match should be lower than with mismatch."""
        torch.manual_seed(0)
        n_masked, dim = 8, 16

        loss_fn = iBOTPatchLoss(student_temp=0.1)

        # Create teacher distribution for masked patches
        teacher_probs = F.softmax(torch.randn(n_masked, dim), dim=-1)

        # Perfect match: student logits that produce same probabilities
        student_logits_perfect = torch.log(teacher_probs + 1e-8) * loss_fn.student_temp

        # Poor match: random student logits
        student_logits_random = torch.randn(n_masked, dim)

        loss_perfect = loss_fn.forward(student_logits_perfect, teacher_probs)
        loss_random = loss_fn.forward(student_logits_random, teacher_probs)

        assert loss_perfect.ndim == 0  # scalar
        assert loss_random.ndim == 0  # scalar
        assert loss_perfect < loss_random  # perfect match should have lower loss

    def test_forward_with_mismatch(self):
        """Loss should be high when student predictions are wrong."""
        torch.manual_seed(0)
        n_masked, dim = 8, 16

        loss_fn = iBOTPatchLoss(student_temp=0.1)

        # Teacher predicts one thing, student predicts opposite
        teacher_probs = torch.zeros(n_masked, dim)
        teacher_probs[:, 0] = 1.0  # All probability on first class

        student_logits = torch.zeros(n_masked, dim)
        student_logits[:, -1] = 100.0  # All probability on last class

        loss = loss_fn.forward(student_logits, teacher_probs)

        assert loss.ndim == 0
        assert loss.item() > 1.0  # high loss for mismatch

    def test_sinkhorn_knopp_teacher(self):
        """Test Sinkhorn-Knopp normalization of teacher predictions."""
        torch.manual_seed(42)
        n_samples, dim = 50, 32

        loss_fn = iBOTPatchLoss(student_temp=0.1)

        teacher_logits = torch.randn(n_samples, dim)

        probs = loss_fn.sinkhorn_knopp_teacher(
            teacher_logits,
            teacher_temp=0.1,
            num_samples=n_samples,
            n_iterations=3,
        )

        assert probs.shape == (n_samples, dim)
        # Each sample should be a probability distribution
        assert torch.allclose(probs.sum(dim=-1), torch.ones(n_samples), atol=1e-4)
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_loss_is_positive(self):
        """Cross-entropy loss should always be positive."""
        torch.manual_seed(0)
        n_masked, dim = 15, 20

        loss_fn = iBOTPatchLoss(student_temp=0.1)

        teacher_probs = F.softmax(torch.randn(n_masked, dim), dim=-1)
        student_logits = torch.randn(n_masked, dim)

        loss = loss_fn.forward(student_logits, teacher_probs)

        # Cross-entropy loss should be positive
        assert loss.item() >= 0

    def test_temperature_effect(self):
        """Higher temperature should make loss less sensitive to errors."""
        torch.manual_seed(0)
        n_masked, dim = 8, 16

        # High temperature (0.5)
        loss_fn_high_temp = iBOTPatchLoss(student_temp=0.5)

        # Low temperature (0.05)
        loss_fn_low_temp = iBOTPatchLoss(student_temp=0.05)

        # Create somewhat mismatched predictions
        teacher_probs = F.softmax(torch.randn(n_masked, dim), dim=-1)
        student_logits = torch.randn(n_masked, dim)

        loss_high = loss_fn_high_temp.forward(student_logits, teacher_probs)
        loss_low = loss_fn_low_temp.forward(student_logits, teacher_probs)

        # Low temperature should have higher loss (more sensitive to errors)
        assert loss_low.item() > loss_high.item()

    def test_batch_size_invariance(self):
        """Test that loss is invariant to batch size (mean over patches)."""
        torch.manual_seed(0)
        dim = 16

        loss_fn = iBOTPatchLoss(student_temp=0.1)

        # Small batch
        n_masked_small = 4
        teacher_probs_small = F.softmax(torch.randn(n_masked_small, dim), dim=-1)
        student_logits_small = torch.randn(n_masked_small, dim)
        loss_small = loss_fn.forward(student_logits_small, teacher_probs_small)

        # Large batch (double the patches, same distributions)
        teacher_probs_large = torch.cat(
            [teacher_probs_small, teacher_probs_small], dim=0
        )
        student_logits_large = torch.cat(
            [student_logits_small, student_logits_small], dim=0
        )
        loss_large = loss_fn.forward(student_logits_large, teacher_probs_large)

        # Losses should be the same (mean is invariant to duplication)
        assert torch.allclose(loss_small, loss_large, atol=1e-6)
