import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses.dino import KoLeoLoss, DINOv2Loss


@pytest.mark.unit
class TestKoLeoLoss:
    """Unit tests for KoLeoLoss."""

    def test_initialization(self):
        loss_fn = KoLeoLoss(epsilon=1e-6)
        assert loss_fn.epsilon == 1e-6
        assert loss_fn.pdist.eps == 1e-6

    def test_forward_basic(self):
        """Test basic KoLeo loss behavior: clustered vs spread."""
        loss_fn = KoLeoLoss()

        # Case 1: Clustered
        x_clustered = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [1.01, -0.01]], dtype=torch.float32
        )
        x_clustered = F.normalize(x_clustered, dim=-1)
        # Should have small distances -> log(small) -> large negative -> neg log -> large positive
        loss_clustered = loss_fn(x_clustered)

        # Case 2: Spread (orthogonal)
        x_spread = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
        )
        x_spread = F.normalize(x_spread, dim=-1)
        loss_spread = loss_fn(x_spread)

        assert loss_clustered > loss_spread

    def test_forward_shapes(self):
        """Test KoLeo loss with different input shapes."""
        loss_fn = KoLeoLoss()
        B, D = 16, 128

        # [Batch, Dim]
        x = torch.randn(B, D)
        loss = loss_fn(x)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

        # [Views, Batch, Dim]
        V = 2
        x_views = torch.randn(V, B, D)
        loss_views = loss_fn(x_views)
        assert loss_views.ndim == 0
        assert not torch.isnan(loss_views)

    def test_single_sample_batch(self):
        """Test behavior with batch size < 2 (should return 0)."""
        loss_fn = KoLeoLoss()
        x = torch.randn(1, 128)
        loss = loss_fn(x)
        assert loss == 0.0


@pytest.mark.unit
class TestDINOv2LossIntegration:
    """Tests for DINOv2Loss integration with KoLeo."""

    def test_koleo_integration(self):
        n_views = 2
        batch_size = 4
        out_dim = 10
        embed_dim = 32

        # Mock inputs
        student_cls_logits = torch.randn(n_views, batch_size, out_dim)
        teacher_cls_probs = torch.softmax(
            torch.randn(n_views, batch_size, out_dim), dim=-1
        )
        student_cls_features = torch.randn(n_views, batch_size, embed_dim)

        # 1. Test with KoLeo weight > 0
        loss_fn = DINOv2Loss(
            koleo_loss_weight=1.0, dino_loss_weight=1.0, ibot_loss_weight=0.0
        )

        loss = loss_fn(
            student_cls_logits=student_cls_logits,
            teacher_cls_probs=teacher_cls_probs,
            student_cls_features=student_cls_features,
        )

        # Manually compute components
        dino_part = loss_fn.dino_loss(student_cls_logits, teacher_cls_probs)
        koleo_part = loss_fn.koleo_loss(student_cls_features)

        assert torch.isclose(loss, dino_part + koleo_part)

    def test_koleo_disabled(self):
        """Test that KoLeo loss is not computed when weight is 0."""
        n_views = 2
        batch_size = 4
        out_dim = 10
        embed_dim = 32

        student_cls_logits = torch.randn(n_views, batch_size, out_dim)
        teacher_cls_probs = torch.softmax(
            torch.randn(n_views, batch_size, out_dim), dim=-1
        )
        student_cls_features = torch.randn(n_views, batch_size, embed_dim)

        loss_fn = DINOv2Loss(
            koleo_loss_weight=0.0, dino_loss_weight=1.0, ibot_loss_weight=0.0
        )

        loss = loss_fn(
            student_cls_logits=student_cls_logits,
            teacher_cls_probs=teacher_cls_probs,
            student_cls_features=student_cls_features,
        )

        dino_part = loss_fn.dino_loss(student_cls_logits, teacher_cls_probs)
        assert torch.isclose(loss, dino_part)
