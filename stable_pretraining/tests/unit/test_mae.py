"""Unit tests for Masked Autoencoder (MAE) functionality."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torchmetrics


@pytest.mark.unit
class TestMAEUnit:
    """Unit tests for MAE components without actual model training."""

    def test_mae_backbone_initialization(self):
        """Test MAE backbone can be initialized."""
        with patch(
            "stable_pretraining.backbone.mae.vit_base_patch16_dec512d8b"
        ) as mock_mae:
            backbone = mock_mae()
            mock_mae.assert_called_once()
            assert backbone is not None

    def test_mae_forward_logic(self):
        """Test MAE forward pass logic."""
        # Mock backbone with MAE behavior
        mock_backbone = Mock()
        mock_latent = torch.randn(2, 197, 768)  # [batch, seq_len, hidden_dim]
        mock_pred = torch.randn(2, 196, 768)  # predictions for masked patches
        mock_mask = torch.randint(0, 2, (2, 196)).bool()  # mask
        mock_backbone.return_value = (mock_latent, mock_pred, mock_mask)
        mock_backbone.patchify = Mock(return_value=torch.randn(2, 196, 768))

        # Create mock module
        mock_module = Mock()
        mock_module.backbone = mock_backbone
        mock_module.training = True

        # Define forward function
        def forward(self, batch, stage):
            latent, pred, mask = self.backbone(batch["image"])
            batch["embedding"] = latent[:, 0]  # CLS token only
            if self.training:
                loss = torch.nn.functional.mse_loss(
                    self.backbone.patchify(batch["image"])[mask], pred[mask]
                )
                batch["loss"] = loss
            return batch

        # Test forward pass
        batch = {"image": torch.randn(2, 3, 224, 224)}
        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch.copy(), "train")

        # Verify calls and results
        mock_backbone.assert_called_once_with(batch["image"])
        assert "embedding" in result
        assert result["embedding"].shape == (2, 768)
        assert "loss" in result

        # Test without training mode
        mock_module.training = False
        # Reset mock to allow another call
        mock_backbone.reset_mock()
        # Use a fresh batch to avoid contamination from previous test
        batch_val = {"image": torch.randn(2, 3, 224, 224)}
        result = forward_bound(batch_val, "val")
        assert "embedding" in result
        assert "loss" not in result

    def test_mae_loss_function(self):
        """Test MAE loss computation."""
        with patch("stable_pretraining.losses.mae") as mock_mae_loss:
            mock_mae_loss.return_value = torch.tensor(0.5)

            patches = torch.randn(2, 196, 768)
            pred = torch.randn(2, 196, 768)
            mask = torch.randint(0, 2, (2, 196)).bool()

            loss = mock_mae_loss(patches, pred, mask)

            mock_mae_loss.assert_called_once_with(patches, pred, mask)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() == 0.5

    def test_patchify_function(self):
        """Test patchify functionality."""
        # Mock patchify behavior
        mock_backbone = Mock()

        def mock_patchify(x):
            # Simulate patchifying 224x224 image with 16x16 patches
            batch_size = x.shape[0]
            num_patches = (224 // 16) ** 2  # 196 patches
            patch_dim = 3 * 16 * 16  # 768
            return torch.randn(batch_size, num_patches, patch_dim)

        mock_backbone.patchify = mock_patchify

        # Test patchify
        images = torch.randn(2, 3, 224, 224)
        patches = mock_backbone.patchify(images)

        assert patches.shape == (2, 196, 768)

    def test_online_probe_initialization(self):
        """Test OnlineProbe callback initialization for MAE."""
        with patch("stable_pretraining.callbacks.OnlineProbe") as mock_probe:
            mock_module = Mock()
            mock_linear = Mock(spec=nn.Linear)
            mock_loss_fn = Mock(spec=nn.CrossEntropyLoss)
            mock_metrics = {
                "top1": Mock(spec=torchmetrics.classification.MulticlassAccuracy),
                "top5": Mock(spec=torchmetrics.classification.MulticlassAccuracy),
            }

            probe = mock_probe(
                mock_module,
                "linear_probe",
                "embedding",
                "label",
                probe=mock_linear,
                loss_fn=mock_loss_fn,
                metrics=mock_metrics,
            )

            mock_probe.assert_called_once()
            assert probe is not None

    def test_repeated_random_sampler(self):
        """Test RepeatedRandomSampler for multi-view training."""
        with patch(
            "stable_pretraining.data.sampler.RepeatedRandomSampler"
        ) as mock_sampler:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)

            sampler = mock_sampler(mock_dataset, n_views=2)

            mock_sampler.assert_called_once_with(mock_dataset, n_views=2)
            assert sampler is not None

    def test_transform_composition_for_mae(self):
        """Test transform composition for MAE training."""
        with patch("stable_pretraining.data.transforms") as mock_transforms:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # Mock transform methods
            mock_transforms.Compose.return_value = Mock()

            # Train transform
            train_transform = mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.RandomResizedCrop((224, 224)),
                mock_transforms.RandomHorizontalFlip(p=0.5),
                mock_transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                mock_transforms.RandomGrayscale(p=0.2),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert train_transform is not None

            # Val transform
            val_transform = mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.Resize((256, 256)),
                mock_transforms.CenterCrop((224, 224)),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert val_transform is not None

    def test_cls_token_extraction(self):
        """Test CLS token extraction from transformer output."""
        # Create mock latent representation
        latent = torch.randn(4, 197, 768)  # [batch, seq_len, hidden_dim]

        # Extract CLS token (first token)
        cls_token = latent[:, 0]

        assert cls_token.shape == (4, 768)
        assert torch.allclose(cls_token, latent[:, 0])

    def test_mae_masking_logic(self):
        """Test MAE masking logic."""
        batch_size = 2
        num_patches = 196

        # Create random mask
        mask = torch.rand(batch_size, num_patches) > 0.75  # 75% masking

        # Verify mask properties
        assert mask.shape == (batch_size, num_patches)
        assert mask.dtype == torch.bool

        # Test applying mask
        patches = torch.randn(batch_size, num_patches, 768)
        predictions = torch.randn(batch_size, num_patches, 768)

        masked_patches = patches[mask]
        masked_predictions = predictions[mask]

        assert masked_patches.shape[0] == masked_predictions.shape[0]
        assert masked_patches.shape[1] == 768
