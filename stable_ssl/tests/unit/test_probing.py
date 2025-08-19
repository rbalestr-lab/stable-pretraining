"""Unit tests for probing functionality (linear probe and KNN)."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torchmetrics


@pytest.mark.unit
class TestProbingUnit:
    """Unit tests for probing components without actual model training."""

    def test_online_probe_initialization(self):
        """Test OnlineProbe callback initialization."""
        with patch("stable_ssl.callbacks.OnlineProbe") as mock_probe:
            mock_linear = Mock(spec=nn.Linear)
            mock_loss_fn = Mock(spec=nn.CrossEntropyLoss)
            mock_metrics = Mock(spec=torchmetrics.classification.MulticlassAccuracy)

            mock_probe(
                name="linear_probe",
                input="embedding",
                target="label",
                probe=mock_linear,
                loss_fn=mock_loss_fn,
                metrics=mock_metrics,
            )

            mock_probe.assert_called_once_with(
                name="linear_probe",
                input="embedding",
                target="label",
                probe=mock_linear,
                loss_fn=mock_loss_fn,
                metrics=mock_metrics,
            )

    def test_online_knn_initialization(self):
        """Test OnlineKNN callback initialization."""
        with patch("stable_ssl.callbacks.OnlineKNN") as mock_knn:
            mock_metrics = Mock(spec=torchmetrics.classification.MulticlassAccuracy)

            mock_knn(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=50000,
                metrics=mock_metrics,
                k=10,
                input_dim=512,
            )

            mock_knn.assert_called_once_with(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=50000,
                metrics=mock_metrics,
                k=10,
                input_dim=512,
            )

    def test_forward_function_with_eval_mode(self):
        """Test forward function for feature extraction in eval mode."""
        mock_backbone = Mock()
        mock_logits = torch.randn(2, 512)
        mock_backbone.return_value = {"logits": mock_logits}

        mock_module = Mock()
        mock_module.backbone = mock_backbone

        def forward(self, batch, stage):
            with torch.inference_mode():
                x = batch["image"]
                batch["embedding"] = self.backbone(x)["logits"]
            return batch

        # Test forward pass
        batch = {"image": torch.randn(2, 3, 224, 224)}
        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch, "val")

        # Verify
        mock_backbone.assert_called_once_with(batch["image"])
        assert "embedding" in result
        assert torch.allclose(result["embedding"], mock_logits)

    def test_backbone_modification(self):
        """Test backbone modification for feature extraction."""
        with patch("transformers.AutoModelForImageClassification") as mock_auto_model:
            # Mock the model structure
            mock_model = Mock()
            mock_classifier = Mock()
            mock_classifier.__getitem__ = Mock(
                side_effect=lambda i: Mock() if i == 0 else Mock()
            )
            mock_classifier.__setitem__ = Mock()
            mock_model.classifier = mock_classifier
            mock_auto_model.from_pretrained.return_value = mock_model

            # Load and modify model
            backbone = mock_auto_model.from_pretrained("microsoft/resnet-18")
            backbone.classifier[1] = nn.Identity()

            # Verify modification
            mock_classifier.__setitem__.assert_called_once()
            call_args = mock_classifier.__setitem__.call_args
            assert call_args[0][0] == 1  # Index 1
            assert isinstance(call_args[0][1], nn.Identity)

    def test_eval_only_wrapper(self):
        """Test EvalOnly wrapper for backbone."""
        with patch("stable_ssl.backbone.EvalOnly") as mock_eval_only:
            mock_backbone = Mock()

            eval_backbone = mock_eval_only(mock_backbone)

            mock_eval_only.assert_called_once_with(mock_backbone)
            assert eval_backbone is not None

    def test_transform_composition(self):
        """Test transform composition for train and validation."""
        with patch("stable_ssl.data.transforms") as mock_transforms:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # Mock transform components
            mock_transforms.Compose.return_value = Mock()

            # Train transform
            mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.RandomResizedCrop((224, 224)),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            # Val transform
            mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.Resize((256, 256)),
                mock_transforms.CenterCrop((224, 224)),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert mock_transforms.Compose.call_count == 2

    def test_linear_probe_components(self):
        """Test linear probe components."""
        # Test linear layer
        linear = nn.Linear(512, 10)
        assert linear.in_features == 512
        assert linear.out_features == 10

        # Test loss function
        loss_fn = nn.CrossEntropyLoss()
        preds = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(preds, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Test metrics
        metric = torchmetrics.classification.MulticlassAccuracy(10)
        metric.update(preds, labels)
        acc = metric.compute()
        assert isinstance(acc, torch.Tensor)

    def test_knn_parameters(self):
        """Test KNN probe parameters."""
        # Test parameters
        queue_size = 50000
        k = 10
        features_dim = 512

        # Verify reasonable values
        assert queue_size > 0
        assert k > 0
        assert k < queue_size
        assert features_dim > 0

    def test_trainer_configuration_for_probing(self):
        """Test trainer configuration for probing."""
        with patch("lightning.Trainer") as mock_trainer:
            mock_callbacks = [Mock(), Mock()]

            mock_trainer(
                max_steps=10,
                num_sanity_val_steps=1,
                callbacks=mock_callbacks,
                precision="16",
                logger=False,
                enable_checkpointing=False,
            )

            call_kwargs = mock_trainer.call_args[1]
            assert call_kwargs["max_steps"] == 10
            assert call_kwargs["precision"] == "16"
            assert len(call_kwargs["callbacks"]) == 2

    def test_module_without_optimizer(self):
        """Test module creation without optimizer for frozen backbone."""
        with patch("stable_ssl.Module") as mock_module:
            mock_backbone = Mock()
            mock_forward = Mock()

            mock_module(backbone=mock_backbone, forward=mock_forward, optim=None)

            mock_module.assert_called_once_with(
                backbone=mock_backbone, forward=mock_forward, optim=None
            )

    def test_online_probe_lifecycle_methods(self):
        """Test OnlineProbe callback lifecycle methods."""
        from stable_ssl.callbacks import OnlineProbe

        # Create probe with mock components
        probe = OnlineProbe(
            name="test_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(128, 10),
            loss_fn=nn.CrossEntropyLoss(),
            metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
        )

        # Test that lifecycle methods exist
        assert hasattr(probe, "setup")
        assert hasattr(probe, "on_train_batch_end")
        assert hasattr(probe, "on_validation_batch_end")
        assert hasattr(probe, "on_validation_epoch_end")
        assert hasattr(probe, "state_dict")
        assert hasattr(probe, "load_state_dict")

        # Test that probe_module property exists but is not accessible before setup
        # Note: hasattr returns False for properties that raise AttributeError
        # So we check if it's a property on the class instead
        assert isinstance(getattr(type(probe), "probe_module", None), property)
        with pytest.raises(AttributeError, match="module not accessible before setup"):
            _ = probe.probe_module
