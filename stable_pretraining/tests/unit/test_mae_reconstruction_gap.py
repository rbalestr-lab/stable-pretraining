
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

benchmarks_path = Path(__file__).parent.parent.parent.parent / "benchmarks" / "imagenette"
sys.path.insert(0, str(benchmarks_path))

from utils import (
    DECODER_DIM_MAP,
    create_linear_decoder,
    create_mae_with_custom_decoder,
    parse_decoder_type,
)


@pytest.mark.unit
class TestMAEReconstructionGapUnit:

    def test_parse_decoder_type_linear(self):
        config = parse_decoder_type("linear")
        assert config == {"type": "linear"}

    def test_parse_decoder_type_transformer_base(self):
        config = parse_decoder_type("base-8b")
        assert config["type"] == "transformer"
        assert config["embed_dim"] == 512
        assert config["depth"] == 8
        assert config["num_heads"] == 8

    def test_parse_decoder_type_transformer_tiny(self):
        config = parse_decoder_type("tiny-4b")
        assert config["type"] == "transformer"
        assert config["embed_dim"] == 192
        assert config["depth"] == 4
        assert config["num_heads"] == 3  # 192 // 64 = 3

    def test_parse_decoder_type_transformer_small(self):
        config = parse_decoder_type("small-8b")
        assert config["type"] == "transformer"
        assert config["embed_dim"] == 384
        assert config["depth"] == 8
        assert config["num_heads"] == 6  # 384 // 64 = 6

    def test_parse_decoder_type_transformer_large(self):
        config = parse_decoder_type("large-8b")
        assert config["type"] == "transformer"
        assert config["embed_dim"] == 768
        assert config["depth"] == 8
        assert config["num_heads"] == 12  # 768 // 64 = 12

    def test_parse_decoder_type_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid decoder_type"):
            parse_decoder_type("invalid-format-string")

    def test_parse_decoder_type_invalid_size(self):
        with pytest.raises(ValueError, match="Invalid decoder size"):
            parse_decoder_type("invalid-8b")

    def test_parse_decoder_type_invalid_depth_format(self):
        with pytest.raises(ValueError, match="Invalid depth format"):
            parse_decoder_type("base-8x")

    def test_parse_decoder_type_invalid_depth_number(self):
        with pytest.raises(ValueError, match="Invalid depth number"):
            parse_decoder_type("base-invalidb")

    def test_create_linear_decoder(self):
        decoder = create_linear_decoder(encoder_embed_dim=192, patch_size=16, in_chans=3)
        assert isinstance(decoder, nn.Linear)
        assert decoder.in_features == 192
        assert decoder.out_features == 16 * 16 * 3

    def test_create_linear_decoder_different_params(self):
        decoder = create_linear_decoder(encoder_embed_dim=384, patch_size=14, in_chans=3)
        assert isinstance(decoder, nn.Linear)
        assert decoder.in_features == 384
        assert decoder.out_features == 14 * 14 * 3

    @patch("stable_pretraining.backbone.mae.vit_tiny_patch16")
    def test_create_mae_vit_tiny_transformer_decoder(self, mock_vit_tiny):
        mock_model = Mock()
        mock_vit_tiny.return_value = mock_model

        config = parse_decoder_type("base-8b")
        model = create_mae_with_custom_decoder("vit_tiny", config)

        mock_vit_tiny.assert_called_once_with(
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8
        )
        assert model == mock_model

    @patch("stable_pretraining.backbone.mae.vit_small_patch16")
    def test_create_mae_vit_small_transformer_decoder(self, mock_vit_small):
        mock_model = Mock()
        mock_vit_small.return_value = mock_model

        config = parse_decoder_type("tiny-4b")
        model = create_mae_with_custom_decoder("vit_small", config)

        mock_vit_small.assert_called_once_with(
            decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3
        )
        assert model == mock_model

    @patch("stable_pretraining.backbone.mae.vit_base_patch16")
    def test_create_mae_vit_base_transformer_decoder(self, mock_vit_base):
        mock_model = Mock()
        mock_vit_base.return_value = mock_model

        config = parse_decoder_type("small-8b")
        model = create_mae_with_custom_decoder("vit_base", config)

        mock_vit_base.assert_called_once_with(
            decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6
        )
        assert model == mock_model

    def test_create_mae_invalid_model(self):
        config = parse_decoder_type("base-8b")
        with pytest.raises(ValueError, match="Unknown model"):
            create_mae_with_custom_decoder("invalid_model", config)

    def test_signed_alpha_forward_positive(self):
        mock_module = Mock()
        mock_module.alpha = 1.0
        mock_module.mask_ratio = 0.75
        mock_module.classifier = Mock(return_value=torch.randn(2, 10))

        mock_backbone = Mock()
        mock_latent = torch.randn(2, 197, 192)
        mock_pred = torch.randn(2, 196, 768)
        mock_mask = torch.randint(0, 2, (2, 196)).bool()
        mock_backbone.return_value = (mock_latent, mock_pred, mock_mask)
        mock_backbone.patchify = Mock(return_value=torch.randn(2, 196, 768))
        mock_module.backbone = mock_backbone

        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "label": torch.randint(0, 10, (2,)),
        }

        def forward(self, batch, stage):
            imgs = batch["image"]
            latent, pred, mask = self.backbone(imgs, mask_ratio=self.mask_ratio)
            target = self.backbone.patchify(imgs)
            loss_rec = torch.nn.functional.mse_loss(pred[mask], target[mask])
            batch["embedding"] = latent[:, 0]
            batch["loss_rec"] = loss_rec

            if hasattr(self, "classifier") and self.alpha != 0.0:
                logits = self.classifier(batch["embedding"])
                loss_cls = torch.nn.functional.cross_entropy(logits, batch["label"])
                batch["loss"] = loss_rec + self.alpha * loss_cls
                batch["loss_cls"] = loss_cls
            else:
                batch["loss"] = loss_rec

            return batch

        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch.copy(), "train")

        assert "loss" in result
        assert "loss_rec" in result
        assert "loss_cls" in result
        assert "embedding" in result
        expected_loss = result["loss_rec"] + mock_module.alpha * result["loss_cls"]
        torch.testing.assert_close(result["loss"], expected_loss)

    def test_signed_alpha_forward_negative(self):
        mock_module = Mock()
        mock_module.alpha = -1.0
        mock_module.mask_ratio = 0.75
        mock_module.classifier = Mock(return_value=torch.randn(2, 10))

        mock_backbone = Mock()
        mock_latent = torch.randn(2, 197, 192)
        mock_pred = torch.randn(2, 196, 768)
        mock_mask = torch.randint(0, 2, (2, 196)).bool()
        mock_backbone.return_value = (mock_latent, mock_pred, mock_mask)
        mock_backbone.patchify = Mock(return_value=torch.randn(2, 196, 768))
        mock_module.backbone = mock_backbone

        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "label": torch.randint(0, 10, (2,)),
        }

        def forward(self, batch, stage):
            imgs = batch["image"]
            latent, pred, mask = self.backbone(imgs, mask_ratio=self.mask_ratio)
            target = self.backbone.patchify(imgs)
            loss_rec = torch.nn.functional.mse_loss(pred[mask], target[mask])
            batch["embedding"] = latent[:, 0]
            batch["loss_rec"] = loss_rec

            if hasattr(self, "classifier") and self.alpha != 0.0:
                logits = self.classifier(batch["embedding"])
                loss_cls = torch.nn.functional.cross_entropy(logits, batch["label"])
                batch["loss"] = loss_rec + self.alpha * loss_cls
                batch["loss_cls"] = loss_cls
            else:
                batch["loss"] = loss_rec

            return batch

        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch.copy(), "train")

        expected_loss = result["loss_rec"] + mock_module.alpha * result["loss_cls"]
        torch.testing.assert_close(result["loss"], expected_loss)
        assert result["loss"] < result["loss_rec"]

    def test_signed_alpha_forward_zero(self):
        mock_module = Mock()
        mock_module.alpha = 0.0
        mock_module.mask_ratio = 0.75

        mock_backbone = Mock()
        mock_latent = torch.randn(2, 197, 192)
        mock_pred = torch.randn(2, 196, 768)
        mock_mask = torch.randint(0, 2, (2, 196)).bool()
        mock_backbone.return_value = (mock_latent, mock_pred, mock_mask)
        mock_backbone.patchify = Mock(return_value=torch.randn(2, 196, 768))
        mock_module.backbone = mock_backbone

        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "label": torch.randint(0, 10, (2,)),
        }

        def forward(self, batch, stage):
            imgs = batch["image"]
            latent, pred, mask = self.backbone(imgs, mask_ratio=self.mask_ratio)
            target = self.backbone.patchify(imgs)
            loss_rec = torch.nn.functional.mse_loss(pred[mask], target[mask])
            batch["embedding"] = latent[:, 0]
            batch["loss_rec"] = loss_rec

            if hasattr(self, "classifier") and self.alpha != 0.0:
                logits = self.classifier(batch["embedding"])
                loss_cls = torch.nn.functional.cross_entropy(logits, batch["label"])
                batch["loss"] = loss_rec + self.alpha * loss_cls
                batch["loss_cls"] = loss_cls
            else:
                batch["loss"] = loss_rec

            return batch

        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch.copy(), "train")

        torch.testing.assert_close(result["loss"], result["loss_rec"])
        assert "loss_cls" not in result  # No classification loss computed

    def test_reconstruction_gap_metric_computation(self):
        trajectory = [
            {"epoch": 0, "mse": 1.0, "accuracy": 10.0, "alpha": 1.0},
            {"epoch": 10, "mse": 0.8, "accuracy": 30.0, "alpha": 1.0},
            {"epoch": 20, "mse": 0.6, "accuracy": 50.0, "alpha": 1.0},
            {"epoch": 30, "mse": 0.5, "accuracy": 60.0, "alpha": 1.0},
        ]

        mse_values = [t["mse"] for t in trajectory]
        acc_values = [t["accuracy"] for t in trajectory]

        mse_range = max(mse_values) - min(mse_values)
        acc_range = max(acc_values) - min(acc_values)

        assert mse_range == 0.5
        assert acc_range == 50.0

        assert all(trajectory[i]["mse"] >= trajectory[i + 1]["mse"] for i in range(len(trajectory) - 1))
        assert all(trajectory[i]["accuracy"] <= trajectory[i + 1]["accuracy"] for i in range(len(trajectory) - 1))
