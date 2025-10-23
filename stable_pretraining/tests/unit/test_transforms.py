"""Unit tests for transforms that don't require actual images."""

import numpy as np
import pytest
import torch

import stable_pretraining.data.transforms as transforms


@pytest.mark.unit
class TestTransformUtils:
    """Test transform utilities and basic functionality."""

    def test_collator(self):
        """Test the Collator utility."""
        import stable_pretraining as spt

        assert spt.data.Collator._test()

    def test_compose_transforms(self):
        """Test composing multiple transforms."""
        transform = transforms.Compose(transforms.RGB(), transforms.ToImage())
        # Test with mock data in expected format (dict with 'image' key)
        mock_data = {"image": torch.randn(3, 32, 32)}
        result = transform(mock_data)
        assert isinstance(result, dict)
        assert "image" in result
        assert isinstance(result["image"], torch.Tensor)

    def test_to_image_transform(self):
        """Test ToImage transform with different inputs."""
        transform = transforms.ToImage()

        # Test with numpy array
        np_image = np.random.rand(32, 32, 3).astype(np.float32)
        data = {"image": np_image}
        result = transform(data)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, 32, 32)

        # Test with torch tensor
        torch_image = torch.randn(3, 32, 32)
        data = {"image": torch_image}
        result = transform(data)
        assert isinstance(result["image"], torch.Tensor)

    def test_rgb_transform(self):
        """Test RGB transform ensures 3 channels."""
        transform = transforms.RGB()

        # Test with grayscale image
        gray_image = torch.randn(1, 32, 32)
        data = {"image": gray_image}
        result = transform(data)
        assert result["image"].shape == (3, 32, 32)

        # Test with RGB image (should be unchanged)
        rgb_image = torch.randn(3, 32, 32)
        data = {"image": rgb_image}
        result = transform(data)
        assert result["image"].shape == (3, 32, 32)

    def test_normalize_transform(self):
        """Test normalization with mean and std."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.ToImage(mean=mean, std=std)

        # Create a tensor with known values
        image = torch.ones(3, 32, 32)
        data = {"image": image}
        result = transform(data)

        # Check that normalization was applied
        assert not torch.allclose(result["image"], image)

    def test_transform_params_initialization(self):
        """Test that transforms can be initialized with various parameters."""
        # Test each transform can be created
        transforms_to_test = [
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomChannelPermutation(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomRotation(degrees=90),
        ]

        for t in transforms_to_test:
            assert t is not None

    # ---------------------------
    # Spurious correlation tests
    # ---------------------------

    def test_add_sample_idx_transform(self):
        """Test that AddSampleIdx correctly increments indices."""
        transform = transforms.AddSampleIdx()
        x1 = {"image": torch.zeros(3, 32, 32)}
        x2 = {"image": torch.zeros(3, 32, 32)}
        out1 = transform(x1)
        out2 = transform(x2)
        assert out1["idx"] == 0
        assert out2["idx"] == 1

    def test_add_patch_transform(self):
        """Test that AddPatch overlays a colored patch."""
        img = torch.zeros(3, 32, 32)
        data = {"image": img.clone()}
        transform = transforms.AddPatch(
            patch_size=0.25, color=(1.0, 0.0, 0.0), position="top_left_corner"
        )
        result = transform(data)
        # Top-left corner should now contain red pixels
        patch_area = result["image"][:, :8, :8]
        assert torch.allclose(patch_area[0], torch.ones_like(patch_area[0]), atol=1e-3)
        assert torch.allclose(
            patch_area[1:], torch.zeros_like(patch_area[1:]), atol=1e-3
        )

    def test_add_color_tint_transform(self):
        """Test AddColorTint applies an additive tint."""
        img = torch.zeros(3, 16, 16)
        data = {"image": img}
        transform = transforms.AddColorTint(tint=(1.0, 0.5, 0.5), alpha=0.5)
        result = transform(data)
        # Image should not be all zeros anymore
        assert torch.any(result["image"] > 0)

    def test_add_border_transform(self):
        """Test AddBorder draws a colored border."""
        img = torch.zeros(3, 20, 20)
        data = {"image": img}
        transform = transforms.AddBorder(thickness=0.1, color=(0, 1, 0))
        result = transform(data)
        # Corners should have green (0,1,0)
        assert torch.allclose(result["image"][1, 0, 0], torch.tensor(1.0), atol=1e-3)
        assert torch.allclose(result["image"][0, 0, 0], torch.tensor(0.0), atol=1e-3)

    def test_add_watermark_transform(self, tmp_path):
        """Test AddWatermark overlays another image."""
        # Create a dummy watermark (white square)
        wm_path = tmp_path / "wm.png"
        from torchvision.utils import save_image

        save_image(torch.ones(3, 8, 8), wm_path)
        data = {"image": torch.zeros(3, 32, 32)}
        transform = transforms.AddWatermark(
            str(wm_path), size=0.25, position="center", alpha=1.0
        )
        result = transform(data)
        # There should be a bright region in the center
        center = result["image"][:, 12:20, 12:20]
        assert torch.mean(center) > 0.5

    def test_class_conditional_injector(self):
        """Test ClassConditionalInjector applies transform to correct labels only."""
        base_transform = transforms.AddPatch(color=(0, 1, 0))
        injector = transforms.ClassConditionalInjector(
            transformation=base_transform,
            target_labels=[1],
            proportion=1.0,
            total_samples=5,
            seed=42,
        )

        # Prepare samples with idx + label
        samples = [
            {"image": torch.zeros(3, 16, 16), "label": torch.tensor(label), "idx": idx}
            for idx, label in enumerate([0, 1, 1, 0, 1])
        ]

        outputs = [injector(s) for s in samples]

        # Check that only samples with label of 1 were modified
        for s_in, s_out in zip(samples, outputs):
            mean_pixel = s_out["image"].mean().item()
            if s_in["label"] == 1:
                assert mean_pixel > 0  # patch added
            else:
                assert mean_pixel == 0  # unchanged
