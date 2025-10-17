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

    def test_add_sample_idx_transform(self):
        """Test adding the sample idex for the injections."""
        pass

    def test_add_patch_transform(self):
        """Test adding the spurious patch into images."""
        pass

    def test_class_conditional_injector(self):
        """Test conditional injections."""
        pass

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
            transforms.AddSampleIdx(),
            transforms.ClassConditionalInjector(
                transformation=transforms.AddPatch(
                    patch_size=0.1, color=(1.0, 0.0, 0.0), position="center"
                ),
                total_samples=10000,
            ),
        ]

        for t in transforms_to_test:
            assert t is not None
