"""Unit tests for transforms that don't require actual images."""

import numpy as np
import pytest
import torch

import stable_pretraining.data.transforms as transforms


from PIL import Image


@pytest.mark.unit
class TestPatchMasking:
    """Test suite for transforms.PatchMasking transform."""

    @pytest.fixture
    def pil_sample(self):
        """Create a sample dict with PIL image (like HuggingFace dataset item)."""
        img = Image.new("RGB", (224, 224), color=(255, 128, 64))
        return {"image": img}

    @pytest.fixture
    def tensor_sample(self):
        """Create a sample dict with tensor image."""
        img = torch.rand(3, 224, 224)
        return {"image": img}

    @pytest.fixture
    def uint8_tensor_sample(self):
        """Create a sample dict with uint8 tensor image."""
        img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        return {"image": img}

    def test_exact_drop_ratio_pil(self, pil_sample):
        """Test that exact drop ratio is respected with PIL images."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(pil_sample)

        # Check output keys
        assert "masked_image" in result
        assert "patch_mask" in result

        # Check patch mask shape (224/16 = 14 patches per side)
        assert result["patch_mask"].shape == (14, 14)

        # Check exact drop ratio
        total_patches = 14 * 14
        kept_patches = result["patch_mask"].sum().item()
        masked_patches = total_patches - kept_patches
        expected_masked = int(total_patches * 0.5)

        assert masked_patches == expected_masked, (
            f"Expected exactly {expected_masked} masked patches, got {masked_patches}"
        )

        # Check output type matches input
        assert isinstance(result["masked_image"], Image.Image)

    def test_exact_drop_ratio_tensor(self, tensor_sample):
        """Test that exact drop ratio is respected with tensor images."""
        transform = transforms.PatchMasking(
            patch_size=32,
            drop_ratio=0.75,
            source="image",
            target="masked_image",
        )

        result = transform(tensor_sample)

        # Check patch mask shape (224/32 = 7 patches per side)
        assert result["patch_mask"].shape == (7, 7)

        # Check exact drop ratio
        total_patches = 7 * 7
        kept_patches = result["patch_mask"].sum().item()
        masked_patches = total_patches - kept_patches
        expected_masked = int(total_patches * 0.75)

        assert masked_patches == expected_masked

        # Check output type
        assert isinstance(result["masked_image"], torch.Tensor)
        assert result["masked_image"].shape == tensor_sample["image"].shape

    def test_fill_value_applied_correctly(self, tensor_sample):
        """Test that custom mask value is applied to masked patches."""
        fill_value = 0.5
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=1.0,  # Mask all patches
            source="image",
            target="masked_image",
            fill_value=fill_value,
        )

        result = transform(tensor_sample)

        # With drop_ratio=1.0, all patches should be masked
        assert result["patch_mask"].sum().item() == 0

        # All pixels should be fill_value
        assert torch.allclose(
            result["masked_image"], torch.tensor(fill_value), atol=1e-6
        )

    def test_no_masking_when_drop_ratio_zero(self, tensor_sample):
        """Test that no masking occurs when drop_ratio is 0."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.0,
            source="image",
            target="masked_image",
        )

        result = transform(tensor_sample)

        # All patches should be kept
        total_patches = (224 // 16) ** 2
        assert result["patch_mask"].sum().item() == total_patches

        # Image should be unchanged
        assert torch.allclose(result["masked_image"], tensor_sample["image"], atol=1e-6)

    def test_default_fill_value_pil(self, pil_sample):
        """Test that PIL images use default mask value of 128/255."""
        transform = transforms.PatchMasking(
            patch_size=224,  # One big patch
            drop_ratio=1.0,  # Mask it
            source="image",
            target="masked_image",
        )

        result = transform(pil_sample)

        # Convert to array to check values
        img_array = np.array(result["masked_image"])
        expected_value = 128

        # All pixels should be mid-gray
        assert np.allclose(img_array, expected_value, atol=1)

    def test_default_fill_value_tensor(self, tensor_sample):
        """Test that float tensors use default mask value of 0.0."""
        transform = transforms.PatchMasking(
            patch_size=224,  # One big patch
            drop_ratio=1.0,  # Mask it
            source="image",
            target="masked_image",
        )

        result = transform(tensor_sample)

        # All pixels should be 0.0
        assert torch.allclose(result["masked_image"], torch.tensor(0.0), atol=1e-6)

    def test_uint8_tensor_handling(self, uint8_tensor_sample):
        """Test that uint8 tensors are handled correctly."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(uint8_tensor_sample)

        # Should return float tensor (normalized)
        assert result["masked_image"].dtype == torch.float32
        assert result["masked_image"].min() >= 0.0
        assert result["masked_image"].max() <= 1.0

    def test_different_patch_sizes(self, tensor_sample):
        """Test with various patch sizes."""
        for patch_size in [8, 16, 32, 56]:
            transform = transforms.PatchMasking(
                patch_size=patch_size,
                drop_ratio=0.3,
                source="image",
                target="masked_image",
            )

            result = transform(tensor_sample)

            expected_patches_per_side = 224 // patch_size
            assert result["patch_mask"].shape == (
                expected_patches_per_side,
                expected_patches_per_side,
            )

    def test_non_divisible_image_size(self):
        """Test with image size not perfectly divisible by patch_size."""
        img = torch.rand(3, 225, 225)  # Not divisible by 16
        sample = {"image": img}

        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(sample)

        # Should handle gracefully (14x14 patches, ignoring remainder)
        assert result["patch_mask"].shape == (14, 14)
        assert result["masked_image"].shape == img.shape

    def test_custom_source_target_keys(self):
        """Test with custom dictionary keys."""
        img = Image.new("RGB", (224, 224))
        sample = {"my_image": img}

        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="my_image",
            target="my_output",
        )

        result = transform(sample)

        assert "my_output" in result
        assert "patch_mask" in result
        assert "my_image" in result  # Original should still be there

    def test_randomness(self, tensor_sample):
        """Test that different calls produce different masks."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result1 = transform(tensor_sample.copy())
        result2 = transform(tensor_sample.copy())

        # Masks should be different (with very high probability)
        assert not torch.equal(result1["patch_mask"], result2["patch_mask"])

    def test_grayscale_image(self):
        """Test with grayscale PIL image."""
        img = Image.new("L", (224, 224), color=128)
        sample = {"image": img}

        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(sample)

        assert isinstance(result["masked_image"], Image.Image)
        assert result["patch_mask"].shape == (14, 14)


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
