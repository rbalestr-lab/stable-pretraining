"""Unit tests for EfficientMaskedTimmViT."""

import pytest
import torch
import torch.nn as nn
import timm

# Import your module - adjust the import path as needed
# Assuming the class is in a file called efficient_masked_vit.py
from stable_pretraining.backbone import EfficientMaskedTimmViT


# ============================================================================
# Test Configuration
# ============================================================================

# List of timm models to test
TIMM_MODELS = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
]

# Smaller set for quick tests
QUICK_MODELS = [
    "vit_tiny_patch16_224",
    "deit_tiny_patch16_224",
]

BATCH_SIZE = 4
IMAGE_SIZE = 224
CHANNELS = 3


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=QUICK_MODELS)
def model_name(request):
    """Fixture providing model names for parametrized tests."""
    return request.param


@pytest.fixture
def vit_model(model_name):
    """Fixture providing a fresh timm ViT model."""
    return timm.create_model(model_name, pretrained=False, num_classes=1000)


@pytest.fixture
def masked_vit(vit_model):
    """Fixture providing a wrapped masked ViT model."""
    return EfficientMaskedTimmViT(vit_model)


@pytest.fixture
def sample_input():
    """Fixture providing sample input images."""
    return torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def sample_patches(vit_model):
    """Fixture providing sample patch embeddings."""
    x = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        patches = vit_model.patch_embed(x)
    return patches


# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.unit
class TestBasicFunctionality:
    """Test basic operations of EfficientMaskedTimmViT."""

    def test_initialization(self, vit_model):
        """Test that model initializes correctly."""
        masked_vit = EfficientMaskedTimmViT(vit_model)
        assert masked_vit.vit is vit_model
        assert hasattr(masked_vit, "forward")

    def test_initialization_invalid_model(self):
        """Test that initialization fails with invalid model."""
        invalid_model = nn.Linear(10, 10)
        with pytest.raises(RuntimeError, match="patch_embed"):
            EfficientMaskedTimmViT(invalid_model)

    def test_forward_no_nans(self, masked_vit, sample_input):
        """Test forward pass with no NaN patches."""
        output = masked_vit(sample_input)
        assert output.shape[0] == BATCH_SIZE
        assert not torch.isnan(output).any(), "Output should not contain NaNs"

    def test_output_shape(self, masked_vit, vit_model, sample_input):
        """Test that output shape matches original model."""
        masked_output = masked_vit(sample_input)

        with torch.no_grad():
            original_output = vit_model(sample_input)

        assert masked_output.shape == original_output.shape, (
            f"Shape mismatch: {masked_output.shape} vs {original_output.shape}"
        )

    def test_deterministic_output(self, masked_vit, sample_input):
        """Test that output is deterministic for same input."""
        masked_vit.eval()
        with torch.no_grad():
            output1 = masked_vit(sample_input)
            output2 = masked_vit(sample_input)

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)


# ============================================================================
# NaN Handling Tests
# ============================================================================


@pytest.mark.unit
class TestNaNHandling:
    """Test NaN patch handling functionality."""

    def test_with_nan_patches(self, masked_vit, vit_model, sample_input):
        """Test forward pass with NaN patches."""
        with torch.no_grad():
            patches = vit_model.patch_embed(sample_input)

            # Add same number of NaN patches to each sample (but different locations)
            patches[0, 10:12, :] = float("nan")  # Patches 10-11 for sample 0
            patches[1, 20:22, :] = float("nan")  # Patches 20-21 for sample 1
            patches[2, 5:7, :] = float("nan")  # Patches 5-6 for sample 2
            patches[3, 15:17, :] = float("nan")  # Patches 15-16 for sample 3

        output = masked_vit(patches)

        assert output.shape[0] == BATCH_SIZE
        assert not torch.isnan(output).any(), "Output should not contain NaNs"

    def test_all_nan_patches_raises_error(self, masked_vit, sample_patches):
        """Test that all-NaN input raises appropriate error."""
        sample_patches[:] = float("nan")

        with pytest.raises(ValueError, match="All patches are NaN"):
            masked_vit(sample_patches)

    def test_mismatched_nan_counts_raises_error(self, masked_vit, sample_patches):
        """Test that mismatched NaN counts raise error."""
        # Different number of NaN patches per sample
        sample_patches[0, 10:12, :] = float("nan")  # 2 NaN patches
        sample_patches[1, 20:24, :] = float("nan")  # 4 NaN patches (different count!)

        with pytest.raises(ValueError, match="same number of NaN patches"):
            masked_vit(sample_patches)

    def test_single_nan_patch(self, masked_vit, sample_patches):
        """Test with single NaN patch per sample."""
        sample_patches[0, 10, :] = float("nan")
        sample_patches[1, 20, :] = float("nan")
        sample_patches[2, 5, :] = float("nan")
        sample_patches[3, 15, :] = float("nan")

        output = masked_vit(sample_patches)
        assert not torch.isnan(output).any()

    def test_many_nan_patches(self, masked_vit, vit_model, sample_input):
        """Test with many NaN patches (keep only few patches)."""
        with torch.no_grad():
            patches = vit_model.patch_embed(sample_input)
            num_patches = patches.shape[1]
            keep_patches = 10  # Keep only 10 patches

            # Set all but 10 patches to NaN (different 10 for each sample)
            for i in range(BATCH_SIZE):
                mask = torch.ones(num_patches, dtype=torch.bool)
                keep_idx = torch.randperm(num_patches)[:keep_patches]
                mask[keep_idx] = False
                patches[i, mask, :] = float("nan")

        output = masked_vit(patches)
        assert not torch.isnan(output).any()


# ============================================================================
# Output Correctness Tests
# ============================================================================


@pytest.mark.unit
class TestOutputCorrectness:
    """Test that outputs are mathematically correct."""

    def test_no_nan_matches_original(self, masked_vit, vit_model, sample_input):
        """Test that output matches original model when no NaNs."""
        masked_vit.eval()
        vit_model.eval()

        with torch.no_grad():
            masked_output = masked_vit(sample_input)
            original_output = vit_model(sample_input)

        # Should be very close (minor floating point differences acceptable)
        torch.testing.assert_close(
            masked_output,
            original_output,
            rtol=1e-4,
            atol=1e-4,
            msg="Output should match original model when no NaN patches",
        )

    def test_nan_output_different_from_original(
        self, masked_vit, vit_model, sample_input
    ):
        """Test that output differs from original when NaNs present."""
        with torch.no_grad():
            patches = vit_model.patch_embed(sample_input)

            # Add NaN patches
            for i in range(BATCH_SIZE):
                patches[i, 10:15, :] = float("nan")

            masked_output = masked_vit(patches)

            # Create non-NaN version by replacing with zeros
            patches_no_nan = patches.clone()
            patches_no_nan = torch.nan_to_num(patches_no_nan, nan=0.0)

            # Forward through original model
            # Need to manually add pos_embed and go through blocks
            x = patches_no_nan
            if hasattr(vit_model, "cls_token"):
                cls_tokens = vit_model.cls_token.expand(BATCH_SIZE, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
            x = x + vit_model.pos_embed
            if hasattr(vit_model, "pos_drop"):
                x = vit_model.pos_drop(x)
            for blk in vit_model.blocks:
                x = blk(x)
            x = vit_model.norm(x)
            if hasattr(vit_model, "head"):
                original_output = vit_model.head(x[:, 0])
            else:
                original_output = x

        # Outputs should be different (masked removes patches, zero-filling doesn't)
        assert not torch.allclose(masked_output, original_output, rtol=1e-3), (
            "Masked output should differ from zero-filled original"
        )

    def test_consistent_output_for_same_nan_pattern(self, masked_vit, sample_patches):
        """Test that same NaN pattern produces same output."""
        masked_vit.eval()

        # Create NaN pattern
        nan_patches = sample_patches.clone()
        for i in range(BATCH_SIZE):
            nan_patches[i, 10:12, :] = float("nan")

        with torch.no_grad():
            output1 = masked_vit(nan_patches.clone())
            output2 = masked_vit(nan_patches.clone())

        torch.testing.assert_close(output1, output2)


# ============================================================================
# Multi-Model Tests
# ============================================================================


@pytest.mark.parametrize("model_name", TIMM_MODELS)
@pytest.mark.unit
class TestMultipleModels:
    """Test across multiple timm models."""

    def test_model_compatibility(self, model_name):
        """Test that model works with various timm architectures."""
        vit = timm.create_model(model_name, pretrained=False)
        masked_vit = EfficientMaskedTimmViT(vit)

        x = torch.randn(2, 3, 224, 224)
        output = masked_vit(x)

        assert output.shape[0] == 2, f"Failed for {model_name}"
        assert not torch.isnan(output).any(), f"NaN in output for {model_name}"

    def test_model_with_nans(self, model_name):
        """Test each model handles NaN patches correctly."""
        vit = timm.create_model(model_name, pretrained=False)
        masked_vit = EfficientMaskedTimmViT(vit)

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            patches = vit.patch_embed(x)
            patches[0, 5:8, :] = float("nan")
            patches[1, 10:13, :] = float("nan")

        output = masked_vit(patches)
        assert not torch.isnan(output).any(), f"Failed NaN handling for {model_name}"


# ============================================================================
# Gradient and Training Tests
# ============================================================================


@pytest.mark.unit
class TestGradientsAndTraining:
    """Test gradient flow and training compatibility."""

    def test_gradient_flow(self, masked_vit, sample_input):
        """Test that gradients flow properly."""
        masked_vit.train()

        # Forward pass
        output = masked_vit(sample_input)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in masked_vit.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients found in model parameters"

    def test_gradient_flow_with_nans(self, masked_vit, vit_model, sample_input):
        """Test gradient flow with NaN patches."""
        masked_vit.train()

        with torch.no_grad():
            patches = vit_model.patch_embed(sample_input)
            patches[:, 10:12, :] = float("nan")

        patches.requires_grad = False  # Patches don't need grad, model params do
        output = masked_vit(patches)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in masked_vit.parameters()
        )
        assert has_grad

    def test_training_step(self, masked_vit, sample_input):
        """Test a full training step."""
        masked_vit.train()
        optimizer = torch.optim.Adam(masked_vit.parameters(), lr=1e-4)

        # Forward
        output = masked_vit(sample_input)
        target = torch.randint(0, 1000, (BATCH_SIZE,))
        loss = nn.functional.cross_entropy(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_sample_batch(self, masked_vit, sample_input):
        """Test with batch size of 1."""
        single_input = sample_input[:1]
        output = masked_vit(single_input)
        assert output.shape[0] == 1

    def test_large_batch(self, masked_vit):
        """Test with larger batch size."""
        large_input = torch.randn(32, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        output = masked_vit(large_input)
        assert output.shape[0] == 32

    def test_partial_nans_in_patch(self, masked_vit, sample_patches):
        """Test when only some elements in a patch are NaN."""
        # Set only some dimensions to NaN (not all)
        sample_patches[0, 10, :128] = float("nan")  # Half the dimensions
        sample_patches[1, 20, :128] = float("nan")
        sample_patches[2, 5, :128] = float("nan")
        sample_patches[3, 15, :128] = float("nan")

        output = masked_vit(sample_patches)
        assert not torch.isnan(output).any()

    def test_eval_mode(self, masked_vit, sample_input):
        """Test model in eval mode."""
        masked_vit.eval()
        with torch.no_grad():
            output = masked_vit(sample_input)
        assert not torch.isnan(output).any()

    def test_different_input_sizes(self, vit_model):
        """Test with different image sizes if model supports it."""
        # Note: Standard ViTs expect fixed size, but test anyway
        masked_vit = EfficientMaskedTimmViT(vit_model)

        try:
            # Some models might support different sizes
            x = torch.randn(2, 3, 384, 384)
            output = masked_vit(x)
            assert output.shape[0] == 2
        except (RuntimeError, AssertionError):
            # Expected for fixed-size models
            pytest.skip("Model doesn't support variable input sizes")


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperMethods:
    """Test internal helper methods."""

    def test_get_num_extra_tokens(self, masked_vit):
        """Test _get_num_extra_tokens method."""
        num_extra = masked_vit._get_num_extra_tokens()
        assert isinstance(num_extra, int)
        assert num_extra >= 0
        assert num_extra <= 2  # Max is cls + dist

    def test_add_extra_tokens(self, masked_vit):
        """Test _add_extra_tokens method."""
        embed_dim = masked_vit.vit.embed_dim
        x = torch.randn(2, 10, embed_dim)
        x_with_tokens = masked_vit._add_extra_tokens(x)

        num_extra = masked_vit._get_num_extra_tokens()
        assert x_with_tokens.shape[1] == x.shape[1] + num_extra


# ============================================================================
# Performance Tests (Optional)
# ============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance benchmarking tests."""

    def test_performance_comparison(self, vit_model, sample_input):
        """Compare performance of masked vs original model."""
        import time

        masked_vit = EfficientMaskedTimmViT(vit_model)
        masked_vit.eval()
        vit_model.eval()

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = masked_vit(sample_input)
                _ = vit_model(sample_input)

        # Benchmark masked model
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = masked_vit(sample_input)
        masked_time = time.time() - start

        # Benchmark original model
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = vit_model(sample_input)
        original_time = time.time() - start

        print(f"\nMasked model: {masked_time:.3f}s")
        print(f"Original model: {original_time:.3f}s")
        print(f"Overhead: {(masked_time / original_time - 1) * 100:.1f}%")

        # Masked should be reasonably fast (within 2x of original)
        assert masked_time < original_time * 2.0, "Masked model is too slow"


# ============================================================================
# Regression Tests
# ============================================================================


@pytest.mark.unit
class TestRegression:
    """Tests to prevent regressions in specific scenarios."""

    def test_deit_distillation_token(self):
        """Test DeiT models with distillation token specifically."""
        deit = timm.create_model("deit_tiny_patch16_224", pretrained=False)
        masked_deit = EfficientMaskedTimmViT(deit)

        x = torch.randn(2, 3, 224, 224)
        output = masked_deit(x)
        # Check actual model structure instead of assuming
        num_extra = masked_deit._get_num_extra_tokens()
        has_cls = hasattr(deit, "cls_token") and deit.cls_token is not None
        has_dist = hasattr(deit, "dist_token") and deit.dist_token is not None
        expected_num = int(has_cls) + int(has_dist)

        assert num_extra == expected_num, (
            f"Expected {expected_num} tokens (cls={has_cls}, dist={has_dist}), got {num_extra}"
        )
        assert not torch.isnan(output).any()

    def test_very_small_keep_set(self, vit_model):
        """Test with very few patches kept (edge case)."""
        masked_vit = EfficientMaskedTimmViT(vit_model)

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            patches = vit_model.patch_embed(x)

            # Keep only 3 patches
            patches[:, 3:, :] = float("nan")

        output = masked_vit(patches)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
