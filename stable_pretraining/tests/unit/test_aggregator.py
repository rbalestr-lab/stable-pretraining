"""Unit tests for TensorAggregator module."""

import pytest
import torch
import torch.nn as nn
from loguru import logger
from stable_pretraining.backbone import TensorAggregator

# Disable logging during tests
logger.disable("__main__")


# Import the module to test
# from tensor_aggregator import TensorAggregator, AggregatorMLP
# For now, assume the class is available
# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def feature_dim():
    """Standard feature dimension."""
    return 768


@pytest.fixture
def seq_len():
    """Standard sequence length."""
    return 197


# ============================================================================
# Initialization Tests
# ============================================================================
@pytest.mark.unit
class TestInitialization:
    """Test TensorAggregator initialization."""

    def test_single_mode_string(self):
        """Test initialization with single string mode."""
        agg = TensorAggregator("mean")

        assert agg.input_type == "single"
        assert agg.agg_modes == {"default": "mean"}
        assert agg.adaptive_pool_size == 1

    def test_list_of_modes(self):
        """Test initialization with list of modes."""
        agg = TensorAggregator(["cls", "mean", "max"])

        assert agg.input_type == "list"
        assert agg.agg_modes == {0: "cls", 1: "mean", 2: "max"}

    def test_dict_of_modes(self):
        """Test initialization with dict of modes."""
        agg = TensorAggregator({"layer1": "cls", "layer2": "mean"})

        assert agg.input_type == "dict"
        assert agg.agg_modes == {"layer1": "cls", "layer2": "mean"}

    def test_adaptive_pool_size(self):
        """Test custom adaptive pool size."""
        agg = TensorAggregator("adaptive", adaptive_pool_size=2)

        assert agg.adaptive_pool_size == 2

    def test_invalid_mode_raises_error(self):
        """Test that invalid aggregation mode raises error."""
        with pytest.raises(ValueError, match="Invalid aggregation mode"):
            TensorAggregator("invalid_mode")

    def test_invalid_input_spec_type(self):
        """Test that invalid input_spec type raises error."""
        with pytest.raises(ValueError, match="Invalid input_spec type"):
            TensorAggregator(123)


# ============================================================================
# 2D Tensor Tests (No-op)
# ============================================================================
@pytest.mark.unit
class Test2DTensors:
    """Test that 2D tensors are returned unchanged."""

    def test_2d_tensor_noop(self, batch_size, feature_dim):
        """Test that 2D tensor is returned as-is."""
        agg = TensorAggregator("mean")
        x = torch.randn(batch_size, feature_dim)

        out = agg(x)

        assert out.shape == (batch_size, feature_dim)
        assert torch.equal(out, x)

    def test_2d_tensor_any_mode(self, batch_size, feature_dim):
        """Test that 2D tensor is unchanged regardless of mode."""
        for mode in ["mean", "max", "cls", "flatten", "adaptive"]:
            agg = TensorAggregator(mode)
            x = torch.randn(batch_size, feature_dim)

            out = agg(x)

            assert out.shape == x.shape
            assert torch.equal(out, x)


# ============================================================================
# 3D Tensor Tests (Sequence Data)
# ============================================================================
@pytest.mark.unit
class Test3DTensors:
    """Test aggregation of 3D tensors (B, L, D)."""

    def test_mean_pooling_3d(self, batch_size, seq_len, feature_dim):
        """Test mean pooling over sequence dimension."""
        agg = TensorAggregator("mean")
        x = torch.randn(batch_size, seq_len, feature_dim)

        out = agg(x)

        assert out.shape == (batch_size, feature_dim)
        expected = x.mean(dim=1)
        assert torch.allclose(out, expected)

    def test_max_pooling_3d(self, batch_size, seq_len, feature_dim):
        """Test max pooling over sequence dimension."""
        agg = TensorAggregator("max")
        x = torch.randn(batch_size, seq_len, feature_dim)

        out = agg(x)

        assert out.shape == (batch_size, feature_dim)
        expected = x.max(dim=1)[0]
        assert torch.allclose(out, expected)

    def test_cls_token_3d(self, batch_size, seq_len, feature_dim):
        """Test CLS token extraction (first token)."""
        agg = TensorAggregator("cls")
        x = torch.randn(batch_size, seq_len, feature_dim)

        out = agg(x)

        assert out.shape == (batch_size, feature_dim)
        expected = x[:, 0]
        assert torch.equal(out, expected)

    def test_flatten_3d(self, batch_size, seq_len, feature_dim):
        """Test flattening of 3D tensor."""
        agg = TensorAggregator("flatten")
        x = torch.randn(batch_size, seq_len, feature_dim)

        out = agg(x)

        assert out.shape == (batch_size, seq_len * feature_dim)
        expected = x.reshape(batch_size, -1)
        assert torch.equal(out, expected)

    def test_adaptive_pooling_3d(self, batch_size, seq_len, feature_dim):
        """Test adaptive pooling on 3D tensor."""
        agg = TensorAggregator("adaptive", adaptive_pool_size=1)
        x = torch.randn(batch_size, seq_len, feature_dim)

        out = agg(x)

        assert out.shape == (batch_size, feature_dim)

    def test_adaptive_pooling_3d_size_2(self, batch_size, seq_len, feature_dim):
        """Test adaptive pooling with size > 1."""
        pool_size = 2
        agg = TensorAggregator("adaptive", adaptive_pool_size=pool_size)
        x = torch.randn(batch_size, seq_len, feature_dim)

        out = agg(x)

        assert out.shape == (batch_size, feature_dim * pool_size)


# ============================================================================
# 4D Tensor Tests (Image/Feature Maps)
# ============================================================================
@pytest.mark.unit
class Test4DTensors:
    """Test aggregation of 4D tensors (B, C, H, W)."""

    def test_mean_pooling_4d(self, batch_size):
        """Test spatial mean pooling."""
        agg = TensorAggregator("mean")
        x = torch.randn(batch_size, 512, 14, 14)

        out = agg(x)

        assert out.shape == (batch_size, 512)
        expected = x.mean(dim=(2, 3))
        assert torch.allclose(out, expected)

    def test_max_pooling_4d(self, batch_size):
        """Test spatial max pooling."""
        agg = TensorAggregator("max")
        x = torch.randn(batch_size, 512, 14, 14)

        out = agg(x)

        assert out.shape == (batch_size, 512)
        expected = x.amax(dim=(2, 3))
        assert torch.allclose(out, expected)

    def test_flatten_4d(self, batch_size):
        """Test flattening of 4D tensor."""
        channels, height, width = 512, 14, 14
        agg = TensorAggregator("flatten")
        x = torch.randn(batch_size, channels, height, width)

        out = agg(x)

        assert out.shape == (batch_size, channels * height * width)

    def test_adaptive_pooling_4d(self, batch_size):
        """Test adaptive pooling on 4D tensor."""
        agg = TensorAggregator("adaptive", adaptive_pool_size=1)
        x = torch.randn(batch_size, 512, 14, 14)

        out = agg(x)

        assert out.shape == (batch_size, 512)

    def test_adaptive_pooling_4d_size_2(self, batch_size):
        """Test adaptive pooling with size 2x2."""
        pool_size = 2
        agg = TensorAggregator("adaptive", adaptive_pool_size=pool_size)
        x = torch.randn(batch_size, 512, 14, 14)

        out = agg(x)

        assert out.shape == (batch_size, 512 * pool_size * pool_size)

    def test_cls_mode_4d_warning(self, batch_size):
        """Test that cls mode on 4D tensor works but may warn."""
        agg = TensorAggregator("cls")
        x = torch.randn(batch_size, 512, 14, 14)

        out = agg(x)

        assert out.shape == (batch_size, 512)
        expected = x[:, :, 0, 0]
        assert torch.equal(out, expected)


# ============================================================================
# 5D Tensor Tests (Video/3D Data)
# ============================================================================
@pytest.mark.unit
class Test5DTensors:
    """Test aggregation of 5D tensors (B, C, T, H, W)."""

    def test_mean_pooling_5d(self, batch_size):
        """Test spatiotemporal mean pooling."""
        agg = TensorAggregator("mean")
        x = torch.randn(batch_size, 256, 8, 7, 7)

        out = agg(x)

        assert out.shape == (batch_size, 256)
        expected = x.mean(dim=(2, 3, 4))
        assert torch.allclose(out, expected)

    def test_max_pooling_5d(self, batch_size):
        """Test spatiotemporal max pooling."""
        agg = TensorAggregator("max")
        x = torch.randn(batch_size, 256, 8, 7, 7)

        out = agg(x)

        assert out.shape == (batch_size, 256)
        expected = x.amax(dim=(2, 3, 4))
        assert torch.allclose(out, expected)

    def test_flatten_5d(self, batch_size):
        """Test flattening of 5D tensor."""
        channels, time, height, width = 256, 8, 7, 7
        agg = TensorAggregator("flatten")
        x = torch.randn(batch_size, channels, time, height, width)

        out = agg(x)

        assert out.shape == (batch_size, channels * time * height * width)

    def test_adaptive_pooling_5d(self, batch_size):
        """Test adaptive 3D pooling."""
        agg = TensorAggregator("adaptive", adaptive_pool_size=1)
        x = torch.randn(batch_size, 256, 8, 7, 7)

        out = agg(x)

        assert out.shape == (batch_size, 256)

    def test_unsupported_mode_5d(self, batch_size):
        """Test that cls mode is not supported for 5D tensors."""
        agg = TensorAggregator("cls")
        x = torch.randn(batch_size, 256, 8, 7, 7)

        with pytest.raises(ValueError, match="not supported for 5D"):
            agg(x)


# ============================================================================
# List Input Tests
# ============================================================================
@pytest.mark.unit
class TestListInput:
    """Test aggregation with list of tensors."""

    def test_list_single_mode(self, batch_size, seq_len, feature_dim):
        """Test list input with single aggregation mode for all."""
        agg = TensorAggregator("cls")
        tensors = [
            torch.randn(batch_size, seq_len, feature_dim),
            torch.randn(batch_size, seq_len, feature_dim),
            torch.randn(batch_size, seq_len, feature_dim),
        ]

        out = agg(tensors)

        assert out.shape == (batch_size, feature_dim * 3)

    def test_list_per_tensor_modes(self, batch_size, seq_len, feature_dim):
        """Test list input with different mode per tensor."""
        agg = TensorAggregator(["cls", "mean", "max"])
        tensors = [
            torch.randn(batch_size, seq_len, feature_dim),
            torch.randn(batch_size, seq_len, feature_dim),
            torch.randn(batch_size, seq_len, feature_dim),
        ]

        out = agg(tensors)

        assert out.shape == (batch_size, feature_dim * 3)

        # Verify each part
        expected_cls = tensors[0][:, 0]
        expected_mean = tensors[1].mean(dim=1)
        expected_max = tensors[2].max(dim=1)[0]

        assert torch.equal(out[:, :feature_dim], expected_cls)
        assert torch.allclose(out[:, feature_dim : 2 * feature_dim], expected_mean)
        assert torch.allclose(out[:, 2 * feature_dim :], expected_max)

    def test_list_mixed_dimensions(self, batch_size):
        """Test list with tensors of different dimensions."""
        agg = TensorAggregator(["cls", "mean"])
        tensors = [
            torch.randn(batch_size, 197, 768),  # 3D
            torch.randn(batch_size, 512, 14, 14),  # 4D
        ]

        out = agg(tensors)

        assert out.shape == (batch_size, 768 + 512)

    def test_list_ssl_use_case(self, batch_size):
        """Test SSL use case: last 4 transformer layers."""
        agg = TensorAggregator(["cls", "cls", "cls", "cls"])

        # Simulate last 4 layers of a transformer
        layers = [torch.randn(batch_size, 197, 768) for _ in range(4)]

        out = agg(layers)

        assert out.shape == (batch_size, 768 * 4)


# ============================================================================
# Dict Input Tests
# ============================================================================
@pytest.mark.unit
class TestDictInput:
    """Test aggregation with dict of tensors."""

    def test_dict_single_mode(self, batch_size):
        """Test dict input with single mode for all."""
        agg = TensorAggregator("mean")
        features = {
            "layer1": torch.randn(batch_size, 197, 768),
            "layer2": torch.randn(batch_size, 197, 768),
        }

        out = agg(features)

        assert out.shape == (batch_size, 768 * 2)

    def test_dict_per_key_modes(self, batch_size):
        """Test dict input with different mode per key."""
        agg = TensorAggregator(
            {
                "layer1": "cls",
                "layer2": "mean",
                "conv": "max",
            }
        )
        features = {
            "layer1": torch.randn(batch_size, 197, 768),
            "layer2": torch.randn(batch_size, 197, 768),
            "conv": torch.randn(batch_size, 512, 14, 14),
        }

        out = agg(features)

        assert out.shape == (batch_size, 768 + 768 + 512)

    def test_dict_deterministic_ordering(self, batch_size):
        """Test that dict keys are processed in sorted order."""
        agg = TensorAggregator({"z": "cls", "a": "mean"})
        features = {
            "z": torch.randn(batch_size, 197, 768),
            "a": torch.randn(batch_size, 197, 768),
        }

        out1 = agg(features)
        out2 = agg(features)

        # Should be identical (deterministic)
        assert torch.equal(out1, out2)

    def test_dict_missing_mode(self, batch_size):
        """Test that missing mode for a key uses default (mean)."""
        agg = TensorAggregator({"layer1": "cls"})
        features = {
            "layer1": torch.randn(batch_size, 197, 768),
            "layer2": torch.randn(batch_size, 197, 768),  # No mode specified
        }

        out = agg(features)

        # Should still work with warning
        assert out.shape == (batch_size, 768 * 2)


# ============================================================================
# Output Dimension Computation Tests
# ============================================================================
@pytest.mark.unit
class TestOutputDimComputation:
    """Test compute_output_dim method."""

    def test_single_tensor_shape(self):
        """Test output dim for single tensor."""
        agg = TensorAggregator("mean")

        # 3D: (seq_len, features)
        dim = agg.compute_output_dim((197, 768))
        assert dim == 768

        # 4D: (channels, height, width)
        dim = agg.compute_output_dim((512, 14, 14))
        assert dim == 512

    def test_list_shapes(self):
        """Test output dim for list of shapes."""
        agg = TensorAggregator(["cls", "mean", "max"])

        shapes = [(197, 768), (197, 768), (197, 768)]
        dim = agg.compute_output_dim(shapes)

        assert dim == 768 * 3

    def test_dict_shapes(self):
        """Test output dim for dict of shapes."""
        agg = TensorAggregator(
            {
                "layer1": "cls",
                "conv": "mean",
            }
        )

        shapes = {
            "layer1": (197, 768),
            "conv": (512, 14, 14),
        }
        dim = agg.compute_output_dim(shapes)

        assert dim == 768 + 512

    def test_flatten_output_dim(self):
        """Test output dim for flatten mode."""
        agg = TensorAggregator("flatten")

        # 3D
        dim = agg.compute_output_dim((197, 768))
        assert dim == 197 * 768

        # 4D
        dim = agg.compute_output_dim((512, 14, 14))
        assert dim == 512 * 14 * 14

    def test_adaptive_output_dim(self):
        """Test output dim for adaptive mode."""
        agg = TensorAggregator("adaptive", adaptive_pool_size=2)

        # 3D
        dim = agg.compute_output_dim((197, 768))
        assert dim == 768 * 2

        # 4D
        dim = agg.compute_output_dim((512, 14, 14))
        assert dim == 512 * 4  # 2x2


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_batch(self):
        """Test with batch size of 1."""
        agg = TensorAggregator("mean")
        x = torch.randn(1, 197, 768)

        out = agg(x)

        assert out.shape == (1, 768)

    def test_large_batch(self):
        """Test with large batch size."""
        agg = TensorAggregator("cls")
        x = torch.randn(128, 197, 768)

        out = agg(x)

        assert out.shape == (128, 768)

    def test_small_spatial_dims(self):
        """Test with small spatial dimensions."""
        agg = TensorAggregator("mean")
        x = torch.randn(4, 512, 1, 1)

        out = agg(x)

        assert out.shape == (4, 512)

    def test_unsupported_tensor_dimension(self):
        """Test that unsupported tensor dimensions raise error."""
        agg = TensorAggregator("mean")
        x = torch.randn(4, 3, 2, 1, 5, 6)  # 6D tensor

        with pytest.raises(ValueError, match="Unsupported tensor dimension"):
            agg(x)

    def test_empty_list_raises_error(self):
        """Test that empty list raises appropriate error."""
        agg = TensorAggregator(["cls", "mean"])

        # torch.cat on empty list raises RuntimeError or ValueError depending on PyTorch version
        with pytest.raises(
            (RuntimeError, ValueError), match="expected a non-empty list"
        ):
            agg([])

    def test_wrong_input_type_raises_error(self):
        """Test that wrong input type raises error."""
        agg = TensorAggregator("mean")

        with pytest.raises(TypeError, match="Unsupported input type"):
            agg("not a tensor")


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.unit
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_ssl_linear_probe_workflow(self):
        """Test complete SSL linear probe workflow."""
        batch_size = 32
        num_layers = 4
        seq_len = 197
        feature_dim = 768
        num_classes = 1000

        # Create aggregator for last 4 layers
        agg = TensorAggregator(["cls"] * num_layers)

        # Simulate transformer layer outputs
        layers = [
            torch.randn(batch_size, seq_len, feature_dim) for _ in range(num_layers)
        ]

        # Aggregate
        features = agg(layers)
        assert features.shape == (batch_size, feature_dim * num_layers)

        # Simple linear classifier
        classifier = nn.Linear(feature_dim * num_layers, num_classes)
        logits = classifier(features)

        assert logits.shape == (batch_size, num_classes)

    def test_multi_scale_feature_fusion(self):
        """Test multi-scale feature fusion workflow."""
        batch_size = 16

        # Different resolution features
        agg = TensorAggregator(
            {
                "stage1": "mean",
                "stage2": "mean",
                "stage3": "mean",
                "stage4": "mean",
            }
        )

        features = {
            "stage1": torch.randn(batch_size, 256, 56, 56),
            "stage2": torch.randn(batch_size, 512, 28, 28),
            "stage3": torch.randn(batch_size, 1024, 14, 14),
            "stage4": torch.randn(batch_size, 2048, 7, 7),
        }

        fused = agg(features)
        expected_dim = 256 + 512 + 1024 + 2048

        assert fused.shape == (batch_size, expected_dim)

    def test_mixed_modalities(self):
        """Test aggregating different modality features."""
        batch_size = 8

        agg = TensorAggregator(
            {
                "text": "cls",
                "image": "mean",
                "audio": "max",
            }
        )

        # Carefully construct shapes so dimensions match expectations
        features = {
            "text": torch.randn(
                batch_size, 512, 768
            ),  # 3D: (B, L, D) -> cls -> (B, 768)
            "image": torch.randn(
                batch_size, 512, 14, 14
            ),  # 4D: (B, C, H, W) -> mean -> (B, 512)
            "audio": torch.randn(
                batch_size, 100, 256
            ),  # 3D: (B, L, D) -> max -> (B, 256)
        }

        out = agg(features)

        # Expected: 768 (text) + 512 (image) + 256 (audio) = 1536
        assert out.shape == (batch_size, 768 + 512 + 256)


# ============================================================================
# Parameterized Tests
# ============================================================================
@pytest.mark.unit
class TestParameterized:
    """Parameterized tests for various configurations."""

    @pytest.mark.parametrize("mode", ["mean", "max", "cls", "flatten", "adaptive"])
    def test_all_modes_3d(self, mode, batch_size, seq_len, feature_dim):
        """Test all aggregation modes on 3D tensors."""
        agg = TensorAggregator(mode)
        x = torch.randn(batch_size, seq_len, feature_dim)

        out = agg(x)

        assert out.ndim == 2
        assert out.shape[0] == batch_size

    @pytest.mark.parametrize("mode", ["mean", "max", "flatten", "adaptive"])
    def test_all_modes_4d(self, mode, batch_size):
        """Test supported aggregation modes on 4D tensors."""
        agg = TensorAggregator(mode)
        x = torch.randn(batch_size, 512, 14, 14)

        out = agg(x)

        assert out.ndim == 2
        assert out.shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_various_batch_sizes(self, batch_size):
        """Test with various batch sizes."""
        agg = TensorAggregator("mean")
        x = torch.randn(batch_size, 197, 768)

        out = agg(x)

        assert out.shape == (batch_size, 768)

    @pytest.mark.parametrize("num_layers", [1, 2, 4, 8, 12])
    def test_various_layer_counts(self, num_layers, batch_size):
        """Test with various numbers of layers."""
        agg = TensorAggregator(["cls"] * num_layers)
        layers = [torch.randn(batch_size, 197, 768) for _ in range(num_layers)]

        out = agg(layers)

        assert out.shape == (batch_size, 768 * num_layers)
