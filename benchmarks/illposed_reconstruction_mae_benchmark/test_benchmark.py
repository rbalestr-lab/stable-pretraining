#!/usr/bin/env python3
"""
Unit tests for the illposed_reconstruction_mae_benchmark.

Following stable-pretraining testing guidelines (TESTING.md):
- Fast unit tests (no GPU, no downloads, no forward passes)
- Test individual components in isolation
- Use small tensors and mock data
- Each test should run in < 1 second

For proper testing with pytest, run:
    python -m pytest benchmarks/illposed_reconstruction_mae_benchmark/test_benchmark.py -v -m unit

For quick testing without pytest:
    python benchmarks/illposed_reconstruction_mae_benchmark/test_benchmark.py
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.illposed_reconstruction_mae_benchmark.dataset_configs import (
    get_dataset_config,
    list_available_datasets,
    DATASET_CONFIGS,
)
from benchmarks.illposed_reconstruction_mae_benchmark.utils import (
    create_mae_with_custom_decoder,
    parse_decoder_type,
    DECODER_DIM_MAP,
)


# Test tracking
tests_passed = 0
tests_failed = 0


def test(name):
    """Decorator to mark and run test functions."""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed
            try:
                func()
                tests_passed += 1
                print(f"✓ {name}")
            except AssertionError as e:
                tests_failed += 1
                print(f"✗ {name}: {e}")
            except Exception as e:
                tests_failed += 1
                print(f"✗ {name}: Unexpected error: {e}")
        return wrapper
    return decorator


# =============================================================================
# Dataset Configuration Tests
# =============================================================================

@test("test_list_available_datasets")
def test_list_available_datasets():
    """Test that all expected datasets are listed."""
    datasets = list_available_datasets()
    assert len(datasets) == 4
    assert "imagenette" in datasets
    assert "tiny-imagenet" in datasets
    assert "imagenet100" in datasets
    assert "imagenet" in datasets


@test("test_dataset_config_structure")
def test_dataset_config_structure():
    """Test that each dataset config has required fields."""
    required_fields = ["hf_path", "num_classes", "img_size", "splits", "hf_config"]

    for dataset_name in list_available_datasets():
        config = get_dataset_config(dataset_name)

        for field in required_fields:
            assert field in config, f"Dataset {dataset_name} missing field {field}"

        # Validate types
        assert isinstance(config["hf_path"], str)
        assert isinstance(config["num_classes"], int)
        assert isinstance(config["img_size"], int)
        assert isinstance(config["splits"], dict)
        assert config["hf_path"].startswith("randall-lab/")
        assert config["num_classes"] > 0
        assert config["img_size"] > 0


@test("test_dataset_config_splits")
def test_dataset_config_splits():
    """Test that dataset split configurations are valid."""
    for dataset_name in list_available_datasets():
        config = get_dataset_config(dataset_name)
        splits = config["splits"]

        assert "train" in splits, f"Dataset {dataset_name} missing train split"
        assert "val" in splits, f"Dataset {dataset_name} missing val split"
        assert isinstance(splits["train"], str)
        assert isinstance(splits["val"], str)


@test("test_get_dataset_config_invalid")
def test_get_dataset_config_invalid():
    """Test that invalid dataset names raise ValueError."""
    try:
        get_dataset_config("nonexistent_dataset")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


# =============================================================================
# Decoder Configuration Tests
# =============================================================================

@test("test_parse_decoder_type_linear")
def test_parse_decoder_type_linear():
    """Test linear decoder configuration parsing."""
    config = parse_decoder_type("linear")
    assert config["type"] == "linear"
    assert "embed_dim" not in config
    assert "depth" not in config


@test("test_parse_decoder_type_transformer")
def test_parse_decoder_type_transformer():
    """Test transformer decoder configuration parsing."""
    test_cases = [
        ("tiny-4b", {"type": "transformer", "embed_dim": 192, "depth": 4, "num_heads": 3}),
        ("small-6b", {"type": "transformer", "embed_dim": 384, "depth": 6, "num_heads": 6}),
        ("base-8b", {"type": "transformer", "embed_dim": 512, "depth": 8, "num_heads": 8}),
        ("large-10b", {"type": "transformer", "embed_dim": 768, "depth": 10, "num_heads": 12}),
    ]

    for decoder_str, expected in test_cases:
        config = parse_decoder_type(decoder_str)
        assert config["type"] == expected["type"], f"{decoder_str}: type mismatch"
        assert config["embed_dim"] == expected["embed_dim"], f"{decoder_str}: embed_dim mismatch"
        assert config["depth"] == expected["depth"], f"{decoder_str}: depth mismatch"
        assert config["num_heads"] == expected["num_heads"], f"{decoder_str}: num_heads mismatch"


@test("test_parse_decoder_type_invalid")
def test_parse_decoder_type_invalid():
    """Test that invalid decoder types raise ValueError."""
    invalid_cases = ["invalid", "tiny", "8b", "unknown-8b", "tiny-8"]

    for invalid_str in invalid_cases:
        try:
            parse_decoder_type(invalid_str)
            assert False, f"Should have raised ValueError for {invalid_str}"
        except ValueError:
            pass  # Expected


# =============================================================================
# Model Creation Tests
# =============================================================================

@test("test_create_mae_default_configs")
def test_create_mae_default_configs():
    """Test that default model configurations create valid models."""
    model_sizes = ["vit_tiny", "vit_small", "vit_base"]
    decoder_config = parse_decoder_type("base-8b")

    for model_name in model_sizes:
        model = create_mae_with_custom_decoder(model_name, decoder_config)

        # Verify model has required components
        assert hasattr(model, "patch_embed")
        assert hasattr(model, "blocks")
        assert hasattr(model, "decoder_blocks")
        assert hasattr(model, "decoder_embed")

        # Verify architecture dimensions
        assert len(model.blocks) > 0, f"{model_name} should have encoder blocks"
        assert len(model.decoder_blocks) > 0, f"{model_name} should have decoder blocks"


@test("test_create_mae_encoder_overrides")
def test_create_mae_encoder_overrides():
    """Test encoder parameter overrides."""
    decoder_config = parse_decoder_type("base-8b")

    # Test embed_dim override (256 is divisible by 4)
    encoder_overrides = {"embed_dim": 256, "num_heads": 4}
    model = create_mae_with_custom_decoder("vit_tiny", decoder_config, encoder_overrides)
    assert model.patch_embed.proj.out_channels == 256

    # Test depth override
    encoder_overrides = {"depth": 6}
    model = create_mae_with_custom_decoder("vit_tiny", decoder_config, encoder_overrides)
    assert len(model.blocks) == 6

    # Test combined overrides (320 is divisible by 5)
    encoder_overrides = {"embed_dim": 320, "depth": 8, "num_heads": 5}
    model = create_mae_with_custom_decoder("vit_tiny", decoder_config, encoder_overrides)
    assert model.patch_embed.proj.out_channels == 320
    assert len(model.blocks) == 8


@test("test_create_mae_decoder_overrides")
def test_create_mae_decoder_overrides():
    """Test decoder parameter overrides."""
    test_cases = [
        ("tiny-4b", 192, 4),
        ("small-6b", 384, 6),
        ("base-8b", 512, 8),
        ("large-10b", 768, 10),
    ]

    for decoder_str, expected_dim, expected_depth in test_cases:
        decoder_config = parse_decoder_type(decoder_str)
        model = create_mae_with_custom_decoder("vit_tiny", decoder_config)

        assert model.decoder_embed.out_features == expected_dim
        assert len(model.decoder_blocks) == expected_depth


@test("test_create_mae_linear_decoder")
def test_create_mae_linear_decoder():
    """Test linear decoder creation."""
    decoder_config = parse_decoder_type("linear")

    model_sizes = ["vit_tiny", "vit_small", "vit_base"]
    for model_name in model_sizes:
        model = create_mae_with_custom_decoder(model_name, decoder_config)

        # Verify linear decoder exists and has correct type
        assert hasattr(model, "linear_decoder")
        assert isinstance(model.linear_decoder, torch.nn.Linear)

        # Verify dimensions
        in_features = model.linear_decoder.in_features
        out_features = model.linear_decoder.out_features
        assert in_features > 0
        assert out_features > 0  # Should be patch_size^2 * 3


@test("test_create_mae_patch_size_override")
def test_create_mae_patch_size_override():
    """Test patch size override."""
    decoder_config = parse_decoder_type("base-8b")

    for patch_size in [8, 16, 32]:
        encoder_overrides = {"patch_size": patch_size}
        model = create_mae_with_custom_decoder("vit_tiny", decoder_config, encoder_overrides)

        # Verify patch size
        actual_patch_size = model.patch_embed.patch_size[0]
        assert actual_patch_size == patch_size


@test("test_create_mae_img_size_override")
def test_create_mae_img_size_override():
    """Test image size override for different datasets."""
    decoder_config = parse_decoder_type("base-8b")

    # Test with tiny-imagenet size (64x64)
    encoder_overrides = {"img_size": 64}
    model = create_mae_with_custom_decoder("vit_tiny", decoder_config, encoder_overrides)
    assert hasattr(model, "patch_embed")

    # Test with standard imagenet size (224x224)
    encoder_overrides = {"img_size": 224}
    model = create_mae_with_custom_decoder("vit_tiny", decoder_config, encoder_overrides)
    assert hasattr(model, "patch_embed")


@test("test_create_mae_invalid_model")
def test_create_mae_invalid_model():
    """Test that invalid model names raise ValueError."""
    decoder_config = parse_decoder_type("base-8b")

    try:
        create_mae_with_custom_decoder("invalid_model", decoder_config)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


@test("test_encoder_decoder_dimension_compatibility")
def test_encoder_decoder_dimension_compatibility():
    """Test that encoder and decoder dimensions are compatible."""
    decoder_config = parse_decoder_type("base-8b")
    # 512 is divisible by 8
    encoder_overrides = {"embed_dim": 512, "num_heads": 8}

    # This should work - decoder_embed linear layer maps encoder_dim to decoder_dim
    model = create_mae_with_custom_decoder("vit_tiny", decoder_config, encoder_overrides)

    encoder_dim = model.patch_embed.proj.out_channels
    decoder_input_dim = model.decoder_embed.in_features

    # Decoder embed should accept encoder output
    assert encoder_dim == decoder_input_dim


@test("test_linear_decoder_with_custom_encoder")
def test_linear_decoder_with_custom_encoder():
    """Test linear decoder works with custom encoder dimensions."""
    decoder_config = parse_decoder_type("linear")

    # Use dimensions that are properly divisible by num_heads
    custom_configs = [
        {"embed_dim": 256, "num_heads": 4},  # 256/4 = 64
        {"embed_dim": 320, "num_heads": 5},  # 320/5 = 64
        {"embed_dim": 512, "num_heads": 8},  # 512/8 = 64
    ]
    for config in custom_configs:
        model = create_mae_with_custom_decoder("vit_tiny", decoder_config, config)

        # Linear decoder input should match encoder output
        assert model.linear_decoder.in_features == config["embed_dim"]


@test("test_decoder_dim_map_completeness")
def test_decoder_dim_map_completeness():
    """Test that DECODER_DIM_MAP has all expected sizes."""
    expected_sizes = ["tiny", "small", "base", "large"]

    for size in expected_sizes:
        assert size in DECODER_DIM_MAP, f"Missing {size} in DECODER_DIM_MAP"
        assert DECODER_DIM_MAP[size] > 0


@test("test_dataset_configs_completeness")
def test_dataset_configs_completeness():
    """Test that DATASET_CONFIGS matches list_available_datasets."""
    listed_datasets = set(list_available_datasets())
    config_datasets = set(DATASET_CONFIGS.keys())

    assert listed_datasets == config_datasets


def main():
    """Run all tests."""
    print("=" * 80)
    print("Running Unit Tests for illposed_reconstruction_mae_benchmark")
    print("=" * 80)
    print()

    # Run all test functions
    test_list_available_datasets()
    test_dataset_config_structure()
    test_dataset_config_splits()
    test_get_dataset_config_invalid()

    test_parse_decoder_type_linear()
    test_parse_decoder_type_transformer()
    test_parse_decoder_type_invalid()

    test_create_mae_default_configs()
    test_create_mae_encoder_overrides()
    test_create_mae_decoder_overrides()
    test_create_mae_linear_decoder()
    test_create_mae_patch_size_override()
    test_create_mae_img_size_override()
    test_create_mae_invalid_model()

    test_encoder_decoder_dimension_compatibility()
    test_linear_decoder_with_custom_encoder()

    test_decoder_dim_map_completeness()
    test_dataset_configs_completeness()

    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print()

    if tests_failed == 0:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"✗ {tests_failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
