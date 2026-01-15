"""Unit tests for positional embedding utilities."""

import pytest
import torch

from stable_pretraining.backbone.pos_embed import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_sincos_pos_embed,
)


@pytest.mark.unit
class TestGet1DSincosEmbed:
    """Test 1d sincos embed."""

    def test_output_shape(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert pe.shape == (100, 64)

    def test_output_shape_with_cls_token(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100, cls_token=True)
        assert pe.shape == (101, 64)

    def test_cls_token_is_zeros(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100, cls_token=True)
        assert torch.allclose(pe[0], torch.zeros(64))

    def test_dtype_is_float32(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert pe.dtype == torch.float32

    def test_values_bounded(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_different_positions_have_different_embeddings(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=10)
        # All rows should be unique
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(pe[i], pe[j])

    def test_deterministic(self):
        pe1 = get_1d_sincos_pos_embed(embed_dim=64, length=50)
        pe2 = get_1d_sincos_pos_embed(embed_dim=64, length=50)
        assert torch.allclose(pe1, pe2)


@pytest.mark.unit
class TestGet2DSincosEmbed:
    """Test 2d sincos embed."""

    def test_output_shape(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=14)
        assert pe.shape == (196, 64)  # 14*14 = 196

    def test_output_shape_with_cls_token(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=14, cls_token=True)
        assert pe.shape == (197, 64)

    def test_cls_token_is_zeros(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7, cls_token=True)
        assert torch.allclose(pe[0], torch.zeros(64))

    def test_requires_divisible_by_4(self):
        with pytest.raises(ValueError):
            get_2d_sincos_pos_embed(embed_dim=65, grid_size=7)

    def test_values_bounded(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_deterministic(self):
        pe1 = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        pe2 = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        assert torch.allclose(pe1, pe2)


@pytest.mark.unit
class TestGetSincosEmbed:
    """Test get sincos embed."""

    def test_1d_mode(self):
        pe = get_sincos_pos_embed(embed_dim=64, num_patches=100, mode="1d")
        expected = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert torch.allclose(pe, expected)

    def test_2d_mode(self):
        pe = get_sincos_pos_embed(embed_dim=64, num_patches=49, mode="2d", grid_size=7)
        expected = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        assert torch.allclose(pe, expected)

    def test_2d_mode_requires_grid_size(self):
        with pytest.raises(ValueError):
            get_sincos_pos_embed(embed_dim=64, num_patches=49, mode="2d")

    def test_cls_token_passthrough(self):
        pe = get_sincos_pos_embed(
            embed_dim=64, num_patches=100, mode="1d", cls_token=True
        )
        assert pe.shape == (101, 64)
