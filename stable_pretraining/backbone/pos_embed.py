"""Positional embedding utilities for vision transformers."""

import torch
import torch.nn.functional as F
from typing import Literal
import math

__all__ = [
    "get_sincos_pos_embed",
    "get_1d_sincos_pos_embed",
    "get_2d_sincos_pos_embed",
    "interpolate_pos_embed",
    "get_timestep_embed",
]


def get_timestep_embed(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Generate sinusoidal embeddings for continuous timesteps.

    Unlike positional embeddings for sequences, this embeds scalar timestep values.
    Used for diffusion/flow matching time conditioning.
    :param t: Timestep values (B,) or (B, 1), typically in [0, 1]
    :param dim: Embedding dimension
    :param max_period: Maximum period for frequency scaling
    :return: Timestep embeddings of shape (B, dim)
    """
    t = t.view(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / half
    )
    args = t[:, None] * freqs[None, :]
    embedding = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def get_1d_sincos_pos_embed(
    embed_dim: int,
    length: int,
    cls_token: bool = False,
) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings.

    :param embed_dim: Embedding dimension
    :param length: Sequence length (number of positions)
    :param cls_token: If True, prepend a zero embedding for CLS token
    :return: Positional embeddings of shape (length, embed_dim) or
             (length + 1, embed_dim) if cls_token=True
    """
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, embed_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (dim / embed_dim))

    pe = torch.zeros(length, embed_dim)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq[: embed_dim // 2])

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings for image patches.

    :param embed_dim: Embedding dimension (must be divisible by 4)
    :param grid_size: Grid height/width as int (square) or (height, width) tuple
    :param cls_token: If True, prepend a zero embedding for CLS token
    :return: Positional embeddings of shape (H*W, embed_dim) or
             (H*W + 1, embed_dim) if cls_token=True
    """
    if embed_dim <= 0 or embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be positive and divisible by 4, got {embed_dim}"
        )

    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid dimensions must be positive, got ({grid_h}, {grid_w})")

    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)

    dim = embed_dim // 4
    omega = torch.arange(dim, dtype=torch.float32) / dim
    omega = 1.0 / (10000**omega)

    out_h = grid[:, 0:1] @ omega.unsqueeze(0)
    out_w = grid[:, 1:2] @ omega.unsqueeze(0)

    pe = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)],
        dim=1,
    )

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe


def get_sincos_pos_embed(
    embed_dim: int,
    num_patches: int,
    mode: Literal["1d", "2d"] = "1d",
    grid_size: int | tuple[int, int] | None = None,
    cls_token: bool = False,
) -> torch.Tensor:
    """Unified interface for generating sinusoidal positional embeddings.

    :param embed_dim: Embedding dimension
    :param num_patches: Total number of patches (used for 1d mode)
    :param mode: Embedding type - '1d' for sequence, '2d' for image grid
    :param grid_size: Required for '2d' mode
    :param cls_token: If True, prepend a zero embedding for CLS token
    :return: Positional embeddings tensor
    """
    if mode == "2d":
        if grid_size is None:
            raise ValueError("grid_size is required for 2d mode")
        return get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)
    return get_1d_sincos_pos_embed(embed_dim, num_patches, cls_token)


def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    src_size: tuple[int, int],
    tgt_size: tuple[int, int],
    num_prefix_tokens: int = 0,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Interpolate positional embeddings to a new grid size.

    :param pos_embed: Original positional embeddings of shape
                      (1, num_prefix + src_h*src_w, embed_dim) or
                      (num_prefix + src_h*src_w, embed_dim)
    :param src_size: Source grid size as (height, width)
    :param tgt_size: Target grid size as (height, width)
    :param num_prefix_tokens: Number of prefix tokens (CLS, registers) to preserve
    :param mode: Interpolation mode ('nearest', 'bilinear', 'bicubic', 'area')
    :return: Interpolated positional embeddings

    Example::

        old_pos = model.pos_embed  # (1, 197, 768) = 1 + 14*14
        new_pos = interpolate_pos_embed(
            old_pos, src_size=(14, 14), tgt_size=(16, 16), num_prefix_tokens=1
        )  # (1, 257, 768) = 1 + 16*16
    """
    if pos_embed.dim() not in (2, 3):
        raise ValueError(f"pos_embed must be 2D or 3D, got {pos_embed.dim()}D")

    src_h, src_w = src_size
    tgt_h, tgt_w = tgt_size

    if src_h <= 0 or src_w <= 0 or tgt_h <= 0 or tgt_w <= 0:
        raise ValueError(
            f"All grid dims must be positive, src={src_size}, tgt={tgt_size}"
        )

    squeeze_output = False
    if pos_embed.dim() == 2:
        pos_embed = pos_embed.unsqueeze(0)
        squeeze_output = True

    expected_src_len = num_prefix_tokens + src_h * src_w
    if pos_embed.shape[1] != expected_src_len:
        raise ValueError(
            f"pos_embed length {pos_embed.shape[1]} doesn't match expected {expected_src_len}"
        )

    if src_h == tgt_h and src_w == tgt_w:
        return pos_embed.squeeze(0) if squeeze_output else pos_embed

    prefix_pos = pos_embed[:, :num_prefix_tokens, :]
    patch_pos = pos_embed[:, num_prefix_tokens:, :]

    embed_dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, src_h, src_w, embed_dim).permute(0, 3, 1, 2)

    patch_pos = F.interpolate(
        patch_pos,
        size=(tgt_h, tgt_w),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )

    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, tgt_h * tgt_w, embed_dim)
    result = torch.cat([prefix_pos, patch_pos], dim=1)

    return result.squeeze(0) if squeeze_output else result
