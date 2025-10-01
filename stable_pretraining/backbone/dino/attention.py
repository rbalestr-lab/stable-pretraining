# Copyright (c) stable-pretraining contributors
# Licensed under the MIT License
#
# This file implements memory-efficient attention for DINOv2.
# Inspired by concepts from DINOv2: https://github.com/facebookresearch/dinov2

"""Memory-efficient attention with xFormers support for DINOv2.

This module provides attention layers that can use xFormers' memory_efficient_attention
when available, enabling efficient sequence packing with non-materialized masks.

Key features:
- Drop-in replacement for standard attention
- Supports attn_bias parameter (for BlockDiagonalMask)
- Graceful fallback to PyTorch SDPA when xFormers unavailable
- Compatible with timm's attention API
"""

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger as logging

# Check for xFormers availability
try:
    from xformers.ops import memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
    logging.info("xFormers available for memory-efficient attention")
except ImportError:
    XFORMERS_AVAILABLE = False
    logging.info("xFormers not available - will use PyTorch SDPA fallback")


class MemEffAttention(nn.Module):
    """Memory-efficient multi-head attention with xFormers support.

    This attention layer can use xFormers' memory_efficient_attention when available,
    which enables efficient sequence packing with non-materialized masks (attn_bias).
    Falls back to PyTorch's scaled_dot_product_attention when xFormers is not available.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to QKV projection
        proj_bias: Whether to add bias to output projection
        attn_drop: Attention dropout rate (only used in fallback mode)
        proj_drop: Output projection dropout rate

    Example:
        >>> # Single-view (no mask)
        >>> attn = MemEffAttention(dim=384, num_heads=6)
        >>> out = attn(x)  # [B, N, 384]
        >>>
        >>> # Multi-view with sequence packing (requires xFormers)
        >>> from xformers.ops import fmha
        >>> attn_bias = fmha.BlockDiagonalMask.from_seqlens([197, 197, 37, ...])
        >>> out = attn(x_packed, attn_bias=attn_bias)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # QKV projection (single linear layer for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Attention dropout (only used in fallback mode)
        self.attn_drop = attn_drop

        logging.debug(
            f"MemEffAttention: dim={dim}, num_heads={num_heads}, "
            f"head_dim={self.head_dim}, xformers={XFORMERS_AVAILABLE}"
        )

    def init_weights(
        self,
        init_attn_std: Optional[float] = None,
        init_proj_std: Optional[float] = None,
        factor: float = 1.0,
    ):
        """Initialize weights with scaled normal distribution.

        Args:
            init_attn_std: Standard deviation for QKV weights (default: dim^-0.5)
            init_proj_std: Standard deviation for proj weights (default: init_attn_std * factor)
            factor: Scale factor for projection initialization
        """
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor

        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)

        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        """Forward pass with optional attention bias.

        Args:
            x: Input tensor [B, N, C]
            attn_bias: Optional attention bias (BlockDiagonalMask or similar).
                       Requires xFormers when not None.

        Returns:
            Output tensor [B, N, C]
        """
        B, N, C = x.shape

        # QKV projection: [B, N, C] -> [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if XFORMERS_AVAILABLE:
            # Use xFormers memory-efficient attention
            # unbind is more efficient than indexing for xFormers
            q, k, v = unbind(qkv, 2)  # Each: [B, N, num_heads, head_dim]

            # xFormers expects [B, N, num_heads, head_dim] layout
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])

        else:
            # Fallback to PyTorch SDPA
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")

            # PyTorch SDPA expects [B, num_heads, N, head_dim]
            q, k, v = torch.unbind(qkv, 2)
            q = q.transpose(1, 2)  # [B, num_heads, N, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            x = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_drop if self.training else 0.0,
            )

            x = x.transpose(1, 2).contiguous().view(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def create_mem_eff_attention(
    dim: int,
    num_heads: int = 8,
    qkv_bias: bool = False,
    **kwargs,
) -> MemEffAttention:
    """Factory function to create memory-efficient attention layer.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to QKV projection
        **kwargs: Additional arguments passed to MemEffAttention

    Returns:
        MemEffAttention instance

    Example:
        >>> attn = create_mem_eff_attention(dim=384, num_heads=6, qkv_bias=True)
    """
    return MemEffAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, **kwargs)
